#!/usr/bin/env python3
"""
Reproducible mini-benchmark: classical vs quantum vs hybrid optimisers for a
4-qubit variational quantum classifier (VQC), CPU only, pure NumPy simulation.

Optimisers : Adam, SGD (both with exact parameter-shift gradients), COBYLA, SPSA,
             QBO (Bayesian optimisation with a quantum fidelity-kernel GP surrogate)
Baseline   : Logistic regression on the same 16 PCA features (sanity check)
Protocol   : repeated stratified 5-fold CV (default 5 repeats = 25 paired runs)
             feature selection + PCA fitted on the TRAINING part of each fold only
Outputs    : results/runs.csv (one row per optimiser x run), summary.csv,
             stats.csv (Wilcoxon vs QBO, Bonferroni), fig_auc.png, fig_convergence.png

Requires: numpy pandas scipy scikit-learn matplotlib
Usage:
  python verify_optimisers.py --synthetic            # quick test of the pipeline
  python verify_optimisers.py                        # real data: downloads GSE26712
  python verify_optimisers.py --csv my.csv --label-col y   # your own table
NOTE: the simulation is NOISELESS. No hardware, no noise model.
"""
import argparse, gzip, io, os, re, sys, time, urllib.request, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

NQ, NL = 4, 3                    # qubits, layers
NPAR = NQ * NL                   # trainable parameters (12)
KEEP_CENSORED = False
GEO_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE26nnn/GSE26712/matrix/"
           "GSE26712_series_matrix.txt.gz")

# ----------------------------------------------------------------- data ----
def load_geo(cache, cutoff_months, label_hint):
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    if not os.path.exists(cache):
        print("Downloading", GEO_URL)
        urllib.request.urlretrieve(GEO_URL, cache)
    chars, table, intable = [], [], False
    with gzip.open(cache, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                chars.append([c.strip('"') for c in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                intable = True
            elif line.startswith("!series_matrix_table_end"):
                intable = False
            elif intable:
                table.append(line)
    X = pd.read_csv(io.StringIO("".join(table)), sep="\t", index_col=0).T
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    print("\nSample characteristic rows found in the file (check the label!):")
    keys = []
    for i, row in enumerate(chars):
        ne = [c for c in row if c.strip()]
        k = ne[0].split(":")[0].strip().lower() if ne else ""
        keys.append(k)
        print(f"  row {i}: '{k}'  non-empty in {len(ne)}/{len(row)} samples; e.g. {ne[:4]}")
    cand = [i for i, k in enumerate(keys) if label_hint in k]
    if not cand:
        sys.exit(f"\nNo characteristic row contains '{label_hint}'. Re-run with "
                 f"--label-hint <word from the list above> (e.g. a survival-time row).")
    row = chars[cand[0]]
    key = keys[cand[0]]
    unit_years = "year" in key
    cutoff = cutoff_months if cutoff_months else (3.0 if unit_years else 36.0)
    vals = pd.to_numeric(pd.Series([r.split(":", 1)[-1].strip() for r in row]),
                         errors="coerce").values
    keep = ~np.isnan(vals)
    # censoring: alive patients followed for less than the cut-off have unknown outcome
    st = [i for i, k in enumerate(keys) if k == "status"]
    if st and not KEEP_CENSORED:
        status = np.array([r.split(":", 1)[-1].strip().upper() for r in chars[st[0]]])
        died = np.array([s_.startswith("DOD") or s_.startswith("D") for s_ in status])
        censored = (vals <= cutoff) & ~died
        print(f"Dropping {int((censored & keep).sum())} patients censored before the cut-off (outcome unknown).")
        keep &= ~censored
    print(f"\nUsing row {cand[0]} ('{key}') as survival time; cut-off {cutoff} "
          f"({'years' if unit_years else 'months'}); {keep.sum()} samples kept.")
    y = (vals[keep] > cutoff).astype(int)
    X = X.values[keep]
    if X.max() > 50:                      # raw intensities -> log2
        X = np.log2(np.clip(X, 1, None))
    return X, y

def load_synthetic(n=185, d=3000, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.4).astype(int)
    X = r.normal(size=(n, d))
    X[:, :40] += y[:, None] * 0.6
    return X, y

# ------------------------------------------------------- quantum simulator --
def _apply_1q(psi, U, q):
    psi = psi.reshape((-1,) + (2,) * NQ)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q + 1])), 0, q + 1)
    return psi.reshape(-1, 2 ** NQ)

def _cnot(psi, c, t):
    psi = psi.reshape((-1,) + (2,) * NQ).copy()
    idx = [slice(None)] * (NQ + 1); idx[c + 1] = 1
    sub = psi[tuple(idx)]
    tt = t + 1 - (1 if t > c else 0)      # axis of target after control removed
    psi[tuple(idx)] = np.flip(sub, axis=tt)
    return psi.reshape(-1, 2 ** NQ)

def ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]])

def circuit(theta, psi):
    th = theta.reshape(NL, NQ)
    for l in range(NL):
        for q in range(NQ):
            psi = _apply_1q(psi, ry(th[l, q]), q)
        for q in range(NQ):
            psi = _cnot(psi, q, (q + 1) % NQ)
    return psi

def z0_expect(theta, A):
    psi = circuit(theta, A)
    p = np.abs(psi) ** 2
    signs = np.array([1 - 2 * ((i >> (NQ - 1)) & 1) for i in range(2 ** NQ)])
    return p @ signs                       # <Z_0> for every sample

def prob(theta, A):
    return np.clip((1 + z0_expect(theta, A)) / 2, 1e-7, 1 - 1e-7)

def loss(theta, A, y, w):
    p = prob(theta, A)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean())

def grad_ps(theta, A, y, w):
    """exact gradient via the parameter-shift rule (RY gates, shift pi/2)."""
    p = prob(theta, A)
    dl_dz = -(w * (y / p - (1 - y) / (1 - p))) * 0.5 / len(y)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta); e[i] = np.pi / 2
        dz = 0.5 * (z0_expect(theta + e, A) - z0_expect(theta - e, A))
        g[i] = float((dl_dz * dz).sum())
    return g

# ------------------------------------------------------------- optimisers --
class Tracker:
    """wraps the loss so every optimiser logs best-so-far training loss per evaluation"""
    def __init__(self, A, y, w):
        self.A, self.y, self.w, self.n, self.best, self.curve = A, y, w, 0, np.inf, []
    def __call__(self, th):
        v = loss(th, self.A, self.y, self.w); self.n += 1
        self.best = min(self.best, v); self.curve.append(self.best); return v

def opt_adam(T, th0, rng, steps=100, lr=0.1, sgd=False):
    th, m, v = th0.copy(), np.zeros(NPAR), np.zeros(NPAR)
    for t in range(1, steps + 1):
        T(th)
        g = grad_ps(th, T.A, T.y, T.w); T.n += 2 * NPAR
        if sgd:
            th -= lr * g
        else:
            m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
            th -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return th

def opt_sgd(T, th0, rng): return opt_adam(T, th0, rng, sgd=True)

def opt_cobyla(T, th0, rng, maxiter=300):
    r = minimize(T, th0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
    return r.x

def opt_spsa(T, th0, rng, iters=150, a=0.3, c=0.1):
    th = th0.copy()
    for k in range(1, iters + 1):
        ak, ck = a / (k + 10) ** 0.602, c / k ** 0.101
        d = rng.choice([-1., 1.], NPAR)
        g = (T(th + ck * d) - T(th - ck * d)) / (2 * ck) * d
        th = th - ak * g
    return th

def _kernel(P, Q, A0):
    """quantum fidelity kernel k(t,t') = |<psi(t)|psi(t')>|^2, psi(t)=U(t)|0000>"""
    def states(M):
        out = []
        for t in M:
            s = np.zeros((1, 2 ** NQ)); s[0, 0] = 1
            out.append(circuit(t, s)[0])
        return np.array(out)
    SP, SQ = states(P), states(Q)
    return np.abs(SP @ SQ.T) ** 2

def opt_qbo(T, th0, rng, n_init=20, n_total=150, kappa=2.0, n_cand=300, noise=1e-4):
    X = [th0] + [rng.uniform(-np.pi, np.pi, NPAR) for _ in range(n_init - 1)]
    Y = [T(x) for x in X]
    for _ in range(n_total - n_init):
        Xa, Ya = np.array(X), np.array(Y)
        mu0, sd0 = Ya.mean(), Ya.std() + 1e-9
        Yn = (Ya - mu0) / sd0
        K = _kernel(Xa, Xa, None) + noise * np.eye(len(Xa))
        C = rng.uniform(-np.pi, np.pi, (n_cand, NPAR))
        best = Xa[np.argsort(Ya)[:3]]                       # local candidates too
        C = np.vstack([C] + [b + rng.normal(0, .3, (n_cand // 6, NPAR)) for b in best])
        Ks = _kernel(C, Xa, None)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yn))
        mu = Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        var = np.clip(1 - (v * v).sum(0), 1e-12, None)
        lcb = mu - kappa * np.sqrt(var)                     # minimisation
        x_new = C[int(np.argmin(lcb))]
        X.append(x_new); Y.append(T(x_new))
    return np.array(X)[int(np.argmin(Y))]

OPTS = {"Adam": opt_adam, "SGD": opt_sgd, "COBYLA": opt_cobyla,
        "SPSA": opt_spsa, "QBO": opt_qbo}

# -------------------------------------------------------------- pipeline ---
def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b else np.nan
    prec, sens, spec = sd(tp, tp + fp), sd(tp, tp + fn), sd(tn, tn + fp)
    f1 = sd(2 * prec * sens, prec + sens) if prec == prec and sens == sens else np.nan
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = np.nan
    return dict(accuracy=(tp + tn) / len(y), precision=prec, sensitivity=sens,
                specificity=spec, f1=f1, auc=auc, TP=tp, FP=fp, TN=tn, FN=fn)

def prep(Xtr, ytr, Xte, k=500, encoding="angle"):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    sel = SelectKBest(f_classif, k=min(k, Xtr.shape[1])).fit(Xtr, ytr)
    Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    if encoding == "amplitude":             # 16 PCA comps -> 16 amplitudes
        pca = PCA(n_components=2 ** NQ, random_state=0).fit(Xtr)
        Atr, Ate = pca.transform(Xtr), pca.transform(Xte)
        nz = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return nz(Atr), nz(Ate)
    # angle encoding: 4 PCA comps -> RY(angle) on each qubit (product state)
    pca = PCA(n_components=NQ, random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    sd = Ztr.std(0) + 1e-12
    def states(Z):
        ang = np.pi / (1 + np.exp(-(Z / sd)))       # (0, pi)
        out = np.ones((len(Z), 1))
        for q in range(NQ):
            v = np.stack([np.cos(ang[:, q] / 2), np.sin(ang[:, q] / 2)], axis=1)
            out = (out[:, :, None] * v[:, None, :]).reshape(len(Z), -1)
        return out
    return states(Ztr), states(Zte)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv"); ap.add_argument("--label-col", default="y")
    ap.add_argument("--cache", default="data/GSE26712_series_matrix.txt.gz")
    ap.add_argument("--label-hint", default="surv", help="word in the survival-time row")
    ap.add_argument("--cutoff-months", type=float, default=None, help="default: 3 years / 36 months, chosen from the row units")
    ap.add_argument("--keep-censored", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--encoding", choices=["angle", "amplitude"], default="angle")
    ap.add_argument("--optimisers", default="Adam,SGD,COBYLA,SPSA,QBO")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    global KEEP_CENSORED; KEEP_CENSORED = a.keep_censored
    os.makedirs(a.out, exist_ok=True)

    if a.synthetic: X, y = load_synthetic(); print("SYNTHETIC data - pipeline test only")
    elif a.csv:
        df = pd.read_csv(a.csv); y = df.pop(a.label_col).values.astype(int); X = df.values
    else: X, y = load_geo(a.cache, a.cutoff_months, a.label_hint)
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 10:
        sys.exit(f"Label problem: class counts {np.bincount(y)}. Check cut-off/units.")
    print(f"\nData: {X.shape[0]} samples x {X.shape[1]} features, "
          f"class balance {np.bincount(y)} (0/1)")

    names = a.optimisers.split(",") + ["LogReg"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=42)
    rows, curves = [], {n: [] for n in names}
    t0 = time.time()
    for run, (tr, te) in enumerate(cv.split(X, y)):
        Atr, Ate = prep(X[tr], y[tr], X[te], k=500, encoding=a.encoding)
        w = np.where(y[tr] == 1, 0.5 * len(tr) / max((y[tr] == 1).sum(), 1),
                     0.5 * len(tr) / max((y[tr] == 0).sum(), 1))
        for name in names:
            rng = np.random.default_rng(1000 + run)
            if name == "LogReg":
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Atr, y[tr])
                p, nev = clf.predict_proba(Ate)[:, 1], 0
                trl = np.nan
            else:
                T = Tracker(Atr, y[tr], w)
                th0 = rng.uniform(-np.pi, np.pi, NPAR)
                th = OPTS[name](T, th0, rng)
                p, nev, trl = prob(th, Ate), T.n, loss(th, Atr, y[tr], w)
                curves[name].append(np.array(T.curve))
            rows.append(dict(run=run, optimiser=name, n_evals=nev, train_loss=trl,
                             **metrics(y[te], p)))
        print(f"run {run+1}/{5*a.repeats} done  ({time.time()-t0:.0f}s)", flush=True)

    R = pd.DataFrame(rows); R.to_csv(f"{a.out}/runs.csv", index=False)
    S = R.groupby("optimiser")[["accuracy", "precision", "sensitivity", "specificity",
                                "f1", "auc", "n_evals"]].agg(["mean", "std"]).round(4)
    S.to_csv(f"{a.out}/summary.csv"); print("\n=== SUMMARY (mean, std over runs) ===\n", S.to_string())

    # Wilcoxon signed-rank vs QBO on paired AUC, Bonferroni
    others = [n for n in names if n != "QBO" and "QBO" in names]
    st = []
    for n in others:
        x = R[R.optimiser == "QBO"].sort_values("run").auc.values
        z = R[R.optimiser == n].sort_values("run").auc.values
        try: p = wilcoxon(x, z).pvalue
        except ValueError: p = np.nan
        st.append(dict(comparison=f"QBO vs {n}", raw_p=p,
                       bonferroni_p=min(1, p * len(others)) if p == p else np.nan,
                       mean_auc_diff=float(np.mean(x - z))))
    if st:
        ST = pd.DataFrame(st); ST.to_csv(f"{a.out}/stats.csv", index=False)
        print("\n=== Wilcoxon vs QBO (AUC) ===\n", ST.round(4).to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([R[R.optimiser == n].auc.dropna() for n in names]); ax.set_xticklabels(names)
    ax.set_ylabel("Test AUC-ROC"); ax.set_title("AUC across paired runs")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_auc.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, cs in curves.items():
        if not cs: continue
        L = min(len(c) for c in cs); M = np.array([c[:L] for c in cs])
        ax.plot(np.arange(1, L + 1), M.mean(0), label=n)
    ax.set_xscale("log"); ax.set_xlabel("Loss evaluations (training set)")
    ax.set_ylabel("Best training loss so far"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_convergence.png", dpi=200)
    print(f"\nSaved runs.csv, summary.csv, stats.csv and figures in ./{a.out}/")
    print("Report these numbers exactly as produced. Simulation is noiseless.")

if __name__ == "__main__":
    main()
