# Optimiser benchmark for a variational quantum classifier on ovarian cancer gene-expression data

Code and results for the manuscript *"Comparative Analysis of Classical, Quantum, and Hybrid Optimisation
Techniques for Ovarian Cancer Survival Classification with a Variational Quantum Classifier: A Reproducible
Benchmark"* (submitted to Computer Methods and Programs in Biomedicine, CMPB-D-26-08851).

## What this does
Trains one fixed 4-qubit, 12-parameter variational quantum classifier (noiseless NumPy state-vector
simulation, CPU only) with five optimisers and compares them with a logistic-regression reference:

| Optimiser | Type | Loss evaluations per fold |
|---|---|---|
| Adam | classical update, parameter-shift gradients | 2,500 |
| SGD | classical update, parameter-shift gradients | 2,500 |
| COBYLA | gradient-free | 300 |
| SPSA | stochastic gradient estimate | 300 |
| QBO | quantum-kernel Gaussian-process Bayesian optimisation | 150 |

* Data: public GEO series **GSE26712** (downloaded automatically). Outcome: survival beyond 3 years
  (class 1). Patients alive with follow-up of 3 years or less (censored, n = 16) are excluded, leaving 169.
* Evaluation: repeated stratified 5-fold cross-validation, 5 repeats (25 paired test folds, seed 42).
  Scaling, ANOVA feature selection (500 probe sets) and PCA (4 components) are fitted on training folds only.
* Metrics: accuracy, precision, sensitivity, specificity, F1, AUC, confusion counts; paired Wilcoxon
  signed-rank tests of QBO against each other model with Bonferroni correction.

## Run
```
pip install -r requirements.txt
python verify_optimisers.py --synthetic --repeats 1   # quick pipeline test (about 2-4 minutes)
python verify_optimisers.py                           # full experiment (about 16 minutes on a laptop CPU)
```
Outputs are written to `results/`: `runs.csv` (one row per fold and model, with TP/FP/TN/FN),
`summary.csv`, `stats.csv`, `fig_auc.png`, `fig_convergence.png`.
Tested with Python 3.13. Other useful options: `--repeats N`, `--optimisers Adam,QBO`, `--encoding amplitude`,
`--keep-censored`, `--csv your.csv --label-col y`.

## Results included in this repository
The files runs.csv, summary.csv, stats.csv, fig_auc.png and fig_convergence.png in the top level of this repository are the output of the run reported in the manuscript. Mean test AUC: Adam 0.635, SGD 0.630, COBYLA 0.626, QBO 0.610, SPSA 0.579, logistic regression 0.651. No circuit optimiser differed significantly from QBO after Bonferroni correction.

## Limitations
Noiseless simulation only; no quantum hardware or noise model. Small circuit and four input features.
Optimiser settings were fixed in advance and not tuned; evaluation budgets differ between optimisers.
One cohort, no external validation. These results do not demonstrate quantum advantage.

## Licence
MIT (see `LICENSE`).
