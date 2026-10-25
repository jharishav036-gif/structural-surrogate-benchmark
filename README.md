# Explainable machine learning surrogates for structural simulation

Data and code release for the study *"Explainable machine learning
surrogates for structural simulation: a physics-validated benchmark
study"*.

**Authors.** Rishav Jha (1), Suresh Kumar Sahani (1, corresponding),
Kameshwar Sahani (2)

1. MIT Campus, Rajarshi Janak University, Janakpurdham, Nepal
2. Department of Civil Engineering, Kathmandu University, Dhulikhel,
   Nepal

**Correspondence:** sureshsahani54@gmail.com

## What this package contains

The study benchmarks nine regression models as surrogates for three
families of structural simulations - a cantilever beam, a fixed-base
portal frame and a six-panel Pratt truss - across three response
targets each (maximum deflection, maximum normal stress, first natural
frequency). Every training label is produced by our own finite-element
/ direct-stiffness solvers, which are verified against closed-form
solutions. The surrogate predictions are then analysed with SHAP and
compared against independent solver-based sensitivity indices, and the
computational speed-up of surrogate inference over full simulation is
quantified.

```
structural-surrogate-benchmark/
├── README.md                  this file
├── LICENSE                    MIT
├── requirements.txt           package versions used for the study
├── data/
│   ├── beam_dataset.csv       500 designs, 8 features, 3 targets
│   ├── frame_dataset.csv      500 designs, 10 features, 3 targets
│   ├── truss_dataset.csv      500 designs, 9 features, 3 targets
│   ├── extrapolation/         100 designs per family, extended box
│   ├── splits/                train/test split and 10 CV folds
│   ├── data_dictionary.md     column-by-column documentation
│   └── acceptance_rates.csv   physical screening statistics
├── src/
│   ├── solvers.py             beam / frame / truss solvers (NumPy)
│   ├── verification.py        closed-form verification suite
│   ├── sampling.py            parameter boxes, LHS + physical screening
│   ├── generate_data.py       dataset and split construction
│   ├── models.py              the nine-model tournament + search spaces
│   ├── train_eval.py          tuning, 10-fold CV, held-out, extrapolation
│   ├── shap_analysis.py       SHAP + solver sensitivity comparison
│   ├── learning_curves.py     performance vs simulation budget
│   ├── speed_benchmark.py     solver vs surrogate timing
│   ├── statistical_tests.py   Friedman omnibus + Holm-corrected pairwise
│   ├── screening_demo.py      100k-candidate design screening study
│   └── run_pipeline.py        stage orchestration
└── results/
    ├── RESULTS_SUMMARY.md     headline numbers (start here)
    ├── verification/          14 closed-form checks + mesh convergence
    ├── cv/                    per-fold predictions and metrics
    ├── heldout/               predictions + Wilcoxon comparisons
    ├── extrapolation/         extended-box predictions
    ├── tuning/                selected hyperparameters (JSON)
    ├── shap/                  SHAP values, rankings, sensitivity tables
    ├── learning_curves/       budget curves per combination
    ├── speed/                 timing benchmark
    ├── stats/                 Friedman + Holm-corrected comparisons
    ├── screening/             screening study: precision, recall,
    │                          conformal coverage, top designs
    └── logs/                  full run log
```

## Requirements

Python 3.12 with the versions listed in `requirements.txt`
(numpy 2.1.3, scipy 1.14.1, pandas 2.2.3, scikit-learn 1.5.2,
xgboost 2.1.3, lightgbm 4.5.0, catboost 1.2.8, shap 0.54.1).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the study

All randomness is seeded (seed 42 throughout) and the pipeline is
deterministic on a fixed platform.

```bash
# stage by stage
python src/run_pipeline.py --stage verify    # solver checks
python src/run_pipeline.py --stage data      # datasets + splits
python src/run_pipeline.py --stage train     # tournament (about 20 min)
python src/run_pipeline.py --stage analyze   # SHAP + learning curves
python src/run_pipeline.py --stage speed     # timing benchmark
python src/run_pipeline.py --stage stats     # Friedman + Holm tests
python src/run_pipeline.py --stage screen    # screening study (few min)

# or everything at once
python src/run_pipeline.py --stage all
```

Single family/target combinations can be run directly, for example:

```bash
python src/train_eval.py --family beam --target tip_deflection_mm
python src/shap_analysis.py --family frame --target sway_mm
```

Expect small run-to-run differences in timings only; the fitted models
and their metrics reproduce exactly.

## Headline findings

- **Accuracy.** The best model per combination reaches 10-fold CV
  R2 = 0.93-0.99 (held-out 0.90-0.99). A small multilayer perceptron
  is best in eight of nine combinations; support vector regression wins
  the truss frequency task. Friedman tests reject the hypothesis that
  the nine models are interchangeable in every combination (p < 1e-20)
  and 61 of the 72 Holm-corrected pairwise comparisons are significant.
  Full tables: `results/RESULTS_SUMMARY.md`.
- **Physics validation.** The solvers pass all 14 closed-form checks
  (catalogue beam formulas, frame stiffness-ratio limits, method of
  joints and Castigliano for trusses) with relative errors at machine
  precision, and equilibrium residuals below 1e-14 on every dataset
  record.
- **Interpretability.** SHAP rankings of the best models match
  independent solver-based sensitivity indices with Spearman
  correlation 0.90-1.00; the dominant features are exactly the ones
  the analytical formulas point to (L and h for the beam, H and hc for
  the frame, S and load intensity for the truss).
- **Utility.** Batched surrogate inference runs about 1000x (beam,
  truss) to 3800x (frame) faster than a full static-plus-eigenvalue
  solve, at microsecond cost per design. In the 100,000-candidate
  design-screening study the surrogate shortlist achieves precision
  0.96-0.97 and recall 0.95-0.99 against solver verification, with 90%
  split-conformal intervals covering 0.87-0.93 of admissible
  candidates.

## Data availability

The complete data and code for this study are released under the MIT
license. Design datasets are plain CSV files documented in
`data/data_dictionary.md`; every table in `results/` is reproduced by
the pipeline above. File checksums are listed in `checksums.sha256`.
