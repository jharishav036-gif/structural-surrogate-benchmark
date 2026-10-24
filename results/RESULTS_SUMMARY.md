# Results summary

Complete results are in the `results/` directory (CSV per analysis).
This file is a compact overview of the headline numbers.

## Solver verification (results/verification/)

All 14 closed-form checks pass. Cantilever deflections, fixed-end
moments and the first natural frequency agree with catalogue formulas
to relative errors below 3e-11, 1e-8 respectively; the portal-frame
sway agrees with the rigid-beam and flexible-beam limits to 4.8e-3 and
2.9e-4; the two-bar truss matches method-of-joints forces and the
Castigliano deflection to machine precision; the Pratt truss reproduces
statically determinate force invariance and equilibrium residuals below
4e-15. Mesh convergence tables for the eigenvalue analyses are provided
for both discretised families.

## Datasets

| family | n train / test | screening acceptance | deflection range | stress range | f1 range |
|---|---|---|---|---|---|
| beam | 400 / 100 | 98.2% | 0.09-52.7 mm | 2.3-241.8 MPa | 6.6-178.5 Hz |
| frame | 400 / 100 | 100.0% | 0.05-23.2 mm | 2.1-105.9 MPa | 2.6-57.7 Hz |
| truss | 400 / 100 | 85.3% | 2.9-74.7 mm | 28.2-348.8 MPa | 19.3-105.9 Hz |

## Tournament: 10-fold cross-validated R2 (results/summary_metrics.csv)

| family / target | Linear | KNN | SVR | Tree | RF | XGB | LGBM | CatBoost | MLP |
|---|---|---|---|---|---|---|---|---|---|
| beam deflection | 0.616 | 0.627 | 0.842 | 0.598 | 0.765 | 0.846 | 0.829 | 0.904 | **0.978** |
| beam stress | 0.747 | 0.706 | 0.879 | 0.566 | 0.782 | 0.919 | 0.883 | 0.942 | **0.979** |
| beam f1 | 0.787 | 0.756 | 0.916 | 0.799 | 0.897 | 0.970 | 0.964 | 0.986 | **0.989** |
| frame sway | 0.656 | 0.651 | 0.917 | 0.657 | 0.802 | 0.871 | 0.852 | 0.918 | **0.960** |
| frame stress | 0.749 | 0.695 | 0.916 | 0.573 | 0.775 | 0.890 | 0.891 | 0.905 | **0.941** |
| frame f1 | 0.861 | 0.800 | 0.971 | 0.715 | 0.870 | 0.941 | 0.933 | 0.967 | **0.980** |
| truss deflection | 0.808 | 0.667 | 0.926 | 0.566 | 0.744 | 0.885 | 0.875 | 0.910 | **0.976** |
| truss stress | 0.805 | 0.669 | 0.875 | 0.595 | 0.796 | 0.910 | 0.908 | 0.930 | **0.932** |
| truss f1 | 0.891 | 0.779 | **0.974** | 0.652 | 0.851 | 0.955 | 0.947 | 0.972 | 0.955 |

The multilayer perceptron is the best model for eight of the nine
family/target combinations (held-out R2 0.90-0.99); support vector
regression wins the truss first-frequency task (0.987 held-out). Paired
Wilcoxon tests on held-out absolute errors show that 5 to 8 of the 8
competing models are significantly worse than the best model
(p < 0.05) in every combination.

## Statistical comparison (results/stats/)

Friedman omnibus tests on the held-out absolute errors (100 test
designs ranked per combination) reject the null hypothesis that the
nine models are interchangeable in all nine combinations
(chi2 = 113-280, p = 7.7e-21 to 6.8e-56). After Holm-Bonferroni
correction of the 72 best-vs-competitor comparisons, 61 remain
significant at alpha = 0.05. The mean Friedman rank across all nine
combinations gives the overall tournament ranking:

| rank | model | mean Friedman rank | combinations won |
|---|---|---|---|
| 1 | MLP | 3.52 | 5 |
| 2 | CatBoost | 3.59 | 2 |
| 3 | SVR | 3.90 | 2 |
| 4 | XGBoost | 4.37 | 0 |
| 5 | LightGBM | 4.44 | 0 |
| 6 | Random forest | 5.85 | 0 |
| 7 | Decision tree | 6.30 | 0 |
| 8 | k-nearest neighbours | 6.46 | 0 |
| 9 | Linear regression | 6.56 | 0 |

## Extrapolation (results/extrapolation/)

On designs from the 15%-extended parameter box, the best models retain
R2 between 0.82 and 0.92 (interpolation R2 0.90-0.99), with graceful
degradation rather than failure. Full prediction tables for all nine
models are provided per combination.

## Interpretability (results/shap/)

SHAP feature rankings of the best model agree with independent
solver-based sensitivity indices: Spearman rank correlation between
mean absolute SHAP values and standardised finite-difference
sensitivities is 0.90-1.00 across the nine combinations (median 0.98).
The dominant features match the analytical expectations: span L and
depth h for the beam, column height H and depth hc for the frame, span
S and the load intensity for the truss. Per-sample SHAP values,
dependence data for the three most influential features, and the full
ranking comparison tables are provided per combination.

## Learning curves (results/learning_curves/)

Held-out R2 of the best model as a function of training-set size
(40 to 320 simulated designs, 5-fold): all combinations exceed 0.90
beyond about 240 designs. The beam deflection task is the most
data-hungry (-0.02 at n = 40, 0.96 at n = 320), which reflects the
four-order-of-magnitude span of that target.

## Speed benchmark (results/speed/speed_benchmark.csv)

| family | solver per design | surrogate (single) | surrogate (batch of 1000) | speed-up (batch) |
|---|---|---|---|---|
| beam | 0.70 ms | 0.079 ms | 0.7 us | ~982x |
| frame | 2.84 ms | 0.078 ms | 0.7 us | ~3808x |
| truss | 0.72 ms | 0.077 ms | 0.7 us | ~975x |

Batched surrogate inference is three to four orders of magnitude
faster than a full static-plus-eigenvalue solve; the single-call
speed-up (9-37x) is limited by the Python call overhead of the
prediction pipeline rather than by the model itself.

## Design-screening study (results/screening/)

Scenario: find every design in a 100,000-candidate Latin hypercube
sample of each family's parameter box that satisfies a
vibration-sensitive design brief (deflection, stress and first-
frequency limits; see the table below). The three tournament-winning
surrogates screen the candidates; the finite-element solver verifies
the shortlist and a 2,000-design random sample estimates recall.

| family | brief | selected | precision | recall | surrogate screen | brute-force est. |
|---|---|---|---|---|---|---|
| beam | defl <= L/500, 60 MPa, 30 Hz | 39,431 (39.4%) | 0.972 | 0.988 | 0.16 s | 94 s |
| frame | sway <= H/600, 35 MPa, 12 Hz | 37,524 (37.5%) | 0.956 | 0.945 | 0.26 s | 272 s |
| truss | defl <= S/300, 180 MPa, 45 Hz | 30,819 (30.8%) | 0.968 | 0.950 | 0.98 s | 63 s |

Every shortlisted design was verified with the solver (no cap was
reached). The surrogate preparation cost (tuning plus final fits, about
21-33 s per family) breaks even against brute-force simulation after
roughly 9,500 (frame) to 35,500 (beam) evaluations, so a single
100,000-candidate screen repays it several times over.

90% split-conformal prediction intervals in multiplicative form
(calibrated on the held-out partition) cover 0.87-0.93 of admissible
candidates across the nine family/target pairs. Coverage on the raw
candidate box is slightly lower for the truss (0.81-0.86) because
about 15% of raw candidates fall outside the physically admissible
region the models were trained on; the per-target radii, coverages and
pool MAPEs are tabulated in screening_conformal.csv, and the ten most
comfortable verified designs per family (ranked by worst utilisation
ratio) in {family}_top_designs.csv.
