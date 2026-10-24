"""SHAP analysis for the best model of a family/target combination and
comparison against solver-based sensitivity indices.

The best model is taken from results/summary_metrics.csv (highest CV
R2).  Tree ensembles are refitted on the unscaled features (identical
predictions, monotone transform) so that SHAP values are expressed in
the original physical units; for other model classes the permutation
explainer is used.

Sensitivity indices from the solver are computed by central finite
differences at 30 sampled designs and standardised by the feature and
target standard deviations, which makes them directly comparable to
mean absolute SHAP values.

Outputs:
  results/shap/{family}_{target}_shap_values.csv
  results/shap/{family}_{target}_mean_abs_shap.csv
  results/shap/{family}_{target}_dependence_{top3}.csv
  results/shap/{family}_{target}_interaction.csv   (XGBoost only)
  results/shap/{family}_{target}_sensitivity_comparison.csv
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_pipeline  # noqa: E402
from sampling import BOXES, TARGET_NAMES  # noqa: E402
from train_eval import load_family  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42
N_SENS_DESIGNS = 30


def refit_raw(name, params, X, y):
    """Refit a tree model directly on unscaled features for SHAP."""
    clean = {k: v for k, v in params.items() if k != 'inner_cv_r2'
             and k != 'note'}
    if name == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=SEED, n_jobs=1,
                                     **clean).fit(X, y)
    if name == 'XGB':
        from xgboost import XGBRegressor
        return XGBRegressor(objective='reg:squarederror', random_state=SEED,
                            n_jobs=1, verbosity=0, **clean).fit(X, y)
    if name == 'LGBM':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1,
                             subsample_freq=1, **clean).fit(X, y)
    if name == 'CatBoost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=SEED, verbose=0,
                                 allow_writing_files=False,
                                 thread_count=1, **clean).fit(X, y)
    raise ValueError('not a tree model: %s' % name)


def solver_sensitivities(family, target, df, feats):
    """Standardised sensitivity indices from central differences of the
    solver at 30 sampled designs."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_SENS_DESIGNS, len(df)),
                     replace=False)
    lo = np.array([b[2] for b in BOXES[family]])
    hi = np.array([b[3] for b in BOXES[family]])
    step = 0.01 * (hi - lo)

    from sampling import solve_row
    t_idx = TARGET_NAMES[family].index(target)
    accum = {f: [] for f in feats}
    for i in idx:
        x0 = df[feats].values[i].copy()
        for j, fname in enumerate(feats):
            xp, xm = x0.copy(), x0.copy()
            xp[j] += step[j]
            xm[j] -= step[j]
            yp = solve_row(family, xp)[t_idx]
            ym = solve_row(family, xm)[t_idx]
            accum[fname].append((yp - ym) / (2.0 * step[j]))
    sx = df[feats].std().values
    sy = df[target].std()
    return pd.DataFrame({
        'feature': feats,
        'sensitivity_index': [np.mean(np.abs(accum[f])) * sx[j] / sy
                              for j, f in enumerate(feats)],
    })


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True,
                    choices=['beam', 'frame', 'truss'])
    ap.add_argument('--target', required=True)
    args = ap.parse_args()
    family, target = args.family, args.target
    tag = '%s_%s' % (family, target)

    summ = pd.read_csv(os.path.join(ROOT, 'results', 'summary_metrics.csv'))
    sel = summ[(summ.family == family) & (summ.target == target)]
    best = sel.sort_values('cv_r2_mean').iloc[-1]
    name, cv_r2 = best['model'], best['cv_r2_mean']
    print('[%s] best model: %s (CV R2 = %.4f)' % (tag, name, cv_r2))

    df, feats = load_family(family)
    train = df['split'].values == 'train'
    test = ~train
    X, y = df[feats].values, df[target].values

    with open(os.path.join(ROOT, 'results', 'tuning',
                           tag + '_best_params.json')) as fh:
        params_all = json.load(fh)
    params = params_all[name]

    import shap

    if name in ('RF', 'XGB', 'LGBM', 'CatBoost'):
        model = refit_raw(name, params, X[train], y[train])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[test])
        method = 'TreeExplainer on %s (refitted on unscaled features)' % name
    else:
        pipe = make_pipeline(name, seed=SEED).fit(X[train], y[train])
        bg = shap.maskers.Independent(X[train], max_samples=60)
        explainer = shap.explainers.Permutation(pipe.predict, bg)
        sv = explainer(X[test]).values
        method = 'Permutation explainer on pipeline %s' % name

    sv = np.asarray(sv)
    shp = pd.DataFrame(sv, columns=['shap_' + f for f in feats])
    shp.insert(0, 'design_id', df['design_id'].values[test])
    shp.to_csv(os.path.join(ROOT, 'results', 'shap',
                            tag + '_shap_values.csv'), index=False)

    mean_abs = pd.DataFrame({'feature': feats,
                             'mean_abs_shap': np.abs(sv).mean(axis=0)})
    mean_abs = mean_abs.sort_values('mean_abs_shap', ascending=False)
    mean_abs.to_csv(os.path.join(ROOT, 'results', 'shap',
                                 tag + '_mean_abs_shap.csv'), index=False)

    # dependence data for the three most influential features
    top3 = mean_abs['feature'].head(3).tolist()
    for f in top3:
        j = feats.index(f)
        dep = pd.DataFrame({'feature_value': X[test][:, j],
                            'shap_value': sv[:, j]})
        dep.to_csv(os.path.join(ROOT, 'results', 'shap',
                                '%s_dependence_%s.csv' % (tag, f)),
                   index=False)

    # interaction matrix (XGBoost only)
    if name == 'XGB':
        inter = explainer.shap_interaction_values(X[test])
        mi = np.abs(inter).mean(axis=0)
        pd.DataFrame(mi, index=feats, columns=feats).to_csv(
            os.path.join(ROOT, 'results', 'shap',
                         tag + '_interaction.csv'))

    # comparison against solver sensitivities
    sens = solver_sensitivities(family, target, df, feats)
    cmp = mean_abs.merge(sens, on='feature')
    cmp['shap_rank'] = cmp['mean_abs_shap'].rank(ascending=False)
    cmp['sensitivity_rank'] = cmp['sensitivity_index'].rank(ascending=False)
    rho, pv = spearmanr(cmp['mean_abs_shap'], cmp['sensitivity_index'])
    cmp_out = cmp.sort_values('shap_rank')
    cmp_out.to_csv(os.path.join(ROOT, 'results', 'shap',
                                tag + '_sensitivity_comparison.csv'),
                   index=False)
    print('[%s] SHAP vs solver sensitivity: Spearman rho = %.3f '
          '(p = %.2e); explainer: %s' % (tag, rho, pv, method))


if __name__ == '__main__':
    main()
