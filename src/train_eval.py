"""Tuning, cross-validation and evaluation of the nine-model tournament
for one family/target combination.

Stages per (family, target):
  1. hyperparameter selection by randomised search (10 candidates,
     inner 5-fold CV) on the training partition
  2. outer 10-fold cross-validation with the frozen hyperparameters,
     recording every fold prediction
  3. final fit on the full training partition -> held-out test and
     extrapolation predictions
  4. paired Wilcoxon tests comparing absolute errors of the best model
     against every other model on the held-out set

Outputs (per family/target):
  results/tuning/{family}_{target}_best_params.json
  results/cv/{family}_{target}_cv_predictions.csv
  results/cv/{family}_{target}_cv_metrics.csv
  results/heldout/{family}_{target}_predictions.csv
  results/extrapolation/{family}_{target}_predictions.csv
  results/heldout/{family}_{target}_wilcoxon_vs_best.csv
  results/summary_metrics.csv  (appended, one row per model)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import KFold, RandomizedSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODEL_NAMES, SEARCH_SPACES, make_pipeline  # noqa: E402
from sampling import BOXES, TARGET_NAMES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42


def metrics(y, p):
    err = p - y
    return {
        'r2': r2_score(y, p),
        'rmse': float(np.sqrt(mean_squared_error(y, p))),
        'mae': mean_absolute_error(y, p),
        'mape': float(np.mean(np.abs(err / y)) * 100.0),
        'medape': float(np.median(np.abs(err / y)) * 100.0),
    }


def load_family(family):
    df = pd.read_csv(os.path.join(ROOT, 'data', '%s_dataset.csv' % family))
    splits = pd.read_csv(os.path.join(ROOT, 'data', 'splits',
                                      '%s_folds.csv' % family))
    df = df.merge(splits, on='design_id')
    feats = [b[0] for b in BOXES[family]]
    return df, feats


def run_combo(family, target, summary_path):
    t_combo = time.time()
    df, feats = load_family(family)
    X, y = df[feats].values, df[target].values
    train = df['split'].values == 'train'
    test = ~train
    fold = df['fold'].values
    print('[%s/%s] start: %d train / %d test records'
          % (family, target, train.sum(), test.sum()))

    cv_pred_rows, cv_metric_rows, heldout_pred_rows, ext_pred_rows = \
        [], [], [], []
    best_params_all = {}
    timings = {}

    dfe = pd.read_csv(os.path.join(ROOT, 'data', 'extrapolation',
                                   '%s_ext.csv' % family))
    Xe, ye = dfe[feats].values, dfe[target].values

    for name in MODEL_NAMES:
        t0 = time.time()
        pipe = make_pipeline(name, seed=SEED)
        space = SEARCH_SPACES[name]
        if space:
            inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
            search = RandomizedSearchCV(
                pipe, space, n_iter=10, cv=inner, scoring='r2',
                random_state=SEED, n_jobs=-1, refit=True)
            search.fit(X[train], y[train])
            best = search.best_estimator_

            def _jsonable(v):
                if isinstance(v, tuple):
                    return list(v)
                if isinstance(v, (int, float, str, type(None), list)):
                    return v
                return str(v)

            best_params_all[name] = {
                k.replace('model__', ''): _jsonable(v)
                for k, v in search.best_params_.items()}
            best_params_all[name]['inner_cv_r2'] = float(search.best_score_)
        else:
            best = pipe.fit(X[train], y[train])
            best_params_all[name] = {'note': 'no hyperparameters to tune'}
        t_tune = time.time() - t0

        # outer 10-fold CV with frozen hyperparameters
        fold_metrics = []
        for k in range(10):
            va = fold == k
            tr = train & ~va
            m = clone(best).fit(X[tr], y[tr])
            p = m.predict(X[va])
            for i, (yt, yp) in enumerate(zip(y[va], p)):
                cv_pred_rows.append(dict(
                    model=name, fold=k, design_id=int(df['design_id']
                                                      .values[va][i]),
                    y_true=float(yt), y_pred=float(yp)))
            fold_metrics.append(metrics(y[va], p))

        fm = pd.DataFrame(fold_metrics)
        row = {'model': name,
               'cv_r2_mean': fm['r2'].mean(), 'cv_r2_std': fm['r2'].std(),
               'cv_rmse_mean': fm['rmse'].mean(),
               'cv_mae_mean': fm['mae'].mean(),
               'cv_mape_mean': fm['mape'].mean(),
               'cv_medape_mean': fm['medape'].mean()}
        cv_metric_rows.append(row)

        # final fit and evaluation
        final = clone(best).fit(X[train], y[train])
        p_test = final.predict(X[test])
        for i, (yt, yp) in enumerate(zip(y[test], p_test)):
            heldout_pred_rows.append(dict(
                model=name, design_id=int(df['design_id'].values[test][i]),
                y_true=float(yt), y_pred=float(yp)))
        hm = metrics(y[test], p_test)
        row.update({'heldout_r2': hm['r2'], 'heldout_rmse': hm['rmse'],
                    'heldout_mae': hm['mae'], 'heldout_mape': hm['mape'],
                    'heldout_medape': hm['medape']})

        p_ext = final.predict(Xe)
        for i, (yt, yp) in enumerate(zip(ye, p_ext)):
            ext_pred_rows.append(dict(
                model=name, design_id=int(dfe['design_id'].values[i]),
                y_true=float(yt), y_pred=float(yp)))
        em = metrics(ye, p_ext)
        row.update({'ext_r2': em['r2'], 'ext_rmse': em['rmse'],
                    'ext_mae': em['mae'], 'ext_mape': em['mape'],
                    'ext_medape': em['medape'],
                    'tuning_time_s': t_tune})
        timings[name] = time.time() - t0
        print('[%s/%s] %-8s cv R2 = %.4f +- %.4f | held-out R2 = %.4f | '
              'ext R2 = %.4f (%.1f s)'
              % (family, target, name, row['cv_r2_mean'], row['cv_r2_std'],
                 row['heldout_r2'], row['ext_r2'], timings[name]))

    # ---- persist everything -------------------------------------------
    tag = '%s_%s' % (family, target)
    pd.DataFrame(cv_pred_rows).to_csv(
        os.path.join(ROOT, 'results', 'cv', tag + '_cv_predictions.csv'),
        index=False)
    pd.DataFrame(cv_metric_rows).to_csv(
        os.path.join(ROOT, 'results', 'cv', tag + '_cv_metrics.csv'),
        index=False)
    ho = pd.DataFrame(heldout_pred_rows)
    ho.to_csv(os.path.join(ROOT, 'results', 'heldout',
                           tag + '_predictions.csv'), index=False)
    pd.DataFrame(ext_pred_rows).to_csv(
        os.path.join(ROOT, 'results', 'extrapolation',
                     tag + '_predictions.csv'), index=False)
    with open(os.path.join(ROOT, 'results', 'tuning',
                           tag + '_best_params.json'), 'w') as fh:
        json.dump(best_params_all, fh, indent=2)

    # Wilcoxon signed-rank on held-out absolute errors vs the best model
    best_name = (pd.DataFrame(cv_metric_rows).sort_values('cv_r2_mean')
                 .iloc[-1]['model'])
    piv = ho.pivot(index='design_id', columns='model', values='y_pred')
    ytrue = ho.groupby('design_id')['y_true'].first()
    abserr = piv.apply(lambda c: (c - ytrue).abs())
    wrows = []
    for name in MODEL_NAMES:
        if name == best_name:
            continue
        stat, p = wilcoxon(abserr[best_name].values, abserr[name].values)
        wrows.append(dict(model=name, vs=best_name,
                          statistic=float(stat), p_value=float(p)))
    pd.DataFrame(wrows).to_csv(
        os.path.join(ROOT, 'results', 'heldout',
                     tag + '_wilcoxon_vs_best.csv'), index=False)

    summ = pd.DataFrame(cv_metric_rows)
    summ.insert(0, 'family', family)
    summ.insert(1, 'target', target)
    summ['best_by_cv'] = summ['model'] == best_name
    hdr = not os.path.exists(summary_path)
    summ.to_csv(summary_path, mode='a', header=hdr, index=False)
    print('[%s/%s] done in %.1f s (best by CV: %s)'
          % (family, target, time.time() - t_combo, best_name))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True,
                    choices=['beam', 'frame', 'truss'])
    ap.add_argument('--target', required=True)
    args = ap.parse_args()
    if args.target not in TARGET_NAMES[args.family]:
        sys.exit('unknown target for family')
    summary = os.path.join(ROOT, 'results', 'summary_metrics.csv')
    run_combo(args.family, args.target, summary)


if __name__ == '__main__':
    main()
