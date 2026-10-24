"""Learning curves for the best model of a family/target combination:
held-out R2 as a function of the number of simulated designs used for
training.  This is the simulation-budget trade-off of the paper.

Output: results/learning_curves/{family}_{target}_learning_curve.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_pipeline  # noqa: E402
from sampling import TARGET_NAMES  # noqa: E402
from train_eval import load_family  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42
SIZES = [40, 80, 160, 240, 320]   # with cv=5 the fold limit is 320


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
    name = best['model']
    with open(os.path.join(ROOT, 'results', 'tuning',
                           tag + '_best_params.json')) as fh:
        params = json.load(fh)[name]
    clean = {k: v for k, v in params.items() if k not in ('inner_cv_r2',
                                                          'note')}

    df, feats = load_family(family)
    train = df['split'].values == 'train'
    X, y = df[feats].values[train], df[target].values[train]

    pipe = make_pipeline(name, seed=SEED, **clean)
    sizes, tr_scores, va_scores = learning_curve(
        pipe, X, y, train_sizes=SIZES, cv=5, scoring='r2',
        shuffle=True, random_state=SEED, n_jobs=-1)

    rows = []
    for i, s in enumerate(sizes):
        for k in range(tr_scores.shape[1]):
            rows.append(dict(train_size=int(s), fold=k,
                             r2_train=float(tr_scores[i, k]),
                             r2_val=float(va_scores[i, k])))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, 'results', 'learning_curves',
                            tag + '_learning_curve.csv'), index=False)
    agg = out.groupby('train_size').mean(numeric_only=True)
    r0 = agg.loc[sizes[0], 'r2_val']
    r1 = agg.loc[sizes[-1], 'r2_val']
    print('[%s] learning curve (%s): R2(val) %.4f @ n=40 -> %.4f @ n=320'
          % (tag, name, r0, r1))


if __name__ == '__main__':
    main()
