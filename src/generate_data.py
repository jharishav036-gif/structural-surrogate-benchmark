"""Generate the three benchmark datasets and the train/test/fold splits.

Outputs:
  data/{family}_dataset.csv            500 screened designs per family
  data/extrapolation/{family}_ext.csv  100 designs from the extended box
  data/{family}_dataset_stats.csv      descriptive statistics
  data/splits/{family}_folds.csv       design_id -> split + CV fold
  data/acceptance_rates.csv

The split is common to all three targets of a family so that results
are comparable across targets.  Stratification uses deciles of the
deflection target.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import BOXES, sample_screened, solve_row  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42
N_TRAIN_TEST = 500
N_EXT = 100


def build_family(family, accept_rows):
    t0 = time.time()
    print('[%s] sampling %d screened designs ...' % (family, N_TRAIN_TEST))
    X, y, n_cand = sample_screened(family, N_TRAIN_TEST, seed=SEED)
    names = [b[0] for b in BOXES[family]]
    from sampling import TARGET_NAMES
    cols = names + TARGET_NAMES[family]
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df.insert(0, 'design_id', np.arange(1, len(df) + 1))
    out = os.path.join(ROOT, 'data', '%s_dataset.csv' % family)
    df.to_csv(out, index=False)

    row = dict(family=family, n_candidates=n_cand, n_accepted=len(df),
               acceptance_rate=len(df) / n_cand)

    # sanity check on the accepted records: for the truss we can re-solve
    # each design and confirm the equilibrium residual stays at solver
    # precision (the beam and frame are covered by the closed-form
    # verification suite)
    if family == 'truss':
        from solvers import solve_truss
        max_res = 0.0
        for _, r in df.iterrows():
            res = solve_truss(r['S'], r['Ht'], r['A_chord'] * 1e-6,
                              r['A_diag'] * 1e-6, r['A_vert'] * 1e-6,
                              r['E'] * 1e9, r['rho'], r['P'] * 1e3,
                              r['w'] * 1e3)
            max_res = max(max_res, res['residual'])
        row['max_equilibrium_residual'] = max_res
        print('[%s] max equilibrium residual over accepted records: %.2e'
              % (family, max_res))
    accept_rows.append(row)
    print('[%s] accepted %d/%d candidates (%.1f%%), %.1f s'
          % (family, len(df), n_cand, 100.0 * len(df) / n_cand,
             time.time() - t0))

    # descriptive statistics for the record
    df.describe().loc[['min', 'mean', 'std', 'max']].to_csv(
        os.path.join(ROOT, 'data', '%s_dataset_stats.csv' % family))

    # extrapolation set: same screening, 15% extended box
    Xe, ye, n_cand_e = sample_screened(family, N_EXT, extend=0.15,
                                       seed=SEED + 100)
    dfe = pd.DataFrame(np.column_stack([Xe, ye]), columns=cols)
    dfe.insert(0, 'design_id', np.arange(1, len(dfe) + 1))
    dfe.to_csv(os.path.join(ROOT, 'data', 'extrapolation',
                            '%s_ext.csv' % family), index=False)
    print('[%s] extrapolation set: %d designs from the extended box'
          % (family, len(dfe)))

    # ---- shared split: stratify on deflection deciles -----------------
    defl = df[TARGET_NAMES[family][0]].values
    bins = np.minimum(np.floor(pd.qcut(defl, 10, labels=False)), 9)
    ids = df['design_id'].values
    train_ids, test_ids = train_test_split(
        ids, test_size=0.2, random_state=SEED, stratify=bins)
    train_mask = np.isin(ids, train_ids)
    fold = np.full(len(df), -1, dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for k, (_, val_idx) in enumerate(skf.split(ids[train_mask],
                                               bins[train_mask])):
        fold_idx = np.where(train_mask)[0][val_idx]
        fold[fold_idx] = k
    split_col = np.where(train_mask, 'train', 'test')
    pd.DataFrame(dict(design_id=ids, split=split_col, fold=fold)).to_csv(
        os.path.join(ROOT, 'data', 'splits', '%s_folds.csv' % family),
        index=False)
    print('[%s] split: %d train / %d test, 10 stratified folds written'
          % (family, train_mask.sum(), (~train_mask).sum()))


def main():
    os.makedirs(os.path.join(ROOT, 'data', 'extrapolation'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'data', 'splits'), exist_ok=True)
    accept_rows = []
    for family in ('beam', 'frame', 'truss'):
        build_family(family, accept_rows)
    pd.DataFrame(accept_rows).to_csv(
        os.path.join(ROOT, 'data', 'acceptance_rates.csv'), index=False)
    print('datasets complete')


if __name__ == '__main__':
    main()
