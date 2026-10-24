"""Speed benchmark: wall-clock time of the solver against surrogate
inference for each family.

The solver time is the median over 200 dataset designs of a full
static + eigenvalue solve.  Surrogate times are reported for single
prediction calls (median of 300 calls, the relevant regime for
interactive design exploration) and for batched inference of 1000
designs (median per-sample time over 30 batches).

Output: results/speed/speed_benchmark.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_pipeline  # noqa: E402
from sampling import BOXES, solve_row  # noqa: E402
from train_eval import load_family  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42
N_SOLVER_TIMED = 200


def main():
    rows = []
    for family in ('beam', 'frame', 'truss'):
        df, feats = load_family(family)

        # --- solver -----------------------------------------------------
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(df), size=min(N_SOLVER_TIMED, len(df)),
                         replace=False)
        times = []
        for i in idx:
            x = df[feats].values[i]
            t0 = time.perf_counter()
            solve_row(family, x)
            times.append(time.perf_counter() - t0)
        solver_ms = float(np.median(times)) * 1000.0

        # --- surrogate (best model of the deflection target) ------------
        from sampling import TARGET_NAMES
        target = TARGET_NAMES[family][0]
        tag = '%s_%s' % (family, target)
        summ = pd.read_csv(os.path.join(ROOT, 'results',
                                        'summary_metrics.csv'))
        sel = summ[(summ.family == family) & (summ.target == target)]
        best = sel.sort_values('cv_r2_mean').iloc[-1]
        name = best['model']
        with open(os.path.join(ROOT, 'results', 'tuning',
                               tag + '_best_params.json')) as fh:
            params = json.load(fh)[name]
        clean = {k: v for k, v in params.items() if k not in ('inner_cv_r2',
                                                              'note')}
        X, y = df[feats].values, df[target].values
        pipe = make_pipeline(name, seed=SEED, **clean).fit(X, y)
        x0 = X[idx[0]].reshape(1, -1)

        pipe.predict(x0)  # warm-up
        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            pipe.predict(x0)
            times.append(time.perf_counter() - t0)
        single_ms = float(np.median(times)) * 1000.0

        Xb = X[idx[:min(1000, len(idx))]]
        pipe.predict(Xb[:2])
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            pipe.predict(Xb)
            times.append((time.perf_counter() - t0) / len(Xb))
        batch_ms = float(np.median(times)) * 1000.0

        rows.append(dict(
            family=family, target=target, model=name,
            solver_ms_per_design=solver_ms,
            surrogate_ms_single=single_ms,
            surrogate_ms_per_sample_batch1000=batch_ms,
            speedup_single=solver_ms / single_ms,
            speedup_batch=solver_ms / batch_ms))
        print('[%s] solver %.2f ms | surrogate %.4f ms (single) / '
              '%.6f ms (batch) | speed-up %.0fx / %.0fx'
              % (family, solver_ms, single_ms, batch_ms,
                 solver_ms / single_ms, solver_ms / batch_ms))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, 'results', 'speed',
                            'speed_benchmark.csv'), index=False)


if __name__ == '__main__':
    main()
