"""Large-scale design-screening demonstration.

Scenario: an engineer has to find every design in a 100,000-candidate
Latin hypercube sample of a family's parameter box that satisfies a
representative design brief - a deflection limit, a stress cap and a
minimum first natural frequency - of the kind used for
vibration-sensitive support structures.  Screening the candidates with
the surrogate models takes seconds; the question is whether the
surrogate shortlist survives verification by the finite-element solver.

Protocol per family:
  1. refit the best model of each target (selected by CV R2 in the
     tournament) on the training partition - identical configuration to
     the final tournament fits, which is asserted against the stored
     held-out predictions
  2. draw 100,000 candidates (LHS, seed 123) from the raw parameter box
  3. surrogate screen: predict the three targets, apply the brief
  4. verify the selected designs with the solver -> precision of the
     shortlist (up to 40,000 solves, otherwise a random subsample)
  5. solve a random sample of 2,000 candidates regardless of selection
     -> recall of the screen, and 90% split-conformal prediction
     intervals (multiplicative form) calibrated on the held-out
     partition
  6. rank the verified feasible designs by their worst utilisation
     ratio and export the top ten

Design briefs (limits representative of vibration-sensitive
serviceability and strength checks):
  beam : tip deflection <= L/500, stress <= 60 MPa, f1 >= 30 Hz
  frame: sway <= H/600, stress <= 35 MPa, f1 >= 12 Hz
  truss: mid-span deflection <= S/300, stress <= 180 MPa, f1 >= 45 Hz

Outputs:
  results/screening/screening_summary.csv
  results/screening/screening_conformal.csv
  results/screening/screening_breakeven.csv
  results/screening/{family}_top_designs.csv
  results/screening/{family}_verified_sample.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_pipeline  # noqa: E402
from sampling import BOXES, TARGET_NAMES, box_array, solve_row  # noqa: E402
from train_eval import load_family  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEED = 42
SEED_CANDIDATES = 123
N_CANDIDATES = 100000
N_RECALL_SAMPLE = 2000
VERIFY_CAP = 40000
POOL_EXPORT = 5000
CONF_LEVEL = 0.90

FAMILIES = ('beam', 'frame', 'truss')

# feature that the deflection limit scales with, per family
SPAN_FEATURE = {'beam': 'L', 'frame': 'H', 'truss': 'S'}

BRIEFS = {
    'beam': dict(defl_div=500.0, stress_cap=60.0, f1_min=30.0),
    'frame': dict(defl_div=600.0, stress_cap=35.0, f1_min=12.0),
    'truss': dict(defl_div=300.0, stress_cap=180.0, f1_min=45.0),
}
BRIEF_TEXT = {
    'beam': 'tip deflection <= L/500; stress <= 60 MPa; f1 >= 30 Hz',
    'frame': 'sway <= H/600; stress <= 35 MPa; f1 >= 12 Hz',
    'truss': 'mid deflection <= S/300; stress <= 180 MPa; f1 >= 45 Hz',
}


def brief_pass(family, X, Y):
    """Mask of designs satisfying the brief.

    X: (n, d) feature array in input units; Y: (n, 3) response array
    [deflection mm, stress MPa, f1 Hz] - either simulated or predicted.
    """
    brief = BRIEFS[family]
    feats = [b[0] for b in BOXES[family]]
    span = X[:, feats.index(SPAN_FEATURE[family])]
    return ((Y[:, 0] <= 1000.0 * span / brief['defl_div'])
            & (Y[:, 1] <= brief['stress_cap'])
            & (Y[:, 2] >= brief['f1_min']))


def solve_many(family, X):
    """Solve an array of designs; returns targets (n, 3), admissibility
    flags (n,) and per-design solve times in seconds."""
    t1 = np.zeros(len(X))
    t2 = np.zeros(len(X))
    t3 = np.zeros(len(X))
    ok = np.zeros(len(X), dtype=bool)
    times = np.zeros(len(X))
    for i, x in enumerate(X):
        t0 = time.perf_counter()
        a, b, c, k = solve_row(family, x)
        times[i] = time.perf_counter() - t0
        t1[i], t2[i], t3[i], ok[i] = a, b, c, k
    return np.column_stack([t1, t2, t3]), ok, times


def refit_best(family, target, df, feats):
    """Refit the tournament's best model for one target and check that
    it reproduces the stored held-out predictions."""
    summ = pd.read_csv(os.path.join(ROOT, 'results', 'summary_metrics.csv'))
    sel = summ[(summ.family == family) & (summ.target == target)]
    best = sel.sort_values('cv_r2_mean').iloc[-1]
    name = best['model']
    with open(os.path.join(ROOT, 'results', 'tuning',
                           '%s_%s_best_params.json' % (family, target))) as fh:
        params = json.load(fh)[name]
    clean = {k: v for k, v in params.items()
             if k not in ('inner_cv_r2', 'note')}

    train = df['split'].values == 'train'
    t0 = time.perf_counter()
    pipe = make_pipeline(name, seed=SEED, **clean).fit(
        df[feats].values[train], df[target].values[train])
    fit_s = time.perf_counter() - t0

    # the refit must reproduce the tournament's held-out predictions
    test = ~train
    p = pipe.predict(df[feats].values[test])
    ref = pd.read_csv(os.path.join(
        ROOT, 'results', 'heldout',
        '%s_%s_predictions.csv' % (family, target)))
    ref = ref[ref.model == name].sort_values('design_id')
    order = np.argsort(df['design_id'].values[test])
    dmax = float(np.max(np.abs(p[order] - ref['y_pred'].values)))
    if dmax > 1e-6:
        raise AssertionError('refit mismatch for %s/%s: %.2e'
                             % (family, target, dmax))
    return name, pipe, fit_s + float(best['tuning_time_s'])


def conformal_q90(family, target, name):
    """90% split-conformal radius from the held-out partition of the best
    model, using multiplicative residual scores |y - yhat| / yhat so that
    the interval yhat * [1 - q, 1 + q] adapts to the response magnitude
    (the targets span up to three orders of magnitude, so a single
    absolute radius would not transfer).  The radius is the
    ceil((n+1)*level)-th order statistic of the scores - the standard
    finite-sample split-conformal quantile."""
    ho = pd.read_csv(os.path.join(
        ROOT, 'results', 'heldout',
        '%s_%s_predictions.csv' % (family, target)))
    ho = ho[ho.model == name]
    yhat = ho.y_pred.values
    y = ho.y_true.values
    pos = yhat > 0.0
    scores = np.where(pos, np.abs(y - yhat) / np.where(pos, yhat, 1.0),
                      np.inf)
    n = len(scores)
    k = int(np.ceil((n + 1) * CONF_LEVEL))
    return float(np.sort(scores)[min(k - 1, n - 1)])


def conformal_covered(pred_col, true_col, q):
    """Coverage of the multiplicative interval yhat * [1 - q, 1 + q]
    (non-positive predictions are never covered)."""
    pos = pred_col > 0.0
    rel = np.where(pos, np.abs(true_col - pred_col)
                   / np.where(pos, pred_col, 1.0), np.inf)
    return rel <= q


def run_family(family):
    df, feats = load_family(family)
    targets = TARGET_NAMES[family]

    # ---- 1. refit the three best models ------------------------------
    models, prep_s = {}, 0.0
    for target in targets:
        name, pipe, cost_s = refit_best(family, target, df, feats)
        models[target] = (name, pipe)
        prep_s += cost_s
        print('[%s] refit %-6s for %-18s (prep %.1f s incl. tuning)'
              % (family, name, target, cost_s))

    # ---- 2. candidate sample -----------------------------------------
    lo, hi = box_array(family)
    Xc = lo + qmc.LatinHypercube(d=len(lo), seed=SEED_CANDIDATES) \
        .random(N_CANDIDATES) * (hi - lo)

    # ---- 3. surrogate screen -----------------------------------------
    t0 = time.perf_counter()
    pred = np.column_stack([models[t][1].predict(Xc) for t in targets])
    t_screen = time.perf_counter() - t0
    sel = brief_pass(family, Xc, pred)
    n_sel = int(sel.sum())
    print('[%s] screened %d candidates in %.2f s -> %d selected (%.1f%%)'
          % (family, N_CANDIDATES, t_screen, n_sel, 100.0 * n_sel / N_CANDIDATES))

    # ---- 4. verification of the shortlist ----------------------------
    idx_sel = np.where(sel)[0]
    verified_all = len(idx_sel) <= VERIFY_CAP
    if verified_all:
        idx_ver = idx_sel
    else:
        idx_ver = np.random.default_rng(SEED_CANDIDATES).choice(
            idx_sel, VERIFY_CAP, replace=False)
    truth_ver, _, _ = solve_many(family, Xc[idx_ver])
    true_pass_ver = brief_pass(family, Xc[idx_ver], truth_ver)
    precision = float(true_pass_ver.mean())
    print('[%s] verified %d shortlisted designs: precision %.3f'
          % (family, len(idx_ver), precision))

    # ---- 5. random sample: recall, solve time, conformal coverage ----
    rng = np.random.default_rng(SEED_CANDIDATES + 1)
    idx_rec = rng.choice(N_CANDIDATES, N_RECALL_SAMPLE, replace=False)
    truth_rec, admissible, solve_times = solve_many(family, Xc[idx_rec])
    true_pass_rec = brief_pass(family, Xc[idx_rec], truth_rec)
    n_pass_rec = int(true_pass_rec.sum())
    recall = (float((true_pass_rec & sel[idx_rec]).sum() / n_pass_rec)
              if n_pass_rec else float('nan'))
    solver_ms = float(np.median(solve_times)) * 1000.0
    print('[%s] recall on %d-design sample: %.3f (%d truly feasible)'
          % (family, N_RECALL_SAMPLE, recall, n_pass_rec))

    conf_rows = []
    for j, target in enumerate(targets):
        name = models[target][0]
        q90 = conformal_q90(family, target, name)
        yhat = pred[idx_rec][:, j]
        ytrue = truth_rec[:, j]
        covered = conformal_covered(yhat, ytrue, q90)
        cov_all = float(covered.mean())
        cov_adm = float(covered[admissible].mean())
        mape_ver = float(np.mean(np.abs(
            (truth_ver[:, j] - pred[idx_ver][:, j]) / truth_ver[:, j])) * 100)
        conf_rows.append(dict(
            family=family, target=target, model=name, level=CONF_LEVEL,
            conformal_radius_rel=q90, coverage_random_sample=cov_all,
            coverage_admissible=cov_adm,
            mape_verified_pool=mape_ver))
        print('[%s] %-18s q90 %.3f (relative) | coverage %.3f '
              '(admissible %.3f) | MAPE on pool %.1f%%'
              % (family, target, q90, cov_all, cov_adm, mape_ver))

    # ---- 6. top designs by worst utilisation --------------------------
    span_col = feats.index(SPAN_FEATURE[family])
    brief = BRIEFS[family]
    defl_limit = (1000.0 * Xc[idx_ver][:, span_col] / brief['defl_div'])
    util = np.column_stack([
        truth_ver[:, 0] / defl_limit,
        truth_ver[:, 1] / brief['stress_cap'],
        brief['f1_min'] / truth_ver[:, 2]])
    util_max = util.max(axis=1)

    feasible = true_pass_ver
    order = np.argsort(util_max[feasible])
    top = idx_ver[feasible][order][:10]

    cols = (['rank'] + feats
            + ['pred_defl_mm', 'true_defl_mm', 'defl_limit_mm',
               'pred_stress_MPa', 'true_stress_MPa', 'stress_cap_MPa',
               'pred_f1_Hz', 'true_f1_Hz', 'f1_min_Hz',
               'util_defl', 'util_stress', 'util_f1', 'util_max'])
    truth_top, _, _ = solve_many(family, Xc[top])
    rows = []
    for r, (i, tv) in enumerate(zip(top, truth_top), start=1):
        dlim = 1000.0 * Xc[i, span_col] / brief['defl_div']
        rows.append([r] + [float(v) for v in Xc[i]]
                    + [float(pred[i, 0]), float(tv[0]), float(dlim),
                       float(pred[i, 1]), float(tv[1]),
                       float(brief['stress_cap']),
                       float(pred[i, 2]), float(tv[2]),
                       float(brief['f1_min']),
                       float(tv[0] / dlim), float(tv[1] / brief['stress_cap']),
                       float(brief['f1_min'] / tv[2]),
                       float(max(tv[0] / dlim, tv[1] / brief['stress_cap'],
                                 brief['f1_min'] / tv[2]))])
    top_df = pd.DataFrame(rows, columns=cols)
    top_df.to_csv(os.path.join(
        ROOT, 'results', 'screening',
        '%s_top_designs.csv' % family), index=False)

    # verified pool export (random subsample, for figures)
    keep = np.random.default_rng(SEED_CANDIDATES + 2).choice(
        len(idx_ver), min(POOL_EXPORT, len(idx_ver)), replace=False)
    pool = pd.DataFrame(Xc[idx_ver][keep], columns=feats)
    for j, target in enumerate(targets):
        pool['pred_' + target] = pred[idx_ver][keep][:, j]
        pool['true_' + target] = truth_ver[keep][:, j]
    pool['verified_feasible'] = true_pass_ver[keep]
    pool.to_csv(os.path.join(
        ROOT, 'results', 'screening',
        '%s_verified_sample.csv' % family), index=False)

    # ---- summary rows --------------------------------------------------
    solver_est_s = N_CANDIDATES * solver_ms / 1000.0
    summary = dict(
        family=family, brief=BRIEF_TEXT[family],
        n_candidates=N_CANDIDATES, n_selected=n_sel,
        select_rate=n_sel / N_CANDIDATES,
        n_verified=len(idx_ver), verified_all=verified_all,
        n_verified_feasible=int(true_pass_ver.sum()),
        precision=precision,
        n_recall_sample=N_RECALL_SAMPLE,
        n_feasible_in_sample=n_pass_rec, recall=recall,
        surrogate_screen_s=t_screen,
        solver_ms_per_design=solver_ms,
        solver_screen_est_s=solver_est_s,
        screen_speedup=solver_est_s / t_screen,
        surrogate_prep_s=prep_s)
    print('[%s] solver %.2f ms/design -> brute-force screen %.0f s vs '
          '%.2f s surrogate (%.0fx); surrogate prep %.0f s'
          % (family, solver_ms, solver_est_s, t_screen,
             summary['screen_speedup'], prep_s))
    return summary, conf_rows


def main():
    out_dir = os.path.join(ROOT, 'results', 'screening')
    os.makedirs(out_dir, exist_ok=True)

    summaries, conf_rows = [], []
    for family in FAMILIES:
        summary, conf = run_family(family)
        summaries.append(summary)
        conf_rows += conf

    pd.DataFrame(summaries).to_csv(
        os.path.join(out_dir, 'screening_summary.csv'), index=False)
    pd.DataFrame(conf_rows).to_csv(
        os.path.join(out_dir, 'screening_conformal.csv'), index=False)

    br = pd.DataFrame(summaries)[['family', 'surrogate_prep_s',
                                  'solver_ms_per_design',
                                  'surrogate_screen_s', 'n_candidates']]
    br['surrogate_ms_per_eval'] = (br['surrogate_screen_s']
                                   / br['n_candidates'] * 1000.0)
    br['breakeven_queries'] = (br['surrogate_prep_s']
                               / (br['solver_ms_per_design'] / 1000.0
                                  - br['surrogate_ms_per_eval'] / 1000.0))
    br[['family', 'surrogate_prep_s', 'solver_ms_per_design',
        'surrogate_ms_per_eval', 'breakeven_queries']].to_csv(
        os.path.join(out_dir, 'screening_breakeven.csv'), index=False)
    print('screening demonstration complete')


if __name__ == '__main__':
    main()
