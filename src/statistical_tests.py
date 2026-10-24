"""Friedman omnibus tests and Holm-corrected pairwise comparisons of the
nine-model tournament.

For every family/target combination the models are compared on their
held-out absolute errors: the Friedman test (Friedman 1937; Demsar 2006)
ranks the nine models separately on every test design and checks the
null hypothesis that they are interchangeable.  When the null is
rejected, the best-ranked model is compared against each competitor
with a paired Wilcoxon signed-rank test, and the family of eight
comparisons is corrected with the step-down Holm-Bonferroni procedure.
A summary of the mean Friedman ranks across all nine combinations
provides the overall tournament ranking.

The Wilcoxon tests in results/heldout/*_wilcoxon_vs_best.csv are the
uncorrected counterparts of the same comparisons; this module adds the
omnibus test, the multiplicity correction and the cross-combination
ranking.

Outputs:
  results/stats/friedman_omnibus.csv
  results/stats/friedman_holm.csv
  results/stats/model_ranks_overall.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODEL_NAMES  # noqa: E402
from sampling import TARGET_NAMES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ALPHA = 0.05
FAMILIES = ('beam', 'frame', 'truss')


def holm_bonferroni(pvals):
    """Step-down Holm-Bonferroni adjustment of a family of p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.zeros(m)
    running = 0.0
    for i, idx in enumerate(np.argsort(p)):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def combo_tables(family, target):
    """Friedman ranks, omnibus test and Holm-corrected comparisons for
    one family/target combination."""
    ho = pd.read_csv(os.path.join(
        ROOT, 'results', 'heldout',
        '%s_%s_predictions.csv' % (family, target)))
    piv = ho.pivot(index='design_id', columns='model', values='y_pred')
    ytrue = ho.groupby('design_id')['y_true'].first()
    abserr = piv.sub(ytrue, axis=0).abs()[MODEL_NAMES]

    # rank the nine models on every test design (1 = smallest error,
    # ties receive average ranks)
    ranks = np.apply_along_axis(rankdata, 1, abserr.values)
    mean_ranks = ranks.mean(axis=0)

    stat, p = friedmanchisquare(*[abserr[m].values for m in MODEL_NAMES])
    best = MODEL_NAMES[int(mean_ranks.argmin())]

    omnibus = dict(
        family=family, target=target, n_designs=int(len(abserr)),
        friedman_stat=float(stat), friedman_p=float(p),
        best_by_mean_rank=best, best_mean_rank=float(mean_ranks.min()))

    rows = []
    others = [m for m in MODEL_NAMES if m != best]
    raw = []
    for m in others:
        if np.allclose(abserr[best].values, abserr[m].values):
            raw.append(1.0)
        else:
            raw.append(float(wilcoxon(abserr[best].values,
                                      abserr[m].values)[1]))
    adj = holm_bonferroni(raw)
    for m, rp, ap in zip(others, raw, adj):
        rows.append(dict(
            family=family, target=target, model=m, vs=best,
            mean_rank_model=float(mean_ranks[MODEL_NAMES.index(m)]),
            mean_rank_best=float(mean_ranks.min()),
            wilcoxon_p=rp, holm_p=ap,
            significant_holm=bool(ap < ALPHA)))
    return omnibus, rows, dict(zip(MODEL_NAMES, mean_ranks))


def main():
    stats_dir = os.path.join(ROOT, 'results', 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    omnibus_rows, pair_rows = [], []
    rank_acc = {m: [] for m in MODEL_NAMES}
    for family in FAMILIES:
        for target in TARGET_NAMES[family]:
            omnibus, rows, mean_ranks = combo_tables(family, target)
            omnibus_rows.append(omnibus)
            pair_rows += rows
            for m in MODEL_NAMES:
                rank_acc[m].append(mean_ranks[m])
            print('[%s/%s] Friedman chi2 = %.1f, p = %.2e | best by mean '
                  'rank: %s (%.2f)'
                  % (family, target, omnibus['friedman_stat'],
                     omnibus['friedman_p'], omnibus['best_by_mean_rank'],
                     omnibus['best_mean_rank']))

    pd.DataFrame(omnibus_rows).to_csv(
        os.path.join(stats_dir, 'friedman_omnibus.csv'), index=False)
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(stats_dir, 'friedman_holm.csv'), index=False)

    overall = pd.DataFrame({
        'model': MODEL_NAMES,
        'mean_friedman_rank': [float(np.mean(rank_acc[m]))
                               for m in MODEL_NAMES],
        'n_best': [sum(1 for o in omnibus_rows
                       if o['best_by_mean_rank'] == m)
                   for m in MODEL_NAMES],
    }).sort_values('mean_friedman_rank').reset_index(drop=True)
    overall.to_csv(os.path.join(stats_dir, 'model_ranks_overall.csv'),
                   index=False)

    n_sig = sum(r['significant_holm'] for r in pair_rows)
    print('Holm-corrected comparisons: %d of %d significant at alpha = %.2f'
          % (n_sig, len(pair_rows), ALPHA))
    print(overall.to_string(index=False))


if __name__ == '__main__':
    main()
