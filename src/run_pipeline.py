"""Run the full pipeline or a single stage.

Stages:
  verify   solver verification against closed-form solutions
  data     dataset generation and split construction
  train    tuning + 10-fold CV + held-out + extrapolation for all
           nine family/target combinations
  analyze  SHAP analysis, sensitivity comparison and learning curves
  speed    solver vs surrogate speed benchmark
  stats    Friedman omnibus tests + Holm-corrected comparisons
  screen   100k-candidate design-screening demonstration

Usage:
  python src/run_pipeline.py --stage all
"""

import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = sys.executable
SRC = os.path.join(ROOT, 'src')

FAMILIES = ('beam', 'frame', 'truss')
TARGETS = {
    'beam': ['tip_deflection_mm', 'max_stress_MPa', 'f1_Hz'],
    'frame': ['sway_mm', 'max_stress_MPa', 'f1_Hz'],
    'truss': ['mid_deflection_mm', 'max_stress_MPa', 'f1_Hz'],
}


def run(script, *args):
    t0 = time.time()
    cmd = [PY, os.path.join(SRC, script)] + list(args)
    print('$ ' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    status = 'ok' if r.returncode == 0 else 'FAILED (%d)' % r.returncode
    print('-- %s finished in %.1f s [%s]' % (script, dt, status), flush=True)
    return r.returncode


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['verify', 'data', 'train', 'analyze', 'speed',
                             'stats', 'screen', 'all'])
    args = ap.parse_args()
    t0 = time.time()
    rc = 0

    if args.stage in ('verify', 'all'):
        rc |= run('verification.py')
    if args.stage in ('data', 'all') and rc == 0:
        rc |= run('generate_data.py')
    if args.stage in ('train', 'all') and rc == 0:
        for family in FAMILIES:
            for target in TARGETS[family]:
                rc |= run('train_eval.py', '--family', family,
                          '--target', target)
    if args.stage in ('analyze', 'all') and rc == 0:
        for family in FAMILIES:
            for target in TARGETS[family]:
                rc |= run('shap_analysis.py', '--family', family,
                          '--target', target)
                rc |= run('learning_curves.py', '--family', family,
                          '--target', target)
    if args.stage in ('speed', 'all') and rc == 0:
        rc |= run('speed_benchmark.py')
    if args.stage in ('stats', 'all') and rc == 0:
        rc |= run('statistical_tests.py')
    if args.stage in ('screen', 'all') and rc == 0:
        rc |= run('screening_demo.py')

    print('pipeline finished in %.1f s (exit %d)'
          % (time.time() - t0, rc), flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
