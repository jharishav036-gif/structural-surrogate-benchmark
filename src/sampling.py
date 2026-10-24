"""Parameter boxes and Latin hypercube sampling with physical screening.

Designs that violate serviceability or strength limits are rejected and
resampled so that every record in the datasets represents a physically
sensible structure:

* beam  : tip deflection <= L/100 and max stress <= 350 MPa
* frame : sway <= H/100 and max stress <= 350 MPa
* truss : mid-span deflection <= S/250 and max stress <= 350 MPa

The extrapolation sets are drawn from boxes extended by 15% on both
sides of every parameter range and pass through the same screening.
"""

import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers import solve_beam, solve_frame, solve_truss  # noqa: E402


# name: (label, unit, low, high) -- SI units except areas in mm2 (input
# convenience; converted to m2 before solving)
BOXES = {
    'beam': [
        ('L', 'm', 2.0, 6.0),
        ('b', 'm', 0.15, 0.40),
        ('h', 'm', 0.25, 0.70),
        ('E', 'GPa', 68.0, 210.0),
        ('rho', 'kg/m3', 2650.0, 7900.0),
        ('P1', 'kN', 5.0, 60.0),
        ('P2', 'kN', 0.0, 40.0),
        ('q', 'kN/m', 0.0, 25.0),
    ],
    'frame': [
        ('H', 'm', 3.0, 7.0),
        ('S', 'm', 4.0, 10.0),
        ('bc', 'm', 0.20, 0.45),
        ('hc', 'm', 0.25, 0.60),
        ('bb', 'm', 0.20, 0.45),
        ('hb', 'm', 0.30, 0.70),
        ('E', 'GPa', 68.0, 210.0),
        ('rho', 'kg/m3', 2650.0, 7900.0),
        ('W', 'kN', 10.0, 120.0),
        ('w', 'kN/m', 5.0, 60.0),
    ],
    'truss': [
        ('S', 'm', 8.0, 20.0),
        ('Ht', 'm', 1.5, 4.0),
        ('A_chord', 'mm2', 1500.0, 5000.0),
        ('A_diag', 'mm2', 800.0, 3000.0),
        ('A_vert', 'mm2', 800.0, 3000.0),
        ('E', 'GPa', 68.0, 210.0),
        ('rho', 'kg/m3', 2650.0, 7900.0),
        ('P', 'kN', 20.0, 120.0),
        ('w', 'kN/m', 5.0, 40.0),
    ],
}

STRESS_CAP = 350.0   # MPa


def box_array(family, extend=0.0):
    lo = np.array([b[2] for b in BOXES[family]])
    hi = np.array([b[3] for b in BOXES[family]])
    span = hi - lo
    return lo - extend * span, hi + extend * span


def solve_row(family, x):
    """Solve one design.  x follows the column order of BOXES[family]
    with the input units used there."""
    if family == 'beam':
        L, b, h, E, rho, P1, P2, q = x
        r = solve_beam(L, b, h, E * 1e9, rho, P1 * 1e3, P2 * 1e3, q * 1e3)
        return r['tip_deflection_mm'], r['max_stress_MPa'], r['f1_Hz'], \
            r['tip_deflection_mm'] <= 1000.0 * L / 100.0 and \
            r['max_stress_MPa'] <= STRESS_CAP
    if family == 'frame':
        H, S, bc, hc, bb, hb, E, rho, W, w = x
        r = solve_frame(H, S, bc, hc, bb, hb, E * 1e9, rho,
                        W * 1e3, w * 1e3)
        return r['sway_mm'], r['max_stress_MPa'], r['f1_Hz'], \
            r['sway_mm'] <= 1000.0 * H / 100.0 and \
            r['max_stress_MPa'] <= STRESS_CAP
    S, Ht, Ac, Ad, Av, E, rho, P, w = x
    r = solve_truss(S, Ht, Ac * 1e-6, Ad * 1e-6, Av * 1e-6, E * 1e9, rho,
                    P * 1e3, w * 1e3)
    return r['mid_deflection_mm'], r['max_stress_MPa'], r['f1_Hz'], \
        r['mid_deflection_mm'] <= 1000.0 * S / 250.0 and \
        r['max_stress_MPa'] <= STRESS_CAP


TARGET_NAMES = {
    'beam': ['tip_deflection_mm', 'max_stress_MPa', 'f1_Hz'],
    'frame': ['sway_mm', 'max_stress_MPa', 'f1_Hz'],
    'truss': ['mid_deflection_mm', 'max_stress_MPa', 'f1_Hz'],
}


def sample_screened(family, n, extend=0.0, seed=42):
    """Latin hypercube sample of n physically admissible designs.

    Returns (X, y, n_candidates) where X is (n, d) in input units and
    y is (n, 3) target values.  Sampling proceeds in batches with
    deterministic seeds until n admissible designs are collected.
    """
    lo, hi = box_array(family, extend)
    d = len(lo)
    X, y = [], []
    n_cand, batch, rng_seed = 0, max(4 * n, 500), seed
    while len(X) < n:
        s = qmc.LatinHypercube(d=d, seed=rng_seed).random(batch)
        cand = lo + s * (hi - lo)
        for row in cand:
            n_cand += 1
            t1, t2, t3, ok = solve_row(family, row)
            if ok:
                X.append(row)
                y.append([t1, t2, t3])
                if len(X) == n:
                    break
        rng_seed += 1
    return np.array(X), np.array(y), n_cand
