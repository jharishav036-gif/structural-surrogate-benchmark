"""Finite element / direct stiffness solvers for the three structural
problem families of the benchmark.

All solvers use SI units internally (N, m, Pa, kg) and are implemented
with plain NumPy.  Element formulations:

* beam  : 2-node Euler-Bernoulli beam element with cubic Hermite shape
          functions, 2 DOF per node (deflection v, rotation theta),
          consistent mass matrix.
* frame : planar frame element (axial + bending), 3 DOF per node
          (u, v, theta), consistent mass matrix.
* truss : planar bar element, 2 DOF per node, consistent mass matrix.

Every solve returns the three response targets used throughout the
study: maximum deflection (mm), maximum normal stress (MPa) and the
first natural frequency (Hz).
"""

import numpy as np
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# beam family: cantilever, rectangular section, tip point load P1,
# mid-span point load P2 and uniform load q
# ---------------------------------------------------------------------------

def _beam_element(E, I, rho, A, L):
    k = E * I / L**3 * np.array([
        [12.0,   6*L, -12.0,   6*L],
        [ 6*L, 4*L*L,  -6*L, 2*L*L],
        [-12.0,  -6*L,  12.0,  -6*L],
        [ 6*L, 2*L*L,  -6*L, 4*L*L],
    ])
    m = rho * A * L / 420.0 * np.array([
        [156.0,   22*L,   54.0,  -13*L],
        [ 22*L, 4*L*L,    13*L, -3*L*L],
        [ 54.0,   13*L,  156.0,  -22*L],
        [-13*L, -3*L*L,  -22*L, 4*L*L],
    ])
    return k, m


def _udl_vector(q, L):
    # consistent nodal loads for a uniform downward load q (N/m),
    # ordered [v1, th1, v2, th2]
    return np.array([-q*L/2.0, -q*L*L/12.0, -q*L/2.0, q*L*L/12.0])


def solve_beam(L, b, h, E, rho, P1, P2, q, n_el=40):
    """Cantilever beam, rectangular section b x h.

    Parameters (SI): L [m], b [m], h [m], E [Pa], rho [kg/m3],
    P1 tip point load [N], P2 mid-span point load [N], q [N/m].
    """
    I = b * h**3 / 12.0
    A = b * h
    ndof = 2 * (n_el + 1)
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    f = np.zeros(ndof)
    le = L / n_el
    ke, me = _beam_element(E, I, rho, A, le)
    fe = _udl_vector(q, le)
    for e in range(n_el):
        sl = slice(2*e, 2*e + 4)
        K[sl, sl] += ke
        M[sl, sl] += me
        f[sl] += fe
    f[2*n_el] -= P1            # tip node, deflection DOF
    f[2*(n_el // 2)] -= P2     # mid-span node (n_el is even)
    free = np.arange(2, ndof)  # fixed end: v = th = 0 at node 0
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    # element end forces of the root element -> support moment
    qe = ke @ u[0:4] - fe
    M_root = qe[1]
    sigma = abs(M_root) * (h / 2.0) / I / 1e6      # Pa -> MPa

    w2 = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
              eigvals_only=True, subset_by_index=[0, 0])[0]
    f1 = np.sqrt(w2) / (2.0 * np.pi)

    return {
        'tip_deflection_mm': -u[2*n_el] * 1000.0,   # magnitude, downward
        'max_stress_MPa': sigma,
        'f1_Hz': f1,
        'root_moment_Nm': M_root,
    }


# ---------------------------------------------------------------------------
# frame family: fixed-base portal frame, lateral load W at beam level
# (split equally between the two joints), uniform gravity load w on the beam
# ---------------------------------------------------------------------------

def _frame_element(E, A, I, rho, L):
    k = np.zeros((6, 6))
    c1 = E * A / L
    k[0, 0] = k[3, 3] = c1
    k[0, 3] = k[3, 0] = -c1
    kb = E * I / L**3
    kbrow = np.array([
        [12.0,    6*L, -12.0,    6*L],
        [ 6*L, 4*L*L,  -6*L, 2*L*L],
        [-12.0,  -6*L,  12.0,  -6*L],
        [ 6*L, 2*L*L,  -6*L, 4*L*L]])
    bend = [1, 2, 4, 5]
    for i in range(4):
        for j in range(4):
            k[bend[i], bend[j]] += kb * kbrow[i, j]
    m = np.zeros((6, 6))
    ma = rho * A * L / 6.0
    m[0, 0] = m[3, 3] = 2.0 * ma
    m[0, 3] = m[3, 0] = ma
    mb = rho * A * L / 420.0 * np.array([
        [156.0,   22*L,   54.0,  -13*L],
        [ 22*L, 4*L*L,    13*L, -3*L*L],
        [ 54.0,   13*L,  156.0,  -22*L],
        [-13*L, -3*L*L,  -22*L, 4*L*L]])
    for i in range(4):
        for j in range(4):
            m[bend[i], bend[j]] += mb[i, j]
    return k, m


def _transform(c, s):
    T = np.array([
        [ c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return T


def solve_frame(H, S, bc, hc, bb, hb, E, rho, W, w, n_el=8):
    """Fixed-base portal frame: two columns (bc x hc) of height H, one beam
    (bb x hb) of span S.  Lateral load W [N] applied at beam level, split
    equally between the two joints; uniform load w [N/m] on the beam.
    """
    Ac, Ic = bc * hc, bc * hc**3 / 12.0
    Ab, Ib = bb * hb, bb * hb**3 / 12.0

    # node table (shared joints merged by rounded coordinates)
    nodes, node_key = [], {}

    def get_node(x, y):
        key = (round(x, 9), round(y, 9))
        if key not in node_key:
            node_key[key] = len(nodes)
            nodes.append((x, y))
        return node_key[key]

    members = []  # (n1, n2, A, I, section_depth, has_udl)
    for (x0, y0, x1, y1, Asec, Isec, dep, udl) in (
            (0.0, 0.0, 0.0, H, Ac, Ic, hc, False),
            (0.0, H, S, H, Ab, Ib, hb, True),
            (S, 0.0, S, H, Ac, Ic, hc, False)):
        chain = [get_node(x0 + (x1 - x0) * i / n_el,
                          y0 + (y1 - y0) * i / n_el) for i in range(n_el + 1)]
        for i in range(n_el):
            members.append((chain[i], chain[i + 1], Asec, Isec, dep, udl))

    nn = len(nodes)
    ndof = 3 * nn
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    f = np.zeros(ndof)

    for (n1, n2, Asec, Isec, dep, udl) in members:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        ke, me = _frame_element(E, Asec, Isec, rho, L)
        T = _transform(c, s)
        keg, meg = T.T @ ke @ T, T.T @ me @ T
        dofs = np.r_[3 * n1:3 * n1 + 3, 3 * n2:3 * n2 + 3]
        K[np.ix_(dofs, dofs)] += keg
        M[np.ix_(dofs, dofs)] += meg
        if udl:
            fl = np.zeros(6)
            fl[[1, 2, 4, 5]] = _udl_vector(w, L)
            f[dofs] += T.T @ fl

    joints = [get_node(0.0, H), get_node(S, H)]
    for j in joints:
        f[3 * j] += W / 2.0

    bases = [get_node(0.0, 0.0), get_node(S, 0.0)]
    fixed = np.r_[[3 * n for n in bases], [3 * n + 1 for n in bases],
                  [3 * n + 2 for n in bases]]
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    sway = 0.5 * (u[3 * joints[0]] + u[3 * joints[1]]) * 1000.0  # mm

    # maximum normal stress over all members (axial + bending)
    sigma_max = 0.0
    xs = np.linspace(0.0, 1.0, 21)
    for (n1, n2, Asec, Isec, dep, udl) in members:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        ke, _ = _frame_element(E, Asec, Isec, rho, L)
        T = _transform(c, s)
        dofs = np.r_[3 * n1:3 * n1 + 3, 3 * n2:3 * n2 + 3]
        fl = np.zeros(6)
        if udl:
            fl[[1, 2, 4, 5]] = _udl_vector(w, L)
        qe = ke @ (T @ u[dofs]) - fl
        N, V1, M1 = qe[0], qe[1], qe[2]
        w_loc = w if udl else 0.0
        Mx = -M1 + V1 * (xs * L) - w_loc * (xs * L)**2 / 2.0
        sig = abs(N) / Asec + np.abs(Mx) * (dep / 2.0) / Isec
        sigma_max = max(sigma_max, float(np.max(sig)) / 1e6)  # MPa

    w2 = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
              eigvals_only=True, subset_by_index=[0, 0])[0]
    f1 = np.sqrt(w2) / (2.0 * np.pi)

    return {
        'sway_mm': sway,
        'max_stress_MPa': sigma_max,
        'f1_Hz': f1,
    }


# ---------------------------------------------------------------------------
# truss family: 6-panel simply supported Pratt truss (statically determinate)
# ---------------------------------------------------------------------------

def _truss_geometry(S, Ht):
    """Nodes B0..B6 on the bottom chord, T1..T5 on the top chord.

    Members: bottom chord (B), top chord (T), verticals (V), end posts (V)
    and diagonals (D).  Member list: (n1, n2, group) with group in
    {'chord', 'vert', 'diag'}.
    """
    coords = {}
    for i in range(7):
        coords['B%d' % i] = (i * S / 6.0, 0.0)
    for i in range(1, 6):
        coords['T%d' % i] = (i * S / 6.0, Ht)
    members = []
    for i in range(6):                                  # bottom chord
        members.append(('B%d' % i, 'B%d' % (i + 1), 'chord'))
    for i in range(1, 5):                               # top chord
        members.append(('T%d' % i, 'T%d' % (i + 1), 'chord'))
    for i in range(1, 6):                               # verticals
        members.append(('B%d' % i, 'T%d' % i, 'vert'))
    members += [('B0', 'T1', 'vert'), ('B6', 'T5', 'vert')]   # end posts
    for i in range(1, 5):                               # diagonals
        members.append(('T%d' % i, 'B%d' % (i + 1), 'diag'))
    return coords, members


def solve_truss(S, Ht, A_chord, A_diag, A_vert, E, rho, P, w):
    """6-panel Pratt truss: span S [m], height Ht [m], member areas [m2]
    by group, modulus E [Pa], density rho [kg/m3], mid-span bottom-node
    point load P [N] and bottom-chord distributed load w [N/m] applied as
    statically equivalent nodal loads.
    """
    coords, members = _truss_geometry(S, Ht)
    names = list(coords)
    idx = {n: i for i, n in enumerate(names)}
    nn = len(names)
    ndof = 2 * nn
    area = {'chord': A_chord, 'diag': A_diag, 'vert': A_vert}

    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    geo = []
    for (a, b, grp) in members:
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        L = np.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        Asec = area[grp]
        k = E * Asec / L * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]])
        mfac = rho * Asec * L / 6.0
        m = mfac * np.array([
            [2*c*c, 2*c*s, c*c, c*s],
            [2*c*s, 2*s*s, c*s, s*s],
            [c*c, c*s, 2*c*c, 2*c*s],
            [c*s, s*s, 2*c*s, 2*s*s]])
        dofs = np.r_[2*idx[a]:2*idx[a] + 2, 2*idx[b]:2*idx[b] + 2]
        K[np.ix_(dofs, dofs)] += k
        M[np.ix_(dofs, dofs)] += m
        geo.append((a, b, grp, L, c, s, Asec, dofs))

    # loads: P at mid-span bottom node B3; w as tributary nodal loads
    f = np.zeros(ndof)
    f[2 * idx['B3'] + 1] -= P
    pl = S / 6.0
    for i in range(7):
        trib = pl / 2.0 if i in (0, 6) else pl
        f[2 * idx['B%d' % i] + 1] -= w * trib

    # supports: pin at B0, roller at B6
    fixed = np.array([2*idx['B0'], 2*idx['B0'] + 1, 2*idx['B6'] + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    defl = -u[2 * idx['B3'] + 1] * 1000.0   # mm, magnitude (downward)

    forces, sigma_max = [], 0.0
    for (a, b, grp, L, c, s, Asec, dofs) in geo:
        # axial elongation from global displacements (node a -> b)
        elong = (-c) * u[dofs[0]] + (-s) * u[dofs[1]] \
            + c * u[dofs[2]] + s * u[dofs[3]]
        N = E * Asec / L * elong    # tension positive
        forces.append(N)
        sigma_max = max(sigma_max, abs(N) / Asec)

    w2 = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
              eigvals_only=True, subset_by_index=[0, 0])[0]
    f1 = np.sqrt(w2) / (2.0 * np.pi)

    return {
        'mid_deflection_mm': defl,
        'max_stress_MPa': sigma_max / 1e6,
        'f1_Hz': f1,
        'member_forces_N': forces,
        'residual': float(np.max(np.abs((K @ u - f)[free]))) /
                    max(1.0, float(np.max(np.abs(f)))),
    }
