#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pressure_center.py
=======================

Validation of pressure_center.py against analytically-known cases.

Geometry: a thin blade modelled as two parallel (flat or tilted) plates:
    pressure side (ps) at z = z_ps(x)   (lower surface)
    suction side (ss) at z = z_ss(x)   (upper surface),  z_ss > z_ps
outward normals point away from the interior (down on ps, up on ss).

Key fact used for tight tolerances: for a FLAT triangle, the 1-point
centroid rule  int_T p dA = A_T * (p0+p1+p2)/3  and the moment rule using
the triangle centroid are EXACT for any pressure field that is linear in
(x, y, z). So for linear pressure on flat/tilted-planar patches the
discrete result matches the continuous integral to ~machine precision,
independent of mesh resolution.

Cases
-----
TC1  flat slab, constant pressures              -> F along z, cp = centroid
TC2  flat slab, ps linear p0(1+2x), ss = 0      -> F along z, cp shifted in x
TC3  tilted slab, ps linear p0(1+2x), ss = 0    -> F tilted, full 3-D cp test
TC4  flat slab, scattered(jittered) const p     -> robustness to irregular mesh

Run:  .venv/bin/python test_pressure_center.py
"""
import os
import sys
import tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 上级目录，pressure_center.py 所在
sys.path.insert(0, _HERE)
import pressure_center as pc


# --------------------------------------------------------------------------- #
def write_dat(path, pts, p):
    """Write a 4-column (x y z p) data file in the sample format."""
    arr = np.hstack([pts, p.reshape(-1, 1)])
    with open(path, "w") as fh:
        for row in arr:
            fh.write("  ".join("%.10e" % v for v in row))
            fh.write("\n")


def make_grid(Lx, Ly, Nx, Ny, z_of_xy, p_of_xy, jitter=0.0, seed=0):
    """Build (x,y,z,p) point cloud; jitter interior (boundary kept fixed)."""
    xs = np.linspace(0.0, Lx, Nx)
    ys = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x = X.ravel()
    y = Y.ravel()
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        dx = rng.uniform(-jitter, jitter, size=x.size)
        dy = rng.uniform(-jitter, jitter, size=x.size)
        on_boundary = (X.ravel() <= 0) | (X.ravel() >= Lx) | \
                      (Y.ravel() <= 0) | (Y.ravel() >= Ly)
        dx[on_boundary] = 0.0
        dy[on_boundary] = 0.0
        x = x + dx
        y = y + dy
    z = np.broadcast_to(np.asarray(z_of_xy(x, y), dtype=float), x.shape).copy()
    p = np.broadcast_to(np.asarray(p_of_xy(x, y), dtype=float), x.shape).copy()
    pts = np.column_stack([x, y, z])
    return pts, p


def perp_distance_to_line(P, A, d):
    """Perpendicular distance from point P to line through A with unit dir d."""
    d = d / np.linalg.norm(d)
    return np.linalg.norm(np.cross(P - A, d))


def run_case(name, ps_pts, ps_p, ss_pts, ss_p,
             F_exp, cp_on_line_point, cp_on_line_dir,
             tol_rel=1e-6, tol_abs=1e-6, residual_tol=1e-6):
    """Run analysis on generated data and compare to analytical answer."""
    tmp = tempfile.mkdtemp(prefix="pctest_%s_" % name)
    ps_path = os.path.join(tmp, "ps.dat")
    ss_path = os.path.join(tmp, "ss.dat")
    write_dat(ps_path, ps_pts, ps_p)
    write_dat(ss_path, ss_pts, ss_p)

    all_pts = np.vstack([ps_pts, ss_pts])
    bc = all_pts.mean(axis=0)
    res, _, _, _ = pc.run(ps_path, ss_path, verbose=False)
    F = res["F"]
    cp = res["cp"]

    failures = []
    # Force comparison
    Fmag = np.linalg.norm(F_exp)
    F_err = np.linalg.norm(F - F_exp)
    if F_err > max(tol_abs, tol_rel * max(1.0, Fmag)):
        failures.append("F  exp=%s got=%s err=%.3e" %
                        (F_exp.tolist(), F.tolist(), F_err))

    # cp must lie on the analytical line of action
    if cp is None:
        failures.append("cp is None (force ~0)")
    else:
        d = cp_on_line_dir / np.linalg.norm(cp_on_line_dir)
        pd = perp_distance_to_line(cp, cp_on_line_point, d)
        # account for the fact that the code reports the point closest to the
        # body centroid; that is by construction on the line of action, so the
        # perpendicular distance must be ~0.
        if pd > tol_abs:
            failures.append("cp off line of action: perp dist=%.3e" % pd)
        # residual couple must be ~0 for these planar linear cases
    rc_mag = res["residual_couple_mag"]
    if rc_mag > max(residual_tol, residual_tol * max(1.0, Fmag)):
        failures.append("residual couple=%.3e (expected ~0)" % rc_mag)

    status = "PASS" if not failures else "FAIL"
    print("[%s] %s" % (status, name))
    print("   F exp = %s" % F_exp.tolist())
    print("   F got = %s   err=%.3e" % (F.tolist(), F_err))
    if cp is not None:
        pd = perp_distance_to_line(cp,
                                   cp_on_line_point,
                                   cp_on_line_dir / np.linalg.norm(cp_on_line_dir))
        print("   cp    = %s   perp-dist-to-line=%.3e" % (cp.tolist(), pd))
    print("   residual couple |M_res| = %.3e" % rc_mag)
    for f in failures:
        print("   - %s" % f)
    return not failures


# --------------------------------------------------------------------------- #
def main():
    results = []

    # ---- TC1: flat slab, constant pressures ------------------------------- #
    Lx, Ly, t = 1.5, 1.5, 0.02
    z_ps_f = lambda x, y: -t / 2
    z_ss_f = lambda x, y: +t / 2
    p_ps_c, p_ss_c = 5000.0, 3000.0
    p_ps_f = lambda x, y: np.full_like(x, p_ps_c)
    p_ss_f = lambda x, y: np.full_like(x, p_ss_c)
    ps_pts, ps_p = make_grid(Lx, Ly, 20, 20, z_ps_f, p_ps_f)
    ss_pts, ss_p = make_grid(Lx, Ly, 20, 20, z_ss_f, p_ss_f)
    A = Lx * Ly
    F_exp = np.array([0.0, 0.0, (p_ps_c - p_ss_c) * A])   # = (0,0,4500)
    bc = np.vstack([ps_pts, ss_pts]).mean(axis=0)          # (0.75,0.75,0)
    # line of action is vertical through bc (constant pressure -> through centroid)
    results.append(run_case(
        "TC1 flat-constant", ps_pts, ps_p, ss_pts, ss_p,
        F_exp=F_exp,
        cp_on_line_point=bc, cp_on_line_dir=np.array([0.0, 0.0, 1.0])))

    # ---- TC2: flat slab, ps linear p0(1+2x), ss = 0 ----------------------- #
    Lx, Ly, t = 1.0, 1.0, 0.02
    p0 = 1000.0
    z_ps_f = lambda x, y: -t / 2
    z_ss_f = lambda x, y: +t / 2
    p_ps_f = lambda x, y: p0 * (1.0 + 2.0 * x)
    p_ss_f = lambda x, y: np.zeros_like(x)
    ps_pts, ps_p = make_grid(Lx, Ly, 30, 30, z_ps_f, p_ps_f)
    ss_pts, ss_p = make_grid(Lx, Ly, 30, 30, z_ss_f, p_ss_f)
    # F_ps = (0,0, int p dA) = (0,0, p0*2) ; ss zero
    F_exp = np.array([0.0, 0.0, 2.0 * p0])               # (0,0,2000)
    x_cp = 7.0 / 12.0     # = 0.5833333...
    y_cp = 0.5
    line_pt = np.array([x_cp, y_cp, 0.0])                # vertical line dir z
    results.append(run_case(
        "TC2 flat-linear-x", ps_pts, ps_p, ss_pts, ss_p,
        F_exp=F_exp,
        cp_on_line_point=line_pt, cp_on_line_dir=np.array([0.0, 0.0, 1.0])))

    # ---- TC3: tilted slab, ps linear p0(1+2x), ss = 0 --------------------- #
    Lx, Ly, t = 1.0, 1.0, 0.02
    a = 0.3
    p0 = 1000.0
    z_ps_f = lambda x, y: a * x - t / 2
    z_ss_f = lambda x, y: a * x + t / 2
    p_ps_f = lambda x, y: p0 * (1.0 + 2.0 * x)
    p_ss_f = lambda x, y: np.zeros_like(x)
    ps_pts, ps_p = make_grid(Lx, Ly, 30, 30, z_ps_f, p_ps_f)
    ss_pts, ss_p = make_grid(Lx, Ly, 30, 30, z_ss_f, p_ss_f)
    # F = (-2*a*p0, 0, 2*p0)  (ps only; n_ps=(a,0,-1)/sqrt(1+a^2))
    F_exp = np.array([-2.0 * a * p0, 0.0, 2.0 * p0])      # (-600,0,2000)
    x_cp = 7.0 / 12.0
    y_cp = 0.5
    # pressure centroid on ps surface in 3-D
    P0 = np.array([x_cp, y_cp, a * x_cp - t / 2])
    line_dir = np.array([-a, 0.0, 1.0])                  # direction of F
    results.append(run_case(
        "TC3 tilted-linear-x", ps_pts, ps_p, ss_pts, ss_p,
        F_exp=F_exp,
        cp_on_line_point=P0, cp_on_line_dir=line_dir))

    # ---- TC4: flat slab, scattered jittered points, constant pressure ---- #
    Lx, Ly, t = 1.5, 1.5, 0.02
    p_ps_c, p_ss_c = 5000.0, 3000.0
    z_ps_f = lambda x, y: -t / 2
    z_ss_f = lambda x, y: +t / 2
    p_ps_f = lambda x, y: np.full_like(x, p_ps_c)
    p_ss_f = lambda x, y: np.full_like(x, p_ss_c)
    # identical jitter on both sides (same seed) -> same (x,y) footprint
    ps_pts, ps_p = make_grid(Lx, Ly, 40, 40, z_ps_f, p_ps_f, jitter=1e-3, seed=7)
    ss_pts, ss_p = make_grid(Lx, Ly, 40, 40, z_ss_f, p_ss_f, jitter=1e-3, seed=7)
    A = Lx * Ly
    F_exp = np.array([0.0, 0.0, (p_ps_c - p_ss_c) * A])
    # For constant pressure the line of action passes through the AREA
    # centroid of the union of triangulations. With the boundary kept fixed
    # on the rectangle, the triangulation tiles [0,Lx]x[0,Ly] exactly, so the
    # area centroid is the rectangle centroid (Lx/2, Ly/2) -- INDEPENDENT of
    # interior jitter (which only perturbs the point-cloud mean, not the area
    # centroid). So the correct reference is the rectangle centroid, not bc.
    rect_centroid = np.array([Lx / 2.0, Ly / 2.0, 0.0])
    results.append(run_case(
        "TC4 scattered-constant", ps_pts, ps_p, ss_pts, ss_p,
        F_exp=F_exp,
        cp_on_line_point=rect_centroid,
        cp_on_line_dir=np.array([0.0, 0.0, 1.0]),
        tol_rel=1e-6, tol_abs=1e-6))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 40)
    print("SUMMARY: %d/%d passed" % (sum(results), len(results)))
    print("=" * 40)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
