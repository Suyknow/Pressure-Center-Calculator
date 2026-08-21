#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pressure_center.py
=================

Compute the resultant aerodynamic force and the pressure center (center of
pressure) of a stator blade from scattered surface pressure point clouds.

Input
-----
Two plain-text files, one for the pressure side (ps) and one for the suction
side (ss). Each file contains rows of four whitespace-separated columns:

    x  y  z  p

with x,y,z in [m] (blade-surface coordinates) and p in [Pa].

Pressures are treated as **absolute** by default (used as-is, no
subtraction). To work in a gauge basis instead (e.g. to approximate the
closed-body load on the open ps+ss surface, or to match a colleague's
reference), pass ``--pref P_REF`` to subtract a reference pressure from
both sides before integration. For a thin blade the constant (ambient)
part nearly cancels between ps and ss, so the force is approximately
correct even in absolute mode; the pressure center / small residuals
may be slightly affected.

The point cloud (the (x,y,z) of all rows) describes the blade surface; no
extra blade-geometry data is required.

Physics
-------
The pressure on a surface element dA with outward unit normal n produces an
elementary force on the body

    dF = -p * n * dA            (pressure pushes opposite to n)

The resultant force and moment about a reference point r0 are

    F  = -∮ p n dA
    M  = ∮ (r - r0) x dF

"Pressure center" in 3D
-----------------------
In 2D the loading reduces to a single force at a unique point. In 3D the set
of points with zero net moment is generally a *line* (the line of action of
the resultant force), and a generic 3D distribution can additionally leave an
irreducible *residual couple* parallel to F (a pure torque about the force
axis) that cannot be removed by moving the reference point.

This program therefore reports:
  * the resultant force F (Fx, Fy, Fz) and its magnitude |F|;
  * the *line of action*: a point on it + the unit direction F_hat;
  * the pressure-center point = the point on the line of action that is
    *closest to the blade body centroid* (a physically meaningful point that
    lies on / near the blade, unlike the arbitrary origin);
  * the residual couple M_res = (M . F_hat) F_hat, which is invariant to the
    reference point. If |M_res| is ~0 the loading is a pure single force and
    the pressure center is a true zero-moment point; otherwise a small
    torsional couple remains (reported).

Numerics
-------
The surface is a 2D manifold sampled by scattered points. For each patch
(ps, ss):
  1. PCA finds the best-fit 2D plane (the two largest-variance axes).
  2. Points are projected to that plane and a 2D Delaunay triangulation is
     built; triangles are lifted back to 3D.
  3. Per triangle: area = 0.5*|e1 x e2|, normal = (e1 x e2)/|e1 x e2|.
     Force and moment are integrated with the 3-point edge-midpoint rule
     (degree-2 exact): pressure at each edge midpoint is the linear average
     of its two vertex pressures. This is exact for any pressure field that
     is linear in (x,y,z) over a flat triangle, so it integrates both the
     force (linear p) and the moment (linear r x linear p, degree-2) exactly.
Outward-normal orientation is fixed per patch by pointing each patch's
normals *away from the other patch's centroid* (uses the thickness
direction, so it is robust to camber -- the body-centroid heuristic would
flip inward on pressure-side triangles near a camber peak). Falls back to
the body-centroid heuristic when the two patches are coincident. Spurious
long triangles created by the Delaunay convex hull on concave boundaries
are dropped with an edge-length filter.

Python: 3.8+ compatible. Dependencies: numpy, scipy.
"""

import sys
import argparse
import numpy as np
from scipy.spatial import Delaunay


# --------------------------------------------------------------------------- #
#  坐标系转换：结构系 mech (mm) <-> 气动系 aero (m)
#
#  关系：aero = (mech_z, -mech_y, mech_x) / 1000  （x、z 互换，y 取反，mm->m）
#  即 aero = R @ mech，R=[[0,0,1],[0,-1,0],[1,0,0]]，det(R)=+1（真旋转，180°
#  绕 (1,0,1) 轴）。因是真旋转（非镜像），位置(极矢量)与力矩(轴矢量)都用同
#  一个 R 变换；力/力矩单位不变（N、N·m），只有位置在 mech 用 mm。
#  R 对称正交且 R=R^{-1}=R^T，正反变换用同一个 R。
# --------------------------------------------------------------------------- #
R_MECH2AERO = np.array([[0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0]])


def mech_to_aero_pos(p):
    return R_MECH2AERO @ np.asarray(p, dtype=float) / 1000.0


def mech_to_aero_vec(v):
    return R_MECH2AERO @ np.asarray(v, dtype=float)


def aero_to_mech_pos(p):
    return R_MECH2AERO @ np.asarray(p, dtype=float) * 1000.0


def aero_to_mech_vec(v):
    return R_MECH2AERO @ np.asarray(v, dtype=float)


def to_cs(aero_vec, cs_type, is_position):
    """把气动系下的量转到 cs_type 坐标系用于显示。"""
    v = np.asarray(aero_vec, dtype=float)
    if cs_type == "mech":
        v = R_MECH2AERO @ v
        if is_position:
            v = v * 1000.0
    return v


def cs_pos_unit(cs_type):
    return "mm" if cs_type == "mech" else "m"


# --------------------------------------------------------------------------- #
#  I/O
# --------------------------------------------------------------------------- #
def load_surface(path):
    """Load a 4-column (x y z p) surface file, skipping non-numeric lines.

    Returns
    -------
    pts : (N,3) float ndarray
    p   : (N,)   float ndarray
    """
    pts_list = []
    p_list = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.split()
            if len(parts) != 4:
                continue
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue  # e.g. placeholder lines like "......"
            pts_list.append(vals[:3])
            p_list.append(vals[3])
    if not pts_list:
        raise ValueError("No numeric data rows found in '%s'" % path)
    pts = np.asarray(pts_list, dtype=float)
    p = np.asarray(p_list, dtype=float)
    return pts, p


# --------------------------------------------------------------------------- #
#  Triangulation of a scattered surface patch
# --------------------------------------------------------------------------- #
def triangulate_patch(pts, edge_factor=4.0, circum_factor=2.5):
    """Triangulate a near-planar surface patch via PCA + 2D Delaunay.

    Parameters
    ----------
    pts : (N,3) points of one surface patch.
    edge_factor : triangles whose longest edge exceeds
        edge_factor * median(edge length) are dropped (removes convex-hull
        spurs on concave boundaries). <=0 disables filtering.
    circum_factor : triangles whose circumradius exceeds
        circum_factor * median(circumradius) are dropped (alpha-shape-style;
        better respects curved/concave boundaries such as the leading/trailing
        edge, where 2D Delaunay would otherwise fill outside the real
        surface). <=0 disables filtering.

    Returns
    -------
    tris : (M,3) int array of point indices forming valid triangles.
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    if len(pts) < 3:
        raise ValueError("Surface patch has only %d point(s); at least 3 "
                         "non-collinear points are required to build a "
                         "triangulation. (The shipped sample has 1 row just "
                         "to show the format; real files have ~10k rows.)"
                         % len(pts))
    # PCA via covariance eigendecomposition (symmetric, stable).
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    # two largest-variance directions -> best-fit plane basis
    basis = eigvecs[:, [2, 1]]
    coords2d = centered @ basis
    try:
        tri = Delaunay(coords2d)
    except Exception as exc:  # noqa
        raise RuntimeError("Delaunay triangulation failed: %s" % exc)
    tris = tri.simplices

    if edge_factor and edge_factor > 0 and len(tris):
        # edge-length filter
        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        max_edge = np.maximum(np.maximum(e01, e12), e20)
        med = np.median(np.concatenate([e01, e12, e20]))
        if med > 0:
            keep = max_edge <= edge_factor * med
            tris = tris[keep]

    if circum_factor and circum_factor > 0 and len(tris):
        # alpha-shape-style circumradius filter: drops large triangles that
        # span across concave boundaries (e.g. curved leading/trailing edge)
        # where the 2D Delaunay fills the convex hull outside the real
        # surface. R = a*b*c / (4*Area); skinny/long spanning triangles have
        # large R. Threshold relative to the median -> parameter-free.
        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        a = np.linalg.norm(v1 - v0, axis=1)
        b = np.linalg.norm(v2 - v1, axis=1)
        c = np.linalg.norm(v0 - v2, axis=1)
        cross = np.cross(v1 - v0, v2 - v0)
        area2 = np.linalg.norm(cross, axis=1)
        safe = np.where(area2 > 1e-18, area2, 1.0)
        R = (a * b * c) / (2.0 * safe)          # R = abc/(4A) = abc/(2*|cross|)
        medR = np.median(R)
        if medR > 0:
            keep = R <= circum_factor * medR
            tris = tris[keep]
    return tris


# --------------------------------------------------------------------------- #
#  Force / moment integration over both patches
# --------------------------------------------------------------------------- #
def integrate_patches(ps_pts, ps_p, ss_pts, ss_p, body_centroid):
    """Integrate pressure force over ps and ss patches.

    Returns (F, M_about_body_centroid, total_area, n_triangles).
    """
    F = np.zeros(3)
    M = np.zeros(3)
    total_area = 0.0
    n_tri = 0

    # Curvature-robust outward orientation: each patch's outward normal
    # points *away from the other patch*. This uses the thickness direction
    # (robust to camber) rather than the body centroid (which on a cambered
    # blade can sit above pressure-side triangles near the camber peak and
    # flip their normals inward). Falls back to the body-centroid heuristic
    # when the two patches are coincident (no thickness information).
    c_ps = ps_pts.mean(axis=0)
    c_ss = ss_pts.mean(axis=0)
    sep = c_ss - c_ps
    sep_norm = np.linalg.norm(sep)
    if sep_norm > 1e-12:
        outward_dir = {"ps": -sep, "ss": sep}     # away from the other patch
    else:
        outward_dir = None                         # fall back per-triangle

    for label, pts, p in (("ps", ps_pts, ps_p), ("ss", ss_pts, ss_p)):
        if len(pts) < 3:
            continue
        tris = triangulate_patch(pts)
        if len(tris) == 0:
            continue
        n_tri += len(tris)

        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)                # (M,3), |cross| = 2*area
        twice_area = np.linalg.norm(cross, axis=1)
        # guard zero-area
        safe = np.where(twice_area > 1e-18, twice_area, 1.0)
        n = cross / safe[:, None]               # unit normals (M,3)
        area = 0.5 * twice_area

        # 3-point edge-midpoint quadrature (degree-2 exact). This integrates
        # both the force (linear p) AND the moment (linear r x linear p,
        # i.e. degree-2 integrand) exactly for linear pressure fields, and is
        # 2nd-order accurate for smooth pressure. Midpoint values are exact
        # for the linear interpolant we assume on each triangle.
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        p01 = 0.5 * (p[tris[:, 0]] + p[tris[:, 1]])
        p12 = 0.5 * (p[tris[:, 1]] + p[tris[:, 2]])
        p20 = 0.5 * (p[tris[:, 2]] + p[tris[:, 0]])
        w = area / 3.0                          # weight per midpoint (M,)

        # Orient normals outward.
        if outward_dir is not None:
            d_out = outward_dir[label]
            # make all triangles consistent with the patch-mean normal first
            # (fixes any stray triangles from Delaunay degeneracies; safe for
            # graph surfaces whose normals vary by < 90 deg, as for blades)
            n_mean = n.sum(axis=0)
            n_mean = n_mean / (np.linalg.norm(n_mean) + 1e-30)
            disagree = np.sum(n * n_mean, axis=1) < 0
            n[disagree] = -n[disagree]
            # patch-level outward flip (all at once)
            if np.dot(n_mean, d_out) < 0:
                n = -n
        else:
            # fallback: per-triangle body-centroid heuristic
            r_c = (v0 + v1 + v2) / 3.0
            flip = np.sum(n * (r_c - body_centroid), axis=1) < 0
            n[flip] = -n[flip]

        dF01 = -(p01[:, None] * n) * w[:, None]
        dF12 = -(p12[:, None] * n) * w[:, None]
        dF20 = -(p20[:, None] * n) * w[:, None]
        F += (dF01 + dF12 + dF20).sum(axis=0)
        # moment about body centroid
        M += (np.cross(m01 - body_centroid, dF01)
              + np.cross(m12 - body_centroid, dF12)
              + np.cross(m20 - body_centroid, dF20)).sum(axis=0)
        total_area += area.sum()

    return F, M, total_area, n_tri


# --------------------------------------------------------------------------- #
#  Pressure center / line of action
# --------------------------------------------------------------------------- #
def pressure_center(F, M, ref_point):
    """Compute line of action and the cp closest to ref_point.

    Parameters
    ----------
    F : (3,) resultant force.
    M : (3,) moment about ref_point.
    ref_point : (3,) reference point (here the body centroid).

    Returns dict with F, |F|, F_hat, cp, line point, residual couple, etc.
    """
    F = np.asarray(F, dtype=float)
    M = np.asarray(M, dtype=float)
    Fmag2 = float(np.dot(F, F))
    Fmag = float(np.sqrt(Fmag2))
    out = {
        "F": F,
        "Fmag": Fmag,
        "Fx": float(F[0]),
        "Fy": float(F[1]),
        "Fz": float(F[2]),
    }
    if Fmag2 < 1e-30:
        out.update(F_hat=np.zeros(3), cp=None, line_point=None,
                   residual_couple=M, residual_couple_mag=float(np.linalg.norm(M)))
        return out

    F_hat = F / Fmag
    # point on line of action closest to ref_point
    # (r - ref) = (F x M) / |F|^2   with M = moment about ref_point
    r_off = np.cross(F, M) / Fmag2
    cp = ref_point + r_off
    # residual couple parallel to F (invariant to reference point)
    M_res = float(np.dot(M, F_hat)) * F_hat
    out.update(F_hat=F_hat, cp=cp, line_point=cp, residual_couple=M_res,
               residual_couple_mag=float(np.linalg.norm(M_res)),
               r_off=r_off)
    return out


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "  ".join("% .6e" % x for x in v)


def run(ps_path, ss_path, reference_pressure=None, cs_type="mech",
        verbose=True):
    ps_pts, ps_p = load_surface(ps_path)
    ss_pts, ss_p = load_surface(ss_path)
    if reference_pressure is not None:
        ps_p = ps_p - reference_pressure
        ss_p = ss_p - reference_pressure

    all_pts = np.vstack([ps_pts, ss_pts])
    body_centroid = all_pts.mean(axis=0)

    for label, pts in (("ps", ps_pts), ("ss", ss_pts)):
        if len(pts) < 3:
            raise ValueError(
                "Surface '%s' has only %d point(s); need >=3 non-collinear "
                "points. (The shipped sample has 1 row just to show the "
                "format; real files have ~10k rows.)" % (label, len(pts)))

    F, M, area, n_tri = integrate_patches(ps_pts, ps_p, ss_pts, ss_p,
                                          body_centroid)
    res = pressure_center(F, M, body_centroid)

    if verbose:
        pos_unit = cs_pos_unit(cs_type)
        scale = 1000.0 if cs_type == "mech" else 1.0
        def P(v, is_pos=True):
            return to_cs(v, cs_type, is_pos)
        print("=" * 64)
        print("Pressure-center analysis (stator blade)")
        print("输出坐标系: %s（位置[%s]，力[N]，力矩[N·m]）" % (cs_type, pos_unit))
        print("=" * 64)
        print("PS points: %d   SS points: %d" % (len(ps_pts), len(ss_pts)))
        print("Triangles used: %d   total area: %.6e m^2" % (n_tri, area))
        print("Body centroid [%s]: " % pos_unit + _fmt(P(body_centroid)))
        if reference_pressure is not None:
            print("压力模式: 表压（已减参考压 p_ref = %.6e Pa）" % reference_pressure)
        else:
            print("压力模式: 绝对压力（不减参考压）")
        print("-" * 64)
        Fdisp = P(res["F"], False)
        print("Resultant force F [N]:")
        print("  Fx = % .6e" % Fdisp[0])
        print("  Fy = % .6e" % Fdisp[1])
        print("  Fz = % .6e" % Fdisp[2])
        print("  |F| = %.6e" % res["Fmag"])
        print("-" * 64)
        if res["cp"] is None:
            print("Resultant force ~ 0: pressure center undefined.")
        else:
            cpdisp = P(res["cp"])
            print("Pressure-center point [%s] (on line of action," % pos_unit)
            print("                          closest to body centroid):")
            print("  x_cp = % .6e" % cpdisp[0])
            print("  y_cp = % .6e" % cpdisp[1])
            print("  z_cp = % .6e" % cpdisp[2])
            print("Line-of-action direction F_hat: " + _fmt(P(res["F_hat"], False)))
            print("Residual couple (parallel to F) [N.m]: "
                  + _fmt(P(res["residual_couple"], False)))
            print("  |residual couple| = %.6e N.m" % res["residual_couple_mag"])
            if res["residual_couple_mag"] > 1e-9 * max(1.0, res["Fmag"]):
                print("  NOTE: non-zero residual couple => the 3D loading cannot be")
                print("        reduced to a pure single force; a small torque about")
                print("        the force axis remains. The reported point is still on")
                print("        the line of action (zero moment perpendicular to F).")
            else:
                print("  residual ~ 0 => loading is a pure single force at the")
                print("  reported pressure-center point (true zero-moment point).")
        print("=" * 64)
    return res, body_centroid, area, n_tri


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute resultant force & pressure center of a stator "
                    "blade from ps/ss surface pressure point clouds.")
    parser.add_argument("ps", help="pressure-side data file (x y z p)")
    parser.add_argument("ss", help="suction-side data file (x y z p)")
    parser.add_argument("--pref", type=float, default=None,
                        help="换算到表压时减去的参考压 p_ref [Pa]。"
                              "默认按绝对压力处理（不减）；若要表压基准"
                              "（如闭合体近似、或与同事对齐参考压），"
                              "用它减去参考压。")
    parser.add_argument("--cstype", default="mech",
                        choices=["aero", "mech"],
                        help="输出坐标系：aero(气动系,m) 或 mech(结构系,mm)。"
                             "默认 mech。仅影响显示，内部计算仍在气动系(m)进行。")
    args = parser.parse_args(argv)
    cs_type = args.cstype
    try:
        run(args.ps, args.ss, reference_pressure=args.pref,
            cs_type=cs_type, verbose=True)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.stderr.write("Error: %s\n" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
