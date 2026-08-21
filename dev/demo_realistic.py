#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_realistic.py
=================

Smoke test at realistic scale: generate a cambered stator blade with
~10,000 points per side, run the full analysis, and report timing.
(No analytical check here -- the rigorous validation is in
test_pressure_center.py. This only confirms the pipeline runs fast and
produces plausible numbers on real-sized scattered data.)
"""
import os
import sys
import time
import tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 上级目录，pressure_center.py 所在
sys.path.insert(0, _HERE)
import pressure_center as pc


def make_blade(Nchord=100, Nspan=100, chord=0.2, span=0.3,
               max_camber=0.03, thickness=0.01, p0=5000.0, seed=1):
    """Cambered blade: parabolic camber line, finite thickness, extruded in z."""
    s = np.linspace(0.0, 1.0, Nchord)          # chordwise frac
    z = np.linspace(0.0, span, Nspan)           # spanwise
    S, Z = np.meshgrid(s, z, indexing="ij")
    x = S * chord
    camber = 4.0 * max_camber * S * (1.0 - S)   # parabolic, peak at mid-chord
    # pressure-side (lower) and suction-side (upper) surfaces
    y_ps = camber - thickness / 2.0
    y_ss = camber + thickness / 2.0
    # smooth pressure: higher on ps, lower on ss, with spanwise taper
    p_ps = p0 * (1.0 + 0.5 * np.sin(np.pi * S) + 0.2 * (1.0 - Z / span))
    p_ss = p0 * (0.3 + 0.2 * np.sin(np.pi * S) + 0.1 * (1.0 - Z / span))

    ps_pts = np.column_stack([x.ravel(), y_ps.ravel(), Z.ravel()])
    ss_pts = np.column_stack([x.ravel(), y_ss.ravel(), Z.ravel()])
    return ps_pts, p_ps.ravel(), ss_pts, p_ss.ravel()


def write_dat(path, pts, p):
    arr = np.hstack([pts, p.reshape(-1, 1)])
    with open(path, "w") as fh:
        for row in arr:
            fh.write("  ".join("%.10e" % v for v in row) + "\n")


def main():
    ps_pts, ps_p, ss_pts, ss_p = make_blade()
    print("Generated blade: %d ps points, %d ss points"
          % (len(ps_pts), len(ss_pts)))

    tmp = tempfile.mkdtemp(prefix="pcdemo_")
    ps_path = os.path.join(tmp, "ps.dat")
    ss_path = os.path.join(tmp, "ss.dat")
    t0 = time.time()
    write_dat(ps_path, ps_pts, ps_p)
    write_dat(ss_path, ss_pts, ss_p)
    t_io = time.time() - t0

    t0 = time.time()
    pc.run(ps_path, ss_path, verbose=True)
    t_run = time.time() - t0
    print("write time: %.3f s   analysis time: %.3f s" % (t_io, t_run))


if __name__ == "__main__":
    main()
