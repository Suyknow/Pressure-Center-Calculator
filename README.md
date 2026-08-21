> 🌐 **English** | [中文](README_CN.md)

# Pressure-Center-Calculator

Compute the resultant **force** and **pressure center** (center of pressure) of any pressure-loaded surface from scattered surface-pressure point clouds (CFD or wind-tunnel / test data).

## Applicability

Any 3-D curved surface carrying a distributed pressure load, for example:

- **Turbomachinery**: rotor/stator blades, compressor/turbine blades, fans, propellers, rotors, wind/tidal/hydro turbine blades, pump impellers, guide vanes;
- **Aerospace**: wings (upper/lower surfaces), tail fins/stabilizers, control surfaces, fuselage sections, winglets, UAV surfaces, rocket/missile bodies and fins;
- **Automotive**: rear/front wings, splitters/diffusers, body sections;
- **Marine/offshore**: hull sections, rudders/fins, hydrofoils, sails, offshore-platform pressure panels;
- **Civil / wind engineering**: building facades/cladding, roofs (uplift center), canopies/bridges/signboards/solar panels;
- Other: hydrostatic/dynamic pressure faces (gates, dam faces, tank walls), internal-flow components, etc.

Input: 4 columns per line `x y z p` (x,y,z in [m], p in [Pa]). The point cloud is the surface shape — no extra geometry needed.

## Requirements

- Python 3.8+
- numpy, scipy (versions in `requirements.txt`)

Install: `python -m pip install -r requirements.txt`

## Usage

```
python pressure_center.py <ps.dat> [ss.dat] [--inside-point X Y Z] [--cstype aero|mech] [--pref P]
```

- `ps.dat`: pressure point cloud of one surface (`x y z p`). For a two-sided body (blade, wing upper/lower, tail surfaces) this is one side.
- `ss.dat` (optional): the opposing surface. Use it for **two-sided** bodies; omit for a **single surface** (single wing skin, wall, sail) or a closed body.
- `--inside-point X Y Z`: a point inside the body (aero-frame coordinates, m); normals point away from it → robustly fixes the wetted side. **Required for single surfaces**; not needed for two-sided (two-file) bodies (oriented automatically via the thickness direction).
- `--cstype aero|mech`: output coordinate system, default `mech`. The two differ in **axes, not only units** (see below).
- `--pref P`: subtract reference pressure P [Pa] to work in gauge; default is absolute pressure.

Output: resultant force `F=(Fx,Fy,Fz)`, `|F|`, pressure-center point (on the line of action, closest to the body centroid), line-of-action direction, residual couple.

## Coordinate systems

Internal computation is in the **aero** frame (m, N, N·m). `--cstype` switches the **display** frame. **Note: `mech` vs `aero` differ in axes, not only in units (mm vs m):**

```
aero_x =  mech_z / 1000
aero_y = -mech_y / 1000        # y negated
aero_z =  mech_x / 1000        # x and z swapped
i.e. aero = R · mech,  R = [[0,0,1],[0,-1,0],[1,0,0]], then positions /1000 (mm->m)
```

Force and moment are only rotated by `R` (units N, N·m unchanged). `--cstype` affects display only, not the computation.

## Verification (optional)

Verify the program against a known external load (force and moment at a reference point). The report is written to `verify_report.md` in the working directory.

1. Copy `external_load.sample.txt` to `external_load.txt` and fill in the reference point, force, and moment (line format, first line sets the coordinate system):

```
<cs_type>       # coordinate system: aero (positions m) or mech (positions mm)
<x> <y> <z>     # reference point r_ref (mech=mm, aero=m)
<Fx> <Fy> <Fz>  # [N]
<mx> <my> <mz>  # [N·m]
```

One value per line, 9 values total; `#` comments allowed. Position units follow `cs_type` (`mech`=mm, `aero`=m); force/moment are always N, N·m. For single-surface / closed-body verification add `--inside-point` (as above).

2. Run:

```
python verify.py --ps ps.dat [--ss ss.dat] [--inside-point X Y Z] --ext external_load.txt --tol 1e-3
```

`--tol` is the relative tolerance (`1e-3` = 0.1%). The report lists four relative errors (force, moment, pressure center, residual couple, shown as **percentages**) with per-item PASS/FAIL. Normalization scales come from the `.dat` files themselves (characteristic length = max point-cloud extent; moment scale = |F| × characteristic length) — no preset load. External `mx,my,mz` must use the right-hand rule `M=∮(r-r_ref)×dF`.

## Output meaning

- **Pressure center**: the point on the line of action closest to the body centroid (every point on the line of action has zero moment perpendicular to the force).
- **Residual couple**: a pure torque along the force axis that cannot be removed by choosing the point (inherent to a 3-D distribution). If it is small relative to `|F|×structure size`, the load is essentially a single force; otherwise include this torque in structural analysis.

## Files

| File | Purpose |
|---|---|
| `pressure_center.py` | main program (load, triangulate, integrate, pressure center, CLI) |
| `verify.py` | verify against an external load, outputs `verify_report.md` |
| `external_load.sample.txt` | external-load template (copy to `external_load.txt` and fill in) |
| `requirements.txt` | dependencies |
| `verify_report.md` | report generated by `verify.py` (appears after running) |

> Input data files (`*.dat`) must be supplied by the user and are not in the repo (gitignored to avoid leaking real data).

## Assumptions / limitations

- Each patch is a graph over its best-fit plane (no folding); suits typical wing/wall surfaces. A fully closed shell (e.g. a full fuselage) must be split into several graph-like patches.
- Two-sided bodies (two files) are oriented automatically via the thickness direction; single/closed surfaces use `--inside-point`.
- Pressure is treated as absolute by default; for a thin body the constant part nearly cancels between opposing sides, so the force is essentially correct, though the pressure center may be slightly affected. Use `--pref` for a gauge basis.

## License

Unless otherwise stated, the content of this repository and related materials are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Attribution-NonCommercial-ShareAlike 4.0: non-commercial use, with attribution and ShareAlike). See [LICENSE](LICENSE).

> Development/testing notes: see `README_DEV.md`.
