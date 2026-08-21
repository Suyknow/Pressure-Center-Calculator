#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py
=========

用“已知积分点的外部载荷”验算 pressure_center.py 的结果是否自洽/准确。

外部载荷：在某参考点 r_ref 处，给出力 F_k=(Fx,Fy,Fz) 和该点弯矩
M_k=(mx,my,mz)，约定为右手系  M = ∮ (r - r_ref) x dF。

原理
----
合力 F 与参考点无关 -> 直接比。
力矩 M 随参考点变 -> 用转移公式在 r_ref 与本体质心 bc 间换算：

    M_A = M_B + (B - A) x F        (F 不变)

由此对四项做比对：合力、力矩、压力中心、剩余力偶。

用法
----
1) 真实验算（你有外部载荷文件）：
   外部载荷文件 external_load.txt 为行格式（首行 cs_type + xyz(3) + F(3) + M(3)）：
       mech            # cs_type: aero(位置 m) 或 mech(位置 mm)
       x  y  z         # 参考点 r_ref（mech=mm, aero=m）
       Fx Fy Fz        # [N]
       mx my mz        # [N·m]
   python verify.py --ps srv_ps.dat --ss srv_ss.dat --ext external_load.txt --tol 1e-3
   结果写入同目录 verify_report.md；--tol 为相对容差（1e-3 = 0.1%）。
   (可选 --pref P_REF 与程序一致的参考压力扣除)

2) 自测（无需外部真值，验证本脚本的转移公式与坐标转换自洽）：
   python verify.py --selftest
   用内嵌小叶片，任选一个 r_ref，由程序自身输出“反推”出外部载荷，
   再喂回 verify，应在机器精度内完全吻合。
"""
import os
import sys
import argparse
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pressure_center as pc
from pressure_center import (mech_to_aero_pos, mech_to_aero_vec,
                             aero_to_mech_pos, aero_to_mech_vec,
                             to_cs, cs_pos_unit)


# --------------------------------------------------------------------------- #
def analyze(ps_path, ss_path, reference_pressure=None):
    """复用 pressure_center 的内部函数，拿到 F、关于 bc 的力矩 M_bc 等。"""
    ps_pts, ps_p = pc.load_surface(ps_path)
    ss_pts, ss_p = pc.load_surface(ss_path)
    if reference_pressure is not None:
        ps_p = ps_p - reference_pressure
        ss_p = ss_p - reference_pressure
    all_pts = np.vstack([ps_pts, ss_pts])
    bc = all_pts.mean(axis=0)
    F, M_bc, area, n_tri = pc.integrate_patches(ps_pts, ps_p, ss_pts, ss_p, bc)
    res = pc.pressure_center(F, M_bc, bc)
    # 特征长度：点云各轴跨度的最大值（叶片最大尺寸），用于把绝对误差归一为
    # 相对误差——这样阈值与坐标系/单位无关，也不依赖任何预设载荷。
    ptp = all_pts.max(axis=0) - all_pts.min(axis=0)
    L_char = float(np.max(ptp)) if ptp.size else 1.0
    if not (L_char > 1e-9):
        L_char = 1.0
    return {
        "F": F, "M_bc": M_bc, "bc": bc,
        "cp": res["cp"], "F_hat": res["F_hat"],
        "M_res": res["residual_couple"], "M_res_mag": res["residual_couple_mag"],
        "area": area, "n_tri": n_tri,
        "n_ps": len(ps_pts), "n_ss": len(ss_pts),
        "L_char": L_char,
    }


def load_external(path):
    """读取外部积分点载荷，按行格式（UTF-8，支持 # 注释）：

        第 1 行 : cs_type —— aero(气动系,位置 m) 或 mech(结构系,位置 mm)
        第 2-4 行: x, y, z  参考点 r_ref（mech 时 mm，aero 时 m）
        第 5-7 行: Fx, Fy, Fz [N]
        第 8-10 行: mx, my, mz [N·m]（右手系 M=∮(r-r_ref)×dF）

    每行可带 # 注释；纯注释行和空行被跳过（按“数据行”顺序解析）。
    返回 (cs_type, r_ref, F, M)，均为**输入坐标系**下的值。
    """
    data_lines = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                data_lines.append(line)
    if len(data_lines) < 10:
        raise ValueError("外部载荷文件至少需 10 个数据行：cs_type + xyz(3)"
                         " + F(3) + M(3)（实际 %d 行）" % len(data_lines))
    cs = data_lines[0].split()[0].lower()
    if cs == "aero":
        cs_type = "aero"
    elif cs == "mech":
        cs_type = "mech"
    else:
        raise ValueError("第一行 cs_type 应为 aero 或 mech，得到 %r" % cs)
    nums = []
    for ln in data_lines[1:10]:
        tok = ln.split()[0]
        try:
            nums.append(float(tok))
        except ValueError:
            raise ValueError("无法解析数值 %r（行：%r）" % (tok, ln))
    r_ref = np.asarray(nums[0:3], dtype=float)
    F = np.asarray(nums[3:6], dtype=float)
    M = np.asarray(nums[6:9], dtype=float)
    return cs_type, r_ref, F, M


def _fmt(v):
    return "  ".join("% .6e" % x for x in np.asarray(v).ravel())


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / max(1.0, np.linalg.norm(b))


def _pct(rel):
    """相对值 -> 百分比字符串（nan/None -> N/A）。"""
    if rel is None:
        return "N/A"
    try:
        if np.isnan(rel):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    return "%.4g%%" % (float(rel) * 100.0)


def verify(prog, cs_type, r_ref_in, F_in, M_in, tol=1e-3,
           out_path="verify_report.md", verbose=True):
    """比对程序结果(prog, 气动系) 与外部载荷(输入坐标系 cs_type)。

    数学在气动系(m, N, N·m)下做（cp 公式要求单位一致）；显示时把所有量
    转到 cs_type 指定的坐标系（mech: 位置 mm，力/力矩旋转；aero: 不变）。
    相对误差与坐标系无关；绝对误差按 cs 单位显示。
    结果写入 markdown 文件 out_path（默认同目录 verify_report.md）；终端只
    打印简短摘要（避免 Windows cmd 长输出难看）。out_path=None 时不写文件。
    """
    # 外部输入 -> 气动系
    if cs_type == "mech":
        r_ref = mech_to_aero_pos(r_ref_in)
        F_k = mech_to_aero_vec(F_in)
        M_k = mech_to_aero_vec(M_in)
    else:
        r_ref = np.asarray(r_ref_in, dtype=float)
        F_k = np.asarray(F_in, dtype=float)
        M_k = np.asarray(M_in, dtype=float)

    bc = prog["bc"]
    F_p = prog["F"]
    M_bc_p = prog["M_bc"]

    # ---- 数学全在气动系 ----
    dF = np.linalg.norm(F_p - F_k)
    F_scale = max(np.linalg.norm(F_p), np.linalg.norm(F_k))
    relF = dF / F_scale if F_scale > 1e-12 else float("nan")

    M_ref_from_prog = M_bc_p + np.cross(bc - r_ref, F_p)
    dM = np.linalg.norm(M_ref_from_prog - M_k)

    Fk_norm2 = float(np.dot(F_k, F_k))
    if Fk_norm2 < 1e-30:
        cp_ext = None
        dcp = float("nan")
        perp = float("nan")
    else:
        M_bc_ext = M_k + np.cross(r_ref - bc, F_k)
        cp_ext = bc + np.cross(F_k, M_bc_ext) / Fk_norm2
        dcp = float(np.linalg.norm(prog["cp"] - cp_ext))
        d = F_k / np.sqrt(Fk_norm2)
        perp = float(np.linalg.norm(np.cross(prog["cp"] - cp_ext, d)))

    Fk_hat = F_k / np.sqrt(Fk_norm2) if Fk_norm2 > 1e-30 else np.zeros(3)
    M_res_ext = float(np.dot(M_k, Fk_hat)) * Fk_hat
    dMres = float(np.linalg.norm(prog["M_res"] - M_res_ext))

    if Fk_norm2 > 1e-30:
        Fk_mag = np.sqrt(Fk_norm2)
        M_par_ext = float(np.dot(M_k, Fk_hat)) * Fk_hat
        M_perp_ext = M_k - M_par_ext
        equiv_offset_m = float(np.linalg.norm(M_perp_ext)) / Fk_mag   # [m]
    else:
        M_par_ext = np.zeros(3)
        M_perp_ext = np.zeros(3)
        equiv_offset_m = float("nan")

    # 数据驱动归一：尺度全部来自 .dat 文件（点云特征长度 L_char 与实际力幅），
    # 不依赖任何预设载荷/几何。所有检查均为相对误差（无量纲），tol 即相对容差。
    L_char = prog.get("L_char", 1.0)
    M_scale = F_scale * L_char                       # 力矩特征尺度 [N·m]
    relM = dM / M_scale if M_scale > 1e-12 else float("nan")
    rel_cp = dcp / L_char if (L_char > 1e-12 and cp_ext is not None) else float("nan")
    rel_Mres = dMres / M_scale if M_scale > 1e-12 else float("nan")

    def _chk(err):
        return (not np.isnan(err)) and err <= tol
    checks = [
        ("合力 F", relF, _chk(relF)),
        ("力矩 M(r_ref)", relM, _chk(relM)),
        ("压力中心 cp", rel_cp, _chk(rel_cp)),
        ("剩余力偶 M_res", rel_Mres, _chk(rel_Mres)),
    ]
    ok = all(c[2] for c in checks)

    # ---- 显示：转到用户坐标系，生成 markdown ----
    pos_unit = cs_pos_unit(cs_type)
    def P(v, is_pos=True):
        return to_cs(v, cs_type, is_pos)
    def vc(v, is_pos=True):
        return "`%s`" % _fmt(P(v, is_pos))
    scale = 1000.0 if cs_type == "mech" else 1.0
    L_disp = L_char * scale
    M_scale_disp = F_scale * L_char
    nan = float("nan")

    L = []
    L.append("# 验算报告：程序结果 vs 外部积分点载荷")
    L.append("")
    L.append("- 坐标系: **%s**（位置[%s]，力[N]，力矩[N·m]）" % (cs_type, pos_unit))
    L.append("- 数据驱动尺度（来自 .dat）: 特征长度 L = %.4e %s，"
             "力矩尺度 |F|·L = %.4e N·m" % (L_disp, pos_unit, M_scale_disp))
    L.append("- 相对容差 tol = %.3e（= %s）；所有误差按上述尺度归一为相对值"
             % (tol, _pct(tol)))
    L.append("")
    L.append("## 输入")
    L.append("")
    L.append("| 量 | 值 |")
    L.append("|---|---|")
    L.append("| 外部参考点 r_ref [%s] | %s |" % (pos_unit, vc(r_ref)))
    L.append("| 本体质心 bc [%s] | %s |" % (pos_unit, vc(bc)))
    L.append("")
    L.append("## 合力 F [N]")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 程序 | %s |" % vc(F_p, False))
    L.append("| 外部 | %s |" % vc(F_k, False))
    L.append("| 绝对差 | %.6e N |" % dF)
    L.append("| 相对差 | %s |" % _pct(relF))
    L.append("")
    L.append("## 力矩（关于 r_ref）[N·m]")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 程序（由本体质心 bc 转移） | %s |" % vc(M_ref_from_prog, False))
    L.append("| 外部 | %s |" % vc(M_k, False))
    L.append("| 绝对差 | %.6e N·m |" % dM)
    L.append("| 相对差 | %s |" % _pct(relM))
    L.append("")
    if cp_ext is not None:
        L.append("## 压力中心 cp [%s]（位于合力作用线上，最近 bc）" % pos_unit)
        L.append("")
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        L.append("| 程序 | %s |" % vc(prog["cp"]))
        L.append("| 外部（由外部 F, M 求得） | %s |" % vc(cp_ext))
        L.append("| 距离差 | %.6e %s |" % (dcp * scale, pos_unit))
        L.append("| 相对差 | %s |" % _pct(rel_cp))
        L.append("| cp 到外部作用线垂直距 | %.6e %s |" % (perp * scale, pos_unit))
        L.append("")
    L.append("## 剩余力偶 M_res [N·m]（与坐标系、参考点无关的不变量）")
    L.append("")
    L.append("| 项 | 向量 | 大小 |")
    L.append("|---|---|---|")
    L.append("| 程序 | %s | %.3e |" % (vc(prog["M_res"], False), prog["M_res_mag"]))
    L.append("| 外部 | %s | %.3e |"
             % (vc(M_res_ext, False), float(np.linalg.norm(M_res_ext))))
    L.append("| 绝对差 |  | %.6e N·m |" % dMres)
    L.append("| 相对差 |  | %s |" % _pct(rel_Mres))
    L.append("")
    L.append("## 外部弯矩 M 分解（关于 r_ref）")
    L.append("")
    L.append("| 分量 | 向量 | 大小 |")
    L.append("|---|---|---|")
    L.append("| 沿力方向分量（剩余力偶）[N·m] | %s | %.3e |"
             % (vc(M_par_ext, False), float(np.linalg.norm(M_par_ext))))
    L.append("| 垂直力方向分量 [N·m] | %s | %.3e |"
             % (vc(M_perp_ext, False), float(np.linalg.norm(M_perp_ext))))
    L.append("")
    if np.isnan(equiv_offset_m):
        L.append("- r_ref 到外部作用线垂直距: N/A（外部力 ≈ 0）")
    else:
        L.append("- r_ref 到外部作用线垂直距: **%.4f %s** (= |M_⊥| / |F|)"
                  % (equiv_offset_m * scale, pos_unit))
    L.append("")
    L.append("> 注: 沿力方向分量为不可约的剩余力偶（纯扭矩，与参考点选取无关）；"
             "当 r_ref 取在合力作用线上时，垂直力方向分量 / |F| 即为外部载荷"
             "作用线相对参考点的垂直偏离（位置偏差）。")
    L.append("")
    L.append("## 检查结果")
    L.append("")
    L.append("| 项 | 结果 | 相对误差 |")
    L.append("|---|---|---|")
    for name, err, passed in checks:
        tag = "PASS" if passed else ("N/A" if np.isnan(err) else "FAIL")
        L.append("| %s | %s | %s |" % (name, tag, _pct(err)))
    L.append("")
    L.append("**整体: %s**" % ("PASS" if ok else "FAIL"))
    L.append("")
    report = "\n".join(L)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    if verbose:
        where = out_path if out_path else "(未写文件)"
        print("验算报告已写入: %s" % where)
        for name, err, passed in checks:
            tag = "PASS" if passed else ("N/A" if np.isnan(err) else "FAIL")
            print("  [%s] %s  相对误差=%s" % (tag, name, _pct(err)))
        print("整体: %s" % ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------- #
def _selftest_blade():
    """自包含的小弯叶片，供 --selftest 用（不依赖 dev/ 下的 demo_realistic）。"""
    chord, span, t, p0 = 0.2, 0.3, 0.01, 5000.0
    nch, nsp = 40, 40
    s = np.linspace(0.0, 1.0, nch)
    z = np.linspace(0.0, span, nsp)
    S, Z = np.meshgrid(s, z, indexing="ij")
    x = S * chord
    camber = 0.03 * 4.0 * S * (1.0 - S)
    y_ps = camber - t / 2.0
    y_ss = camber + t / 2.0
    p_ps = p0 * (1.0 + 0.5 * np.sin(np.pi * S) + 0.2 * (1.0 - Z / span))
    p_ss = p0 * (0.3 + 0.2 * np.sin(np.pi * S) + 0.1 * (1.0 - Z / span))
    ps = np.column_stack([x.ravel(), y_ps.ravel(), Z.ravel()])
    ss = np.column_stack([x.ravel(), y_ss.ravel(), Z.ravel()])
    return ps, p_ps.ravel(), ss, p_ss.ravel()


def _write_dat(path, pts, p):
    arr = np.hstack([pts, p.reshape(-1, 1)])
    with open(path, "w", encoding="utf-8") as fh:
        for row in arr:
            fh.write("  ".join("%.10e" % v for v in row) + "\n")


def selftest():
    """自包含自测：用内嵌小叶片，任选 r_ref，由程序输出反推外部载荷再喂回
    verify。应在机器精度内完全吻合 -> 验证力矩转移与坐标转换自洽。"""
    ps_pts, ps_p, ss_pts, ss_p = _selftest_blade()
    tmp = tempfile.mkdtemp(prefix="verify_self_")
    ps_path = os.path.join(tmp, "ps.dat")
    ss_path = os.path.join(tmp, "ss.dat")
    _write_dat(ps_path, ps_pts, ps_p)
    _write_dat(ss_path, ss_pts, ss_p)

    prog = analyze(ps_path, ss_path)
    # 任选一个外部参考点（气动系，原点）
    r_ref = np.array([0.0, 0.0, 0.0])
    F_k = prog["F"]
    # 由程序关于 bc 的力矩反推 r_ref 处的力矩：M(r_ref) = M_bc + (bc - r_ref) x F
    M_k = prog["M_bc"] + np.cross(prog["bc"] - r_ref, prog["F"])
    print("自测：cs=aero, r_ref = " + _fmt(r_ref))
    print("（外部载荷由程序输出反推，故应完全吻合）")
    return verify(prog, "aero", r_ref, F_k, M_k, tol=1e-9,
                  out_path=None, verbose=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="验算 pressure_center.py 结果")
    p.add_argument("--ps", help="压力面数据文件")
    p.add_argument("--ss", help="吸力面数据文件")
    p.add_argument("--ext", help="外部载荷文件（行格式：cs_type + xyz(3) + F(3) + M(3)）")
    p.add_argument("--pref", type=float, default=None, help="参考压力扣除 [Pa]")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="相对容差（1e-3 = 0.1%%，默认 1e-3）")
    p.add_argument("--out", default="verify_report.md",
                   help="报告输出 markdown 文件（默认同目录 verify_report.md）")
    p.add_argument("--selftest", action="store_true", help="运行自洽自测")
    args = p.parse_args(argv)

    if args.selftest:
        ok = selftest()
        return 0 if ok else 1

    if not (args.ps and args.ss and args.ext):
        p.error("需要 --ps --ss --ext，或使用 --selftest")
    prog = analyze(args.ps, args.ss, reference_pressure=args.pref)
    cs_type, r_ref, F_k, M_k = load_external(args.ext)
    ok = verify(prog, cs_type, r_ref, F_k, M_k, tol=args.tol,
                out_path=args.out, verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
