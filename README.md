# Pressure-Center-Calculator

**静子叶片压力中心计算** — Compute the resultant aerodynamic force and the pressure center (center of pressure) of a stator blade from two surface-pressure point clouds (pressure side / suction side).

从两个表面压力点云文件（压力面 `ps`、吸力面 `ss`）计算静子叶片的气动合力与压力中心。

输入文件每行 4 列 `x y z p`（x,y,z 单位 m，p 单位 Pa）。

## 依赖

- Python 3.8+
- numpy、scipy（版本见 `requirements.txt`）

安装：`python -m pip install -r requirements.txt`

## 计算

```
python pressure_center.py srv_ps.dat srv_ss.dat
```

选项：

- `--cstype aero|mech`：输出坐标系。`mech`（结构系，位置 mm，**默认**）或 `aero`（气动系，m）。仅影响显示，内部始终在气动系(m)计算。
- `--pref P`：减去参考压 P [Pa]（换算到表压基准）。默认按绝对压力处理，不减。

输出：合力 `F=(Fx,Fy,Fz)`、`|F|`、压力中心点（位于合力作用线上、距本体质心最近）、作用线方向、剩余力偶。

## 验算（可选）

用某参考点处已知的外部合力与弯矩，验算程序结果。结果写入同目录 `verify_report.md`（终端只打印简短摘要）。

1. 把随附模板 `external_load.sample.txt` 复制为 `external_load.txt`，填入参考点、力、力矩（行格式，首行指定坐标系）：

```
<cs_type>       # 坐标系类型: aero(位置 m) 或 mech(位置 mm)
<x>             # 参考点 r_ref 的 x
<y>             # 参考点 r_ref 的 y
<z>             # 参考点 r_ref 的 z
<Fx>            # [N]
<Fy>            # [N]
<Fz>            # [N]
<mx>            # [N·m]
<my>            # [N·m]
<mz>            # [N·m]
```

9 个数值一行一个；可带 `#` 注释。位置单位随 `cs_type`：`mech` 为 mm，`aero` 为 m；力/力矩恒为 N、N·m。

2. 运行：

```
python verify.py --ps srv_ps.dat --ss srv_ss.dat --ext external_load.txt --tol 0.05
```

`--tol` 为相对容差（`0.05` = 5%）。报告含四项相对误差（合力、力矩、压力中心、剩余力偶，以**百分比**显示）与逐项 PASS/FAIL。

坐标系关系：`aero = (mech_z, -mech_y, mech_x)/1000`（x、z 互换，y 取反，mm→m）。外部 `mx,my,mz` 须为右手系 `M=∮(r-r_ref)×dF`。

## 输出含义

- **压力中心**：合力作用线上距本体质心最近的点（作用线上的点都满足“垂直于力的力矩为零”）。
- **剩余力偶**：沿合力方向、不可通过选点消除的纯扭矩（三维分布固有）。若其相对 `|F|×叶片尺寸` 很小，载荷可近似为一个单纯力；否则结构分析时需计入该扭矩。

## 文件

| 文件 | 用途 |
|---|---|
| `pressure_center.py` | 主程序（读取、三角化、积分、压力中心、命令行） |
| `verify.py` | 用外部载荷验算，输出 `verify_report.md` |
| `external_load.sample.txt` | 外部载荷模板（复制为 `external_load.txt` 并填入实际值） |
| `requirements.txt` | 依赖 |
| `verify_report.md` | `verify.py` 生成的验算报告（运行后出现） |

> 输入数据文件（`*.dat`，如 `srv_ps.dat`/`srv_ss.dat`）需自行提供，不在仓库中（已 gitignore，避免泄漏真实叶片数据）。

## 假设 / 局限

- 每个面片是其最佳拟合平面上的“图”（不折叠）；适用常规叶片吸力面/压力面。
- 两面片需分开（有限厚度）以定义外法向。
- 压力默认按绝对压力处理；薄叶片下常数项在 ps/ss 间近似抵消，合力基本正确，压力中心可能略受影响。需表压基准时用 `--pref`。

## License

**CC BY-NC 4.0**（Creative Commons Attribution-NonCommercial 4.0 International）— 非商用：可使用、分享、改编，但**仅供非商业用途**，且须署名并注明改动；商业用途需另行获得许可。见 [LICENSE](LICENSE) 与 https://creativecommons.org/licenses/by-nc/4.0/。

> 开发与测试说明见 `README_DEV.md`。
