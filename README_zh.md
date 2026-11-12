# SE-Modern

[English](README.md) | [简体中文](README_zh.md)

---

[![版本](https://img.shields.io/badge/version-0.1.0-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![许可](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![测试](https://img.shields.io/badge/tests-132%20passing-brightgreen.svg)]()

!!! warning "AI 生成内容声明"
    整个项目由 AI 生成，仅供参考。其中的内容、代码示例与文档均在 AI 辅助下创建。尽管我们已尽力保证准确性，但在用于生产环境之前，请自行核实全部信息并测试代码。使用风险自负。

> 一个 Python 研究原型，重建了 Surface Evolver 的数值内核——基于自动微分的能量梯度、L-BFGS 与信赖域 Newton-CG、增广拉格朗日约束、带不变式检查的拓扑手术、相场与无网格基线——并把每一项替换都对照只读的 Surface Evolver 基准进行验证。

## 目录

- [简介](#introduction)
- [基准结果](#benchmark-results)
- [前置条件](#prerequisites)
- [安装](#installation)
- [用法](#usage)
- [项目结构](#project-structure)
- [构建与测试](#build--test)
- [贡献](#contributing)
- [许可](#license)
- [致谢](#acknowledgments)

## 简介

Surface Evolver（SE）把皂膜表示为具有任意拓扑的显式三角曲面，然后在体积、接触角、周期与各向异性约束下最小化面积，从而求解泡沫、毛细面、极小曲面与润湿问题。那套*数学表述*对这些任务依然是最合适的；过时的是*数值实现*：手写梯度与 Hessian、对精确 Hessian 做直接分解、启发式拓扑手术，以及缺少自动微分、代数多重网格与 GPU 支持。

本项目保留该表述而替换数值实现，形式为一个独立的 Python 原型：

| 方面 | Surface Evolver | 本原型 |
|---|---|---|
| 导数 | 手写的梯度/Hessian | `torch.autograd` —— 提供 `grad` 与 Hessian 向量积，不组装 Hessian |
| 极小化器 | 梯度下降 + 精确 Hessian | L-BFGS（默认）与信赖域 Newton-CG（高精度） |
| 体积约束 | 拉格朗日乘子 / 投影 | 增广拉格朗日，并将径向投影作为回退方案 |
| 接触角 | 壁面 facet 记账 | 带解析梯度的线积分润湿能量 |
| 拓扑变更 | 启发式 `refine`/`swap`/`collapse`/`pop` | 相同操作，但带有前置条件与操作后不变式 |
| 扩散替代方案 | — | Allen-Cahn 相场（无需网格手术即可处理拓扑事件） |
| 光滑一维问题 | — | RBF-FD 无网格求解器 |

本原型**不**修改 SE 的 C 源码、不构建 GUI，也不宣称具备生产级性能。Surface Evolver 严格作为基准使用：它在临时目录中运行，其数据文件与 dump 均以只读方式解析。

## 基准结果

每一项替换都对照 SE 或解析解进行测量。下文所有数值均由已提交的脚本生成，并记录在 [`se-modern/reports/REPORT.md`](se-modern/reports/REPORT.md) 与 [`se-modern/reports/results.json`](se-modern/reports/results.json) 中。

| 分支 | 替换的内容 | 门禁 | 实测结果 |
|---|---|---|---|
| Phase 0 | 网格、`.fe`/`.dmp` 读写、能量 | 面积/体积与 SE 的偏差在 `1e-10` 内 | 相对误差 **2e-16** |
| B1 | 优化器与极小曲面 | 在加密网格上与解析面积对比 | 悬链面从上方二阶收敛 `+0.180% → +0.045% → +0.011%`（与独立 SciPy 解一致到 `1e-12`）；cube→sphere 在 1536 面下距 `4πr²` 仅 **0.084%**，与收敛后的 SE 相对差 **5e-4%** |
| B2 | 体积 / 接触角 / 周期性 / 各向异性 | 体积 `< 1e-8`，角度 `< 0.5°` | 体积误差 **4e-15**；接触角在 **0.245°** 以内 |
| B3 | 拓扑手术 | 100 次随机操作中 0 次不变式失败 | **0 次失败** |
| B4 | 相场（Allen-Cahn） | 半径误差 `< 5%`，合并为 1 个连通分量 | N=32 时 **1.19%**，N=48 时 **0.10%**；连通分量 2 → 1 |
| B5 | 无网格 RBF-FD | 剖面误差 `< 1%`，残差 `< 1e-6` | 剖面误差 **1.22e-4**，残差 **8.8e-11** |

上表中悬链面与 cube→sphere 两项的评测方法值得说明：二者都在**加密网格上以解析面积为准绳**评分，而非在原始数据文件网格上比较。`cube.fe` 的 24 个三角形上离散误差高达 5.72%——两个实现即便一致到 `1e-9`，也都离真值 5.72%，故粗网格互比无法区分实现差异。基准脚本因此先加密（球面 384 与 1536 面；悬链面 9/17/33 环），让双方都收敛，再报告误差如何随加密下降。实测收敛数据见 `se-modern/reports/convergence_cube.json` 与报告的 B1 章节。

本研究的结论——SE 的哪些部分应当保留、替换或混合使用——汇总在[报告的建议章节](se-modern/reports/REPORT.md)中。简言之：保留显式曲面表述（精度、可控拓扑、可与 SE 直接对比），替换数值内核，并在拓扑主导的阶段使用相场模型、在光滑轴对称子问题上使用无网格求解器。

## 前置条件

| 依赖 | 使用的版本 | 是否必需 | 用途 |
|---|---|---|---|
| Python | 3.11.13 | 是 | 运行时（假定 3.11+；`dict`/`tuple` 语法兼容 3.9，但低于 3.11 未测试） |
| `torch` | 2.7.1 | 是 | 自动微分、梯度、HVP、张量 |
| `numpy` | 2.2.6 | 是 | RBF-FD 模板、marching cubes 输入、线性代数 |
| `scipy` | 1.16.3 | 是 | 非线性求根、连通分量标记 |
| `scikit-image` | 0.26.0 | 是 | 相场输出的 marching-cubes 等值面 |
| `pytest` | 9.1.1 | 是（测试） | 测试运行器 |
| Surface Evolver | 2.70 | 可选 | 仅用于基准对比；缺失时依赖 SE 的测试会被跳过 |

默认在 CPU 上以 `torch.float64` 进行数值计算。

SE 按以下顺序查找，命中即止：`PATH` 中的 `evolver64`，然后是 `PATH` 中的 `evolver`，然后是 `E:\Mybat\evolver64.exe`，最后是 `E:\Mybat\evolver.exe`。若要覆盖该顺序，把可执行文件放入 `PATH` 即可。当找不到任何可执行文件时，需要 SE 的测试与基准会通过 `pytest.mark.skipif` 跳过而不是失败——其余部分仍照常运行。

## 安装

本项目没有打包清单，因此直接就地使用。克隆之后，让 `src/` 可被导入：

```bash
git clone <repository-url> evolver_alternative
cd evolver_alternative
```

安装 Python 依赖（上述版本即为门禁验证时所用的版本）：

```bash
python -m pip install torch numpy scipy scikit-image pytest
```

本 README 中的每条命令都在 `se-modern/` 下运行，并把 `src/` 加入导入路径：

```bash
cd se-modern
export PYTHONPATH=src          # Windows cmd 下：set PYTHONPATH=src
```

可选，用于 SE 基准：把 `evolver64`/`evolver` 放入 `PATH`，或安装到 `E:\Mybat\`（默认的回退位置）。

六个 SE 基准数据文件（`cube`、`cat`、`sphere`、`mound`、`phelanc`、`octa`）已随仓库提供在 `se-modern/data/se/` 中，因此克隆仓库后即可运行全部门禁。

## 用法

### 命令行

CLI 每次运行一个基准用例，并以该用例的通过/失败状态作为退出码。

```bash
# 列出可用用例
python -m semodern.cli list

# 运行单个用例（cube 是所有分支共用的合并门禁）
python -m semodern.cli run --case cube

# 其他用例：catenoid、cube_sphere、sessile、periodic、aniso、octa、phasefield、meshfree
python -m semodern.cli run --case catenoid
```

### 优化器

两个极小化器都接受任何暴露 `value(x)`、`grad(x)` 与 `hvp(x, v)` 的对象——即 `semodern.energy.Energy` 接口——起始向量为形状 `(3n,)` 的展平 float64 张量。

仅有面积时问题无下界，因此在闭合网格上做无约束运行会把网格塌缩到零能量。下面的示例最小化面积加上一个固定罚参数的体积项，该问题适定，也正是两个求解器的对比基准。

```python
import sys
sys.path.insert(0, "src")

from semodern import constraints, energy, io_fe, optimize

mesh = io_fe.load_fe("data/se/cube.fe")
base = energy.Energy(mesh.faces)
volume = constraints.VolumeConstraint(mesh.faces, volume0=1.0)
problem = constraints.AugmentedEnergy(base, volume, lam=0.0, rho=10.0)

for name, solver in (("lbfgs", optimize.lbfgs), ("tr_newton_cg", optimize.tr_newton_cg)):
    result = solver(problem, mesh.verts.reshape(-1), max_iter=200, tol=1e-8)
    print(name, result.energy, result.grad_norm, result.iterations, result.converged)
```

在随仓库提供的 `cube.fe` 上的输出——两个求解器达到相同能量，而 Newton-CG 所需的迭代次数少得多，因为它通过 `hvp` 使用了曲率信息：

```text
lbfgs: energy=4.436810 gn=3.316e-09 iters=32 conv=True vol=0.594701337
tr_newton_cg: energy=4.436810 gn=1.360e-11 iters=6 conv=True vol=0.594701334
```

### 体积约束极小化

增广拉格朗日*强制满足* `V(x) = V0`：每次外层迭代先最小化增广目标，然后更新乘子与罚参数。当体积必须被满足而不只是被“劝阻”时，请使用它取代上面的固定罚参数形式。

```python
import sys
sys.path.insert(0, "src")

from semodern import constraints, energy, io_fe

mesh = io_fe.load_fe("data/se/cube.fe")
base = energy.Energy(mesh.faces)
volume = constraints.VolumeConstraint(mesh.faces, volume0=1.0)

outcome = constraints.augmented_lagrangian(
    base, volume, mesh.verts.reshape(-1), outer=8, inner_iter=200
)
print(outcome["volume"], outcome["relative_residual"], outcome["converged"])
```

它会打印体积 `1.000000000181`、相对残差 `1.8e-10` 以及 `converged=True`。

### 拓扑操作

每个操作都会在改动*之前*校验前置条件，并在之后校验整个网格的不变式，因此它要么返回一个合法网格，要么抛出 `ValueError`，且输入保持原样。

```python
import sys
sys.path.insert(0, "src")

from semodern import mesh as mesh_mod, topology

refined, change = topology.refine(topology.tetrahedral_fixture())
print(change.before, "->", change.after, "delta", change.delta)
print("Euler characteristic:", mesh_mod.euler_characteristic(refined))
```

### 相场（扩散界面）

```python
import sys
sys.path.insert(0, "src")

from semodern import phasefield as pf

grid = pf.PhaseFieldGrid.from_signed_distance(
    32, 2.0, 2.0 * (2.0 / 32), pf.sphere_distance((0.0, 0.0, 0.0), 0.3)
)
result = grid.run(120, pf.stable_timestep(grid.eps, grid.dx))
print(result["energy"], result["monotone"], grid.equivalent_radius(), grid.component_count())
```

### 无网格 RBF-FD

```python
import sys
sys.path.insert(0, "src")

from semodern import meshfree

result = meshfree.solve_catenoid(half_height=1.0, radius=1.6, node_count=40, stencil=7)
print(result["relative_error_max"], result["residual_inf"], result["neck_numerical"])
```

### SE 桥接层

```python
import sys
sys.path.insert(0, "src")

from semodern import io_fe, se_bridge

run = se_bridge.run_se_timed("data/se/cube.fe", "g 20; hessian; g 20", "tmp/demo", name="out.dmp")
print(run.area, run.volume, run.energy)      # 从 SE 自身的 printf 输出中抓取
evolved = io_fe.load_dmp(run.dump_path)      # 解析生成的 dump
print(evolved.n_verts, evolved.n_faces)
```

## 项目结构

```
evolver_alternative/
├── LICENSE                      # GPL-3.0
└── se-modern/
    ├── src/semodern/
    │   ├── __init__.py          # 延迟解析子模块
    │   ├── mesh.py              # Mesh、边表、不变式、面积/体积
    │   ├── io_fe.py             # .fe/.dmp 解析器与写出器
    │   ├── energy.py            # 可微能量、Energy（value/grad/hvp）
    │   ├── optimize.py          # L-BFGS、信赖域 Newton-CG、Steihaug-CG
    │   ├── constraints.py       # 体积（增广拉格朗日）、接触角、周期性、各向异性
    │   ├── topology.py          # 带不变式的 refine/swap/collapse/pop
    │   ├── phasefield.py        # Allen-Cahn 扩散界面基线
    │   ├── meshfree.py          # RBF-FD 悬链面求解器
    │   ├── se_bridge.py         # 只读的 Surface Evolver 运行器
    │   ├── metrics.py           # 指标、对比、JSON 持久化
    │   └── cli.py               # `python -m semodern.cli run --case ...`
    ├── benchmarks/
    │   ├── smoke_cube.py        # 每个分支都运行的 7 步合并门禁
    │   ├── catenoid.py          # B1：面积最小化对照 SE
    │   ├── cube_sphere.py       # B1：体积约束下的立方体 → 球
    │   ├── sessile_cap.py       # B2：接触角往返验证
    │   ├── periodic_graph.py    # B2：周期高度场 → 平面
    │   ├── anisotropic.py       # B2：立方各向异性松弛
    │   ├── octa_topology.py     # B3：在 octa.fe 上的拓扑手术
    │   ├── phasefield_sphere.py # B4：半径误差与合并
    │   ├── meshfree_catenoid.py # B5：RBF-FD 对照精确剖面
    │   ├── run_all.py           # 运行全部基准、合并报告
    │   └── make_report.py       # 从 results.json 生成 reports/REPORT.md
    ├── tests/                   # 132 个测试
    ├── data/se/                 # 六个随仓库提供的 SE 数据文件
    └── reports/                 # results.json、results.b*.json、REPORT.md
```

## 构建与测试

本项目没有编译步骤。在 `se-modern/` 下运行测试套件：

```bash
export PYTHONPATH=src          # Windows cmd 下：set PYTHONPATH=src

# 完整测试套件（132 个测试）
python -m pytest -q

# 只跑某一领域
python -m pytest -q tests/test_optimize.py
python -m pytest -q tests/test_phasefield.py
```

`cube.fe` 冒烟测试是所有分支共用的合并门禁：它 dump 初始网格，检查 14 个顶点与 24 个面，把 Python 计算的面积/体积与 SE 报告的值对比，推进两个 SE 步，并断言面积没有增加。任何一项不满足时，它以非零码退出并打印 `SMOKE FAIL: <reason>`。

```bash
python benchmarks/smoke_cube.py    # 打印 "SMOKE PASS"
```

完整基准扫描会运行每个分支的脚本，把所有 `reports/results.*.json` 合并为 `reports/results.json`，并在任何门禁失败时以非零码退出：

```bash
python benchmarks/run_all.py

# 子集选择
python benchmarks/run_all.py --only b1
python benchmarks/run_all.py --skip b4,b5
```

扫描结束后重新生成报告：

```bash
python benchmarks/make_report.py   # 写出 reports/REPORT.md
```

## 贡献

1. 从 `main` 分支切出 `feat/<area>`（例如 `feat/b2-constraints`）。
2. 做外科式的最小改动：保持上文记录的公开签名稳定，因为后续阶段依赖它们（`Energy.value/grad/hvp`、`lbfgs`/`tr_newton_cg`、`OptResult`）。
3. 合并之前，以下各项必须全部通过：
   ```bash
   python -m pytest -q
   python benchmarks/smoke_cube.py
   python benchmarks/<your-area>.py
   git status --short      # 必须只显示你自己的文件
   ```
4. 使用 `git merge --no-ff <branch>` 合并，以保留分支历史。
5. 若某项数值改动改变了门禁结果，请更新 `reports/results.<branch>.json` 并重新运行 `benchmarks/make_report.py`——报告由测量结果生成，绝不手工编辑。

代码遵循以下约定：标识符、字符串、日志与错误消息使用英文；注释解释意图与不变式，而非机制。

## 许可

[GNU General Public License v3.0](LICENSE) —— 完整文本见 `LICENSE` 文件。

## 致谢

- **Ken Brakke**，[Surface Evolver](https://facstaff.susqu.edu/brakke/evolver/evolver.html) 的作者。本项目将其作为只读的参考实现，它定义了本项目所解析的 `.fe` 与 `.dmp` 格式。
- Surface Evolver 自带的测试数据文件（`cube.fe`、`cat.fe`、`sphere.fe`、`mound.fe`、`phelanc.fe`、`octa.fe`）未经修改地重新分发在 `se-modern/data/se/` 下，用于基准测试。
