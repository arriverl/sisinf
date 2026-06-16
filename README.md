# 2026 密码数学挑战赛赛题一：SIS∞ 解题工作区

**开源地址：** [https://github.com/arriverl/sisinf](https://github.com/arriverl/sisinf)

### 全算法验证（2026 接入）

| 类别 | 新增/启用模块 | 验证命令 |
|------|---------------|----------|
| 一 | `lattice_sieve.py`（BKZ+list sieve） | `python scripts/run_full_validation.py --problems 1,3,6,9` |
| 二 | `lattice_kannan.py`（Kannan 嵌入，需 fpylll） | `--problems 2,4,7,10` |
| 三 | `lattice_restricted_svp.py`（Wang enumerate-slice + d4f）+ `norm_sq < q²` | `--problems 5,8` |
| 全十题 | `run_full_validation.py` + 阶梯计分 `sis_scoring.py` | `python scripts/run_full_validation.py --quick` |

冒烟：`python scripts/smoke_algorithms.py`

**服务器部署（推荐 Linux + fpylll）：**

```bash
conda install -c conda-forge fpylll
pip install numpy ortools
bash scripts/run_server_validation.sh
```

报告输出：`results/full_validation/full_validation_report.json`（含阶梯计分）

本目录用于完成“无穷范数下的短整数解问题求解（SIS∞）”赛题，包含：
- 一个可批量求解的脚本框架；
- 标准化输入/输出格式，便于提交与复核。

---

## 1. 赛题要点与数学建模

目标是对给定 `A in Z_q^(n*m)`、`t in Z_q^n`，求向量 `u, v`，使得：

- `A v + u ≡ t (mod q)`
- `||u||_inf <= gamma, ||v||_inf <= gamma`

其中题目参数大致为 `n=m=q in {100,120,140,160}`，`gamma in {15,16,17,18}`。

### 1.1 与经典 SIS 的关系

- 齐次 SIS（`t=0`）可看作格上的短向量搜索（SVP 近似）；
- 非齐次 SIS（`t!=0`）可对应到嵌入后的近似最近向量问题（CVP 近似）；
- 本题核心难点是 **无穷范数约束**，这与大量以欧氏范数为主的 BKZ 思路存在目标失配。

### 1.1.1 赛题三类与小问编号

| 类别 | 小问 | 特征 | 求解侧重 |
|------|------|------|----------|
| 第一类 | 1、3、6、9 | 齐次 SIS（`t≡0`） | BKZ/LLL 短向量种子、模 `q` 核游走、禁止平凡零解 |
| 第二类 | 2、4、7、10 | 非齐次 SIS（`t≢0`） | CVP 提升种子、模拉回、弱化 BKZ |
| 第三类 | 5、8 | 在 L∞ 可行外另需 `‖u‖₂²+‖v‖₂² < q²` | 受限 SVP + 熵分散 + lex ILP（脚本对 5/8 自动开启） |

题号与类别映射见 `scripts/sis_problem_taxonomy.py`；批量求解第一类：

```bash
cd sisinf_challenge2026
python scripts/run_class1_until_success.py --json-dir saiti1/sis_inf_problems_json
```

### 1.2 关键等价变换（残差形式）

定义中心化剩余（逐坐标映射到 `[-q/2, q/2]`）：

`r(v) = Center( t - A v mod q )`

则只要 `|r_i(v)| <= gamma` 全部成立，就可取 `u = r(v)`，立即得到合法解。

因此问题可化为：

> 在盒约束 `v_j in [-gamma, gamma]` 下，寻找一个 `v`，使得 `r(v)` 全坐标落入 `[-gamma, gamma]`。

这个表述把“同时求 u,v”转化成“先找 v，再直接恢复 u”。

---

## 2. 前置知识（比赛报告可直接复用）

## 2.1 格与模线性方程

- 矩阵 `A`、向量 `u,v,t` 都在 `Z_q` 上计算；
- `A v + u ≡ t (mod q)` 是一个带盒约束的模线性系统；
- 模空间中的“接近 0”由中心化后绝对值衡量。

## 2.2 范数几何差异

- 欧氏短不等价于无穷短；
- 对高维平均格，欧氏最短向量再筛选到无穷短通常存在约 `sqrt(n)` 量级损失；
- 因而直接套 BKZ 往往不够，需要“面向无穷范数”的目标函数与搜索动作。

## 2.3 启发式求解常见机制

- 局部搜索：逐坐标试探小步长，快速下降违规度；
- 多重重启：避免陷在局部最优；
- 大扰动/温度策略：跨越高势垒区域；
- 增量更新：避免每次全量重算，提高维度 100-160 时效率。

---

## 3. 创新思路：Dual-Space + Entropy IIR-CLS

我们提出的框架为 **Dual-Space + Entropy IIR-CLS**，核心是把“格空间候选生成”与“残差空间精修”结合：

- 双空间混合搜索：先生成高质量 `v` 候选，再在残差空间做定向修复；
- 熵引导目标函数：在满足 `L_inf` 的同时，主动引导欧氏范数（用于第 5/8 题）；
- 动态权重：前期重可行性，后期重欧氏下限。

## 3.1 核心设计

1) **残差主导建模（Residual-First）**  
将 `u` 消去，直接优化 `r(v)`，避免双变量耦合搜索。

2) **分层目标函数（Lexicographic Objective）**  
按优先级最小化：
- 一级：违规坐标数量 `count(|r_i| > gamma)`；
- 二级：违规幅度和 `sum(max(0, |r_i|-gamma))`；
- 三级：二次平滑项（减少震荡）。

3) **坐标增量精算（Column-Wise Delta Evaluation）**  
修改单个 `v_j` 时，`r` 可通过列向量 `A[:,j]` 快速更新并中心化，避免重复矩阵乘法。

4) **双阶段搜索（Greedy + Kick）**  
- 阶段 A：贪心下降，快速压低违规度；
- 阶段 B：停滞时进行受控扰动（kick）并继续局部搜索。

5) **多重重启 + 早停证据**
多次随机初值试探；可选 `--parallel-workers` 并行重启；首个校验通过即停。

6) **熵引导目标函数（Entropy-Guided）**
- 在溢出目标基础上加入熵奖励项（鼓励分量分布更分散）；
- 对第 5/8 题引入 `euclid_gap = max(0, q^2 - (||u||_2^2 + ||v||_2^2))` 惩罚；
- 通过动态权重把搜索从“找可行”过渡到“提质量”。

7) **模拉回候选（Modular Pull Seeds）**  
对 `center(t)` 取多种形态 \(φ\)（原值、符号、饱和方向、随机方向），构造 \(v \propto -A^\top φ\) 并投影到 \([-γ,γ]^m\)，作为与残差几何一致的初始池补充。

8) **双坐标联合救援（Pair Relief）**  
周期性地对随机坐标对 \((j,k)\) 在小网格 \([-R,R]^2\) 上枚举联合增量，寻找严格 lex 改进；缓解单坐标贪心卡在耦合鞍附近的问题。

9) **残差符号梯度踢（Pull Kick）**  
停滞步使用 \(\Delta v \propto -A^\top \mathrm{sign}(u)\)（裁剪步长）做一次结构化扰动；若无改进再退回原有随机 kick。

## 3.2 论据与可解释性

- 与 BKZ+筛法再过滤相比，本法“目标一致”：直接优化 `L_inf`；
- 每一步动作都有可解释统计量（违规坐标、超界总量、最大违规）；
- 对 `n=100~160`、`gamma` 相对较小的盒约束，单坐标小步调整具备较高性价比；
- 工程上只依赖 `numpy`，便于复现实验和参数消融。

## 3.3 对第 5/8 题欧氏上界约束

题面 OCR 可能丢失平方上标。官方解析采用：

`‖u‖_2^2 + ‖v‖_2^2 < q^2`

若 `≥ q²` 则不得分。脚本对第三类强制 `require_norm_lt_q2`；搜索用稀疏化 + `euclid_excess` 惩罚控制 `L₂` 能量。

---

## 4. 文件组织与使用方法

## 4.0 Python 环境与 ortools / protobuf 冲突（重要）

在**用户级全局**执行 `pip install ortools` 时，会把 **protobuf 升到 6.33+**（ortools 要求），而本机若装有旧版 **opentelemetry-proto**（常见约束为 `protobuf<5`），pip 会提示**依赖冲突**：二者**无法在同一解释器环境**里长期共存。

**推荐做法**：在本目录使用**独立虚拟环境**，与全局包隔离：

```powershell
cd sisinf_challenge2026
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python scripts/solve_sisinf.py --input saiti1/instances_all.json --output results/solutions.json
```

依赖文件：

- `requirements.txt`：基础（`numpy`）
- `requirements-ortools.txt`：可选 CP-SAT（`ortools`）

若你已在全局强行安装 ortools 且其它工具报错，可在**另一终端**用 `pip install "protobuf<5,>=3.19"` 尝试恢复 opentelemetry 所需版本（**可能再次破坏 ortools**）；根本解决办法仍是：**赛题求解用 `.venv`，其它项目用各自 venv**。

## 4.1 目录

- `scripts/solve_sisinf.py`：主求解脚本；
- `scripts/lattice_bkz.py`：可选 BKZ 格种子（需安装 `fpylll`，见 `requirements.txt` 注释）；
- `data/`：输入题目数据（JSON）；
- `results/`：输出解与运行日志。

## 4.2 输入格式（建议）

JSON 顶层为列表，每题一个对象，例如：

```json
[
  {
    "id": 1,
    "n": 100,
    "m": 100,
    "q": 100,
    "gamma": 15,
    "require_norm_ge_q2": false,
    "A": [[...], [...], ...],
    "t": [0, 0, ...]
  }
]
```

其中 `A` 使用常规“按行给出”的二维数组（`A[row][col]`）。

## 4.3 运行

```bash
python scripts/solve_sisinf.py --input data/instances.json --output results/solutions.json
```

可调参数（示例）：

```bash
python scripts/solve_sisinf.py \
  --input data/instances.json \
  --output results/solutions.json \
  --restarts 40 \
  --iters 2500 \
  --delta 2 \
  --max-delta 6 \
  --candidate-count 24 \
  --entropy-weight 0.25 \
  --euclid-weight 1.5 \
  --seed 2026
```

新增“分阶段 + 收敛友好”参数（默认已启用单调接受）：

- `--residual-phase-end 0.45`：前期主攻 `u` 违规压缩；
- `--kernel-phase-start 0.60`：中后期强化 kernel walk / 周期 CP 修补；
- 默认 **不允许上坡 SA 接受**（更稳定，有限状态下更快停到局部最优）；如需更激进探索可显式加 `--allow-uphill-sa`。

可选 **BKZ 格种子**（`n+m` 不宜过大；默认 `--bkz-beta 0` 关闭）与 **并行重启**：

```bash
python scripts/solve_sisinf.py \
  --input data/instances.json \
  --output results/solutions.json \
  --bkz-beta 28 \
  --bkz-max-dim 120 \
  --parallel-workers 4
```

关闭 BKZ 种子：`--no-bkz-seeds`（仍保留启发式 dual-space）。

关闭双空间或动态权重（用于消融）：

```bash
python scripts/solve_sisinf.py \
  --input data/instances.json \
  --output results/solutions_baseline.json \
  --no-dual-space \
  --no-dynamic-schedule \
  --entropy-weight 0 \
  --euclid-weight 0
```

运行基线/消融对比：

```bash
python scripts/run_ablation.py \
  --input data/instances.json \
  --output-dir results/ablation
```

---

## 5. 结果复核指标（提交前自检）

每题检查：

- 同余约束：`A v + u == t (mod q)`；
- 无穷范数：`max(abs(u)) <= gamma` 且 `max(abs(v)) <= gamma`；
- 若要求：`‖u‖_2^2 + ‖v‖_2^2 < q^2`（第三类）；
- 齐次题非平凡性：`u,v` 不能同时全零；
- 输出运行时间与搜索统计。

---

## 6. 后续可强化方向

- 已完成：`objective_uv` 与校验对齐、按需熵、`--parallel-workers`、可选 BKZ 管线（`lattice_bkz.py`）；
- 可选：Kannan 嵌入上的非齐次 CVP 种子、SAT/ILP 收尾修复、更强筛法候选。
