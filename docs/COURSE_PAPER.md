<div align="center">

# 暨 南 大 学

## 本科生课程论文

<br><br>

### 论文题目：

# 2026 密码数学挑战赛赛题一：无穷范数短整数解（SIS∞）的格论启发式与整数规划混合求解研究

<br><br>

| | |
|:---|:---|
| 学　　院： | ________________________________ |
| 专　　业： | ________________________________ |
| 学生姓名： | ________________________________ |
| 学　　号： | ________________________________ |
| 课程名称： | ________________________________ |
| 指导教师： | ________________________________ |

<br>

______ 年 ______ 月 ______ 日

</div>

---

## 摘要

2026 年全国高校密码数学挑战赛赛题一要求在模 $q$ 下求解短整数解问题：给定 $A\in\mathbb{Z}_q^{n\times m}$、$t\in\mathbb{Z}_q^n$，求 $u,v\in\mathbb{Z}^n,\mathbb{Z}^m$ 使得 $Av+u\equiv t\pmod q$ 且 $\|u\|_\infty,\|v\|_\infty\le\gamma$。官方将十道小问划分为**近似最短向量（SVP）**、**近似最近向量（CVP）**与**受限最短向量（restricted SVP）**三类；第三类另要求 $\|u\|_2^2+\|v\|_2^2<q^2$，禁止“先求 $L_2$ 最短向量再过滤”的退化解法。

与以欧氏范数为目标的 BKZ 2.0 不同，赛题优化的是 Chebyshev（无穷）范数，二者在高维格上存在约 $\sqrt{n}$ 量级的传递损失。本文提出**残差主导三阶段混合框架**：按题号自动调度格种子生成（BKZ 2.0、G6K/BDGL 筛法、Kannan 嵌入、Wang enumerate-then-slice）→ 字典序局部搜索 → CP-SAT 整数规划收尾，并对接官方阶梯计分规则。主要创新包括：残差主导统一建模、Chebyshev 字典序目标 $(M,V,S)$、$O(n)$ 列增量残差更新、三类差异化流水线调度及欧氏约束下的 lex 收尾与抛光。实验表明，标准全量（batch${=}6$，ILP 1\,h）下已完成四题 $E_\infty=42$–$44$（阶梯计分均为 0），同余可满足；全维 CP-SAT 为主要压降手段。完整实现已开源：https://github.com/arriverl/sisinf。

**关键词：** 短整数解；无穷范数；格基约化；受限最短向量；约束规划；混合启发式

---

## 目录

| 章节 | 标题 | 页码 |
|------|------|------|
| — | 摘要 | 1 |
| 1 | 引言 | 3 |
| 2 | 原理或方案设计 | 5 |
| 3 | 程序实现或算法分析 | 15 |
| 4 | 总结 | 25 |
| — | 参考文献 | 27 |
| — | 附录 | 29 |

---

# 1. 引言

短整数解（Short Integer Solution, SIS）是格密码学的核心困难假设之一。Ajtai（1996）证明：在适当参数下，SIS 的平均情况困难性可归约到格上某些最坏情况问题，从而成为 Dilithium 等后量子签名方案的安全基础。

2026 年全国高校密码数学挑战赛赛题一在经典 SIS 上附加**无穷范数盒约束**，要求对给定参数

$$
n=m\in\{100,120,140,160\},\quad q\in\{100,120,140,160\},\quad \gamma\in\{15,16,17,18\},
$$

求解

$$
Av+u\equiv t\pmod q,\qquad \|u\|_\infty\le\gamma,\quad \|v\|_\infty\le\gamma. \tag{1}
$$

其中 $A\in\mathbb{Z}_q^{n\times m}$，$t\in\mathbb{Z}_q^n$。该参数区属于**非平凡短向量区**：$\gamma$ 远小于 $q/2$，解不能通过简单取零向量获得。

赛题难点在于三方面：

1. **范数目标失配**：格约化经典工具（LLL、BKZ 2.0、BDGL 筛法）以 $L_2$ 范数最短向量为目标，而赛题优化 $L_\infty$，二者不等价；
2. **三类异构约束**：齐次/非齐次、附加欧氏上界使十题需差异化算法路线；
3. **阶梯计分**：官方不以二元“可行/不可行”衡量，而按 $E_\infty=\max(\|u\|_\infty,\|v\|_\infty)$ 相对 $\gamma$ 的偏移给 0–10 分，要求算法持续压降 $L_\infty$ 能量。

本文在统一残差建模下，将 Chen–Nguyen BKZ 2.0、Becker BDGL 筛法、Kannan CVP 嵌入与 Wang 受限 SVP 工程化为可复现的开源求解器，形成“格种子 + 字典序局部搜索 + CP-SAT 收尾”的混合框架，并给出完整运行方法与实验分析。

---

# 2. 原理或方案设计

## 2.1 中心化与同余约定

对 $x\in\mathbb{Z}$，定义模 $q$ 对称中心化映射：

$$
\mathrm{center}(x)=\begin{cases}
x-q, & x>q/2,\\
x, & \text{否则},
\end{cases}
\qquad
\mathrm{center}(\mathbf{z})=\bigl(\mathrm{center}(z_i)\bigr)_i.
$$

所有坐标比较均在 $(-q/2,q/2]$ 上进行，避免模环绕导致的虚假溢出。

## 2.2 残差主导建模（核心等价变换）

定义残差函数

$$
u=r(v)=\mathrm{center}\bigl(t-Av\bmod q\bigr). \tag{2}
$$

则对任意 $v$，式 (2) 自动满足 $Av+u\equiv t\pmod q$。可行性等价于

$$
\|r(v)\|_\infty\le\gamma \quad\land\quad \|v\|_\infty\le\gamma. \tag{3}
$$

**意义**：将双变量 $(u,v)$ 搜索降为单变量 $v$ 搜索；$u$ 由式 (2) 唯一确定。三类题共享此建模，是本文方案的统一数学基础。

## 2.3 赛题三类划分

| 类别 | 小问编号 | $t$ 特征 | 官方推荐路线 | 附加约束 |
|:----:|:--------:|:--------:|:-------------|:---------|
| 一 | 1, 3, 6, 9 | $t\equiv 0$ | 近似 SVP：BKZ 2.0 + 筛法 | 非平凡：$u,v$ 不能全零 |
| 二 | 2, 4, 7, 10 | $t\not\equiv 0$ | 近似 CVP：Kannan 嵌入 | — |
| 三 | 5, 8 | 一般 | 受限 SVP（Wang et al.） | $\|u\|_2^2+\|v\|_2^2<q^2$ |

第三类禁止将“$L_2$ 最短向量 + 事后过滤”作为主路径：稠密解往往 $L_2$ 过大，无法通过欧氏上界。

## 2.4 阶梯计分函数

记 $E_\infty=\max(\|u\|_\infty,\|v\|_\infty)$。在前提满足时（同余成立；第三类另需 $\|u\|_2^2+\|v\|_2^2<q^2$），官方阶梯计分为：

$$
S(E_\infty)=
\begin{cases}
10, & E_\infty\le\gamma,\\
8,  & E_\infty=\gamma+1,\\
6,  & E_\infty=\gamma+2,\\
4,  & E_\infty=\gamma+3,\\
2,  & E_\infty=\gamma+4,\\
0,  & E_\infty>\gamma+4\ \text{或前提不满足}.
\end{cases}
\tag{4}
$$

以题 5（$\gamma=16$）为例：$E_\infty\le 16$ 满分，$E_\infty=20$ 得 2 分，$E_\infty\ge 21$ 得 0 分。

## 2.5 格论背景与文献路线

### 2.5.1 Ajtai 格与 BKZ 2.0

对第一类齐次题，可构造 $(n+m)$ 维 Ajtai 格基 $B$，其短向量 $\mathbf{z}=(u^\top,v^\top)^\top$ 对应候选解。BKZ 2.0（Chen & Nguyen, 2011）以块大小 $\beta$ 在 $L_2$ 意义下约化基，块越大短向量越好，但计算量指数增长。

### 2.5.2 BDGL 筛法与 G6K

Becker 等（2016）的 BDGL 筛法是当前实用近似 SVP 最强路线之一，与 BKZ 组合可显著降低 $L_2$ 根启发式。本文经 G6K 库接入 `bdgl2` 后端；无 G6K 时回退 list sieve。

### 2.5.3 Kannan 嵌入（第二类）

对非齐次题，Kannan（1987）将 CVP 嵌入到 $(n+m+1)$ 维 SVP：在增广格中搜索接近目标 $t$ 的短向量，再模拉回得到 $v$ 候选。

### 2.5.4 Wang 受限 SVP（第三类）

Wang 等（PQCrypto 2025, ePrint 2025/586）提出 **enumerate-then-slice**：

1. **enumerate**：在 BKZ 约化基的尾块子格上枚举，生成近似短向量**列表**（非单一 $L_2$ 最短）；
2. **slice**：按 $L_\infty$ 盒约束与 $\|u\|_2^2+\|v\|_2^2<q^2$ 筛选；
3. **dimension for free (d4f)**：在尾块降维枚举，降低有效搜索维数。

## 2.6 总体架构：三阶段混合流水线

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│ 题号/类别识别 │ → │ 格种子生成        │ → │ 字典序局部搜索    │ → │ CP-SAT 收尾   │
│ taxonomy    │    │ BKZ/G6K/Kannan/  │    │ 多重重启+邻域算子 │    │ full/lex/chunk│
│             │    │ Wang slice       │    │                  │    │              │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
```

调度入口：`apply_sis_class_defaults(cfg, sis_class)`；论文全量预设：`--full-max` → `apply_full_max_stack`。

| 阶段 | 第一类 (1,3,6,9) | 第二类 (2,4,7,10) | 第三类 (5,8) |
|------|------------------|-------------------|--------------|
| 格种子 | BKZ + G6K/sieve + Wang $L_\infty$ slice + Wagner | Kannan 嵌入 + CVP 提升 | G6K/BKZ 列表 + Wang slice |
| 搜索 | 核游走、$u$ 行 snap、违规 LS | pull kick、模拉回种子 | 稀疏化、$L_2$ 惩罚、熵引导 |
| 收尾 | full CP-SAT | full CP-SAT | lex CP-SAT + `euclid_polish` |

## 2.7 Chebyshev 字典序目标

对候选 $(u,v)$，定义溢出向量

$$
o^u_i=\max(|u_i|-\gamma,0),\quad o^v_j=\max(|v_j|-\gamma,0).
$$

字典序评分三元组（与 `verify_solution`、阶梯计分一致）：

$$
V=\#\{i: o^u_i>0\}+\#\{j: o^v_j>0\},\quad
S=\sum_i o^u_i+\sum_j o^v_j,\quad
M=\max\bigl(\max_i o^u_i,\max_j o^v_j\bigr). \tag{5}
$$

**字典序**：先最小化 $M$，再 $V$，再 $S$。记 `score_key = (M, V, S)`。

### 2.7.1 能量函数（搜索用连续代理）

局部搜索使用加权能量（`energy_from_parts`）：

$$
\mathcal{E}=10^6\cdot V + w_M\cdot M + w_S\cdot S + w_e\cdot e_{\mathrm{excess}} - w_H\cdot H,
\tag{6}
$$

其中：

- $e_{\mathrm{excess}}=\max(0,\|u\|_2^2+\|v\|_2^2-q^2+1)$（第三类启用）；
- $H$ 为 $|u|,|v|$ 的分箱熵（鼓励坐标分散，降低 $L_2$）；
- $w_M,w_S,w_e,w_H$ 由 `SearchConfig` 与搜索进度 `progress` 动态调度。

当 $M\ge 20$ 时，$w_M$ 自动放大（`cheby_boost_factor`），强化对最大溢出坐标的压制。

## 2.8 $O(n)$ 列增量残差更新

修改单坐标 $v_j\leftarrow v_j+\delta$ 时，不必重算 $Av$：

$$
r'(v)= \mathrm{center}\bigl(r(v)-\delta\cdot A_{\cdot j}\bigr). \tag{7}
$$

每次邻域试探仅需一次长度为 $n$ 的向量加减与中心化，复杂度 $O(n)$，在 $n,m\in[100,160]$ 下使百万级邻域试探成为可能。

## 2.9 三类差异化方案详述

### 2.9.1 第一类：齐次近似 SVP

**分解链**：

$$
\text{BKZ}(\beta\approx 28\text{–}56) \to \text{G6K bdgl2 / list sieve} \to \text{Wang }L_\infty\text{ slice} \to \text{Wagner 子集种子} \to \text{核游走} \to \text{full CP-SAT}.
$$

- **Wagner 种子**：对 $A$ 子块做广义生日式组合，生成结构化 $v$ 初值；
- **核游走**：在 $\ker(A\bmod q)$ 基下小系数组合，探索同余等价类；
- **$u$ 行 snap**：对溢出最大的若干行，在局部列窗口内精确修复。

### 2.9.2 第二类：非齐次近似 CVP

$$
\text{Kannan 嵌入格} \xrightarrow{\text{BKZ}} \text{短向量} \xrightarrow{\text{CVP 提升}} v\text{ 候选} \xrightarrow{\text{模拉回}} \text{局部搜索} \to \text{full CP-SAT}.
$$

默认**关闭 BKZ 主种子**（`use_bkz_seeds=False`），避免 $L_2$ 最优但远离 $t$ 的退化解；强化 `modular_pull` 与 `pull_kick`：

$$
\Delta v \propto -A^\top \mathrm{sign}(u),
$$

在停滞时沿残差符号梯度方向扰动。

### 2.9.3 第三类：受限 SVP

$$
\text{短向量列表} \xrightarrow{\text{enumerate}} \text{slice}(L_\infty, L_2<q^2) \xrightarrow{\text{lex CP-SAT}} \text{euclid\_polish}.
$$

- **slice 条件**：$\|u\|_\infty,\|v\|_\infty\le\gamma$ 且 $\|u\|_2^2+\|v\|_2^2<q^2$；
- **lex 收尾**：CP-SAT 先优化 $L_\infty$ 目标，再抛光欧氏能量；
- **熵权重**加大（`entropy_weight≈0.45`），前期引导稀疏/分散解。

## 2.10 CP-SAT 收尾建模

对 $v_j\in[-\gamma,\gamma]$，引入整数 slack $k_i$ 线性化模同余：

$$
(A v)_i + u_i - q\cdot k_i = t_i,\quad u_i\in[-\gamma,\gamma].
$$

三种模式：

| 模式 | 适用 | 说明 |
|------|------|------|
| `full` | 类一、二 | 全维 $m$ 个 $v_j$ 同时优化 |
| `chunk` | 降维试探 | 分块子问题；平台处效果有限 |
| `lex` | 类三 | 字典序：先 $L_\infty$ 再 $L_2$ |

## 2.11 本文创新点归纳

| 编号 | 创新点 | 技术内涵 |
|:----:|--------|----------|
| 1 | 残差主导统一建模 | 式 (2) 贯穿三类题，降维搜索 |
| 2 | Chebyshev 字典序 | 式 (5) 与阶梯计分式 (4) 对齐 |
| 3 | 三类自动调度 | `apply_sis_class_defaults` / `--full-max` |
| 4 | 格种子分层栈 | BKZ / G6K / sieve / Kannan / Wang 模块化 |
| 5 | $O(n)$ 列增量 | 式 (7) 支撑大规模邻域搜索 |
| 6 | 双约束 lex 收尾 | 第三类 $L_\infty$ + $L_2<q^2$ 联合优化 |
| 7 | 可行可得分解 | 文献最优 / 本文实现 / 实验可达 三层对照 |

---

# 3. 程序实现或算法分析

## 3.1 软件架构

```
sisinf_challenge2026/
├── scripts/
│   ├── solve_sisinf.py   # 统一算法库（分类/计分/格种子/启发式/搜索/CP-SAT）
│   └── sis_cli.py        # 命令行（十题验证 / batch / 自检）
├── data/                 # 题目 JSON
└── results/              # 输出与报告
```

**主调用链**：

```
sis_cli.py
  → load instance
  → apply_sis_class_defaults(SearchConfig, class)
  → local_search_one(A, t, q, gamma, cfg)    # 格种子 + 搜索
  → execute_finish(..., mode=full|lex)       # CP-SAT
  → verify_solution + score_from_verify        # 校验 + 计分
```

## 3.2 格种子模块实现要点

格种子逻辑集中在 `solve_sisinf.py` 内 `# ===== lattice_seeds =====` 分段。

### 3.2.1 BKZ 2.0（`solve_sisinf` 内）

- 构造 $(n+m)$ 维 Ajtai 基 $B$；
- fpylll 多轮 BKZ（`BKZ.reduction`），块大小 $\beta$；
- 从约化基列提取 $v$ 候选，经 $\mathrm{center}(t-Av)$ 得 $u$；
- 组合短向量：`bkz_combo_depth` 控制系数深度。

### 3.2.2 G6K BDGL2（`solve_sisinf` 内）

- 初始化 G6K `Siever` 对象；
- 投影到尾块子空间执行 `bdgl2` 筛法；
- 饱和检测与 BKZ 回退；
- `--full-max` 时作为第一类/第三类主筛法后端。

### 3.2.3 Wang 受限 SVP（`solve_sisinf` 内）

实现 Wang 论文三阶段：

```text
enumerate_approx_svp_list(R, tail_rank, coeff_max, pool_size)
  → slice_by_linf_and_norm(u, v, gamma, q)
  → wang_restricted_svp_v_seeds(...)
```

尾块秩 `wang_enum_tail_rank`、池大小 `wang_enum_pool_size`、最大试验 `wang_enum_max_trials` 控制算力。

## 3.3 局部搜索算法

**算法 1** 单 restart 字典序下降（概要）

```
输入: A, t, q, γ, 初值 v₀, 配置 cfg
1. r ← center(t - Av₀); 计算 score = (M,V,S)
2. for step = 1 .. cfg.iters:
3.   按 progress 调度阶段（残差期 / 核游走期）
4.   随机排列坐标 j ∈ {1..m}
5.   for each j:
6.     for δ ∈ [-Δ, Δ]:
7.       用式 (7) 增量计算 r'; 得 score'
8.       if score' 字典序优于 score: 接受移动
9.   if 停滞: pull_kick / 高斯扰动 / Wagner 修补
10.  if verify_ok: return 成功
输出: 最优 (u,v), score
```

**邻域算子一览**：

| 算子 | 触发条件 | 作用 |
|------|----------|------|
| 单坐标贪心 | 每步 | 式 (7) 增量下降 |
| pull_kick | 停滞 | $\Delta v\propto -A^\top\mathrm{sign}(u)$ |
| 核游走 | progress > 0.78 | $\ker(A)$ 小系数组合 |
| pair_relief | 周期 | 双坐标联合枚举 |
| $u$ 行 snap | 类一周期 | 局部行精确修复 |
| block CP | 周期 | 小块 CP-SAT 在线修补 |

## 3.4 复杂度与算力分析

| 组件 | 时间复杂度（概估） | 算力需求 |
|------|-------------------|----------|
| BKZ 2.0 | $O(n^3\cdot 2^{O(\beta)})$ 启发式 | fpylll；$\beta\le 56$ |
| G6K bdgl2 | 指数级（维数相关） | Cython + 多核；需 `install_g6k.sh` |
| Wang enumerate | $O(\binom{r}{k}(2c+1)^k)$ | 尾块秩 $r$，系数界 $c$ |
| 局部搜索 | $O(T\cdot m\cdot\Delta\cdot n)$ | $T$=迭代数；纯 numpy |
| full CP-SAT | 指数级（最坏） | OR-Tools；$m\le 160$ 可实用 |

**推荐硬件**：Linux 服务器，16+ 核 CPU，32GB+ RAM；`--full-max` 长跑建议 64GB。

## 3.5 环境安装

```bash
cd /home/cwh/sisinf          # 或本地 sisinf_challenge2026
conda activate cwh

# 格约化
conda install -c conda-forge fpylll -y

# Python 依赖
pip install numpy ortools "Cython>=3.0" cysignals

# G6K 真筛法（推荐）
bash scripts/install_g6k.sh

# 自检
python3 scripts/sis_cli.py check all
python3 scripts/sis_cli.py check g6k
```

G6K 安装失败时：

```bash
pip install "Cython>=3.0" cysignals numpy setuptools wheel
cd vendor/g6k && python3 setup.py build_ext --inplace
pip install --no-build-isolation -e .
```

## 3.6 运行命令（完整列举）

### 3.6.1 冒烟与自检

```bash
python3 scripts/sis_cli.py check smoke
python3 scripts/sis_cli.py check all
python3 scripts/sis_cli.py check g6k
```

### 3.6.2 快速十题筛查（约 1–3 小时）

```bash
python3 scripts/sis_cli.py --quick
```

### 3.6.3 标准全量验证（推荐）

```bash
python3 scripts/sis_cli.py \
  --batch-rounds 6 \
  --ilp-time-limit 3600 \
  --output-dir results/full_validation
```

### 3.6.4 论文全量方案（`--full-max`）

```bash
python3 scripts/sis_cli.py \
  --full-max \
  --batch-rounds 24 \
  --ilp-time-limit 14400 \
  --output-dir results/full_max_validation
```

### 3.6.5 分三类长跑

```bash
python3 scripts/sis_cli.py batch --class 1 --full-max --max-rounds 24
python3 scripts/sis_cli.py batch --class 2 --full-max --max-rounds 24
python3 scripts/sis_cli.py batch --class 3 --full-max --max-rounds 24
```

### 3.6.6 单题调试

```bash
python3 scripts/sis_cli.py --problems 1 --batch-rounds 6 --ilp-time-limit 3600
python3 scripts/sis_cli.py --problems 5,8 --full-max --batch-rounds 12
```

### 3.6.7 直接调用主求解器

```bash
python3 scripts/solve_sisinf.py \
  --input data/instances.json \
  --output results/solutions.json \
  --restarts 40 --iters 2500 --bkz-beta 28
```

**报告路径**：`results/full_validation/full_validation_report.json`（含每题 $E_\infty$、阶梯得分、运行时间）。

## 3.7 实验结果与分析

### 3.7.1 实验环境

- **系统**：Linux 服务器（conda 环境 `cwh`）
- **依赖**：fpylll、ortools（标准全量未启用 `--full-max`）
- **命令**：`python3 scripts/sis_cli.py --batch-rounds 6 --ilp-time-limit 3600`
- **配置**：每题 6 轮 batch 启发式 + 全维 CP-SAT 收尾，单题 ILP 时限 $T=3600$ s

### 3.7.2 十题全量验证结果

> 数据来源：`sis_cli.py` 标准全量跑批（2026-06-03）。p1–p4 已含 ILP 收尾；p5 为 batch 阶段快照；p6–p10 待跑完后补全。

| 题号 | 类 | $\gamma$ | $n{=}m$ | 启发式 $E_\infty$ | ILP 后 $E_\infty$ | $\|v\|_\infty$ | 同余 | norm_ok | 阶梯得分 | 状态 |
|:----:|:--:|:--------:|:-------:|:-----------------:|:-----------------:|:--------------:|:----:|:-------:|:--------:|:----:|
| 1 | 一 | 15 | 100 | 44 | **42** | 15 | ✓ | ✓ | 0 | 完成 |
| 2 | 二 | 15 | 100 | 42 | **42** | 15 | ✓ | ✓ | 0 | 完成 |
| 3 | 一 | 16 | 120 | 45 | **43** | 15 | ✓ | ✓ | 0 | 完成 |
| 4 | 二 | 16 | 120 | 45 | **44** | 15 | ✓ | ✓ | 0 | 完成 |
| 5 | 三 | 16 | 100 | 45 | — | 8–12 | ✓ | ✗ | — | batch 中 |
| 6 | 一 | 17 | 140 | — | — | — | — | — | — | 待完成 |
| 7 | 二 | 17 | 140 | — | — | — | — | — | — | 待完成 |
| 8 | 三 | 18 | 120 | — | — | — | — | — | — | 待完成 |
| 9 | 一 | 18 | 160 | — | — | — | — | — | — | 待完成 |
| 10 | 二 | 18 | 160 | — | — | — | — | — | — | 待完成 |

**距可行差距**（ILP 后 $E_\infty-\gamma$）：题 1、2 为 27；题 3 为 27；题 4 为 28。均远大于 $\gamma+4$，阶梯得分均为 0。

**ILP 降幅**（启发式 $\to$ ILP）：题 1：$44\to 42$（$-2$）；题 2：$42\to 42$（$0$）；题 3：$45\to 43$（$-2$）；题 4：$45\to 44$（$-1$）。

### 3.7.3 分类汇总（p1–p4 已完成）

**第一类（题 1、3）**

| 题号 | $q$ | 启发式 | ILP 后 | $\|u\|_\infty$ | $\|v\|_\infty$ | 得分 |
|:----:|:---:|:------:|:------:|:--------------:|:--------------:|:----:|
| 1 | 100 | 44 | **42** | 42 | 15 | 0 |
| 3 | 120 | 45 | **43** | 43 | 15 | 0 |

**第二类（题 2、4）**

| 题号 | $q$ | 启发式 | ILP 后 | $\|u\|_\infty$ | $\|v\|_\infty$ | 得分 |
|:----:|:---:|:------:|:------:|:--------------:|:--------------:|:----:|
| 2 | 100 | 42 | **42** | 42 | 15 | 0 |
| 4 | 120 | 45 | **44** | 44 | 15 | 0 |

**第三类（题 5，进行中）**

| 轮次 | $\|u\|_\infty$ | $\|v\|_\infty$ | norm_sq | $q^2$ | 说明 |
|:----:|:--------------:|:--------------:|:-------:|:-----:|:-----|
| 0–4（batch） | 45–46 | 8–12 | 72401–85363 | 10000 | $L_\infty$、$L_2$ 均未达标 |

### 3.7.4 结果分析

1. **同余与非平凡性**：p1–p4 均满足（`congr=1`）。
2. **瓶颈在 $\|u\|_\infty$**：$\|v\|_\infty$ 多题已贴界 $\gamma$ 或 $\gamma-1$，ILP 主要压缩 $u$ 侧。
3. **ILP 平台**：全维 CP-SAT 1 h 内可降 0–2 点，题 1、3 降幅最大；题 2 batch 已达 42 后 ILP 无进一步改善。
4. **维数效应**：$n=120$ 的题 3、4 ILP 后 $E_\infty$ 比 $n=100$ 的题 1、2 高约 1–2。
5. **第三类**：题 5 batch 阶段 norm_sq $\approx 7.3\times 10^4 \gg q^2=10^4$，需等待 lex CP-SAT + `euclid_polish` 收尾。
6. **阶梯计分**：已完成四题得分均为 0（$E_\infty \ge 42 \gg \gamma+4$）。

### 3.7.5 可行可得分解

| 类 | 文献最优 | 本文实现 | 当前实验可达（标准全量） |
|:--:|:---------|:---------|:-------------------------|
| 一 | BKZ + BDGL | BKZ + sieve + Wang + CP-SAT | 题 1、3：$E_\infty=42$–$43$ |
| 二 | Kannan + BKZ | Kannan + CVP + CP-SAT | 题 2、4：$E_\infty=42$–$44$ |
| 三 | Wang 受限 SVP | Wang slice + lex + polish | 题 5 batch：$E_\infty=45$，norm 未达标 |

### 3.7.6 消融观察

| 关闭模块 | 观察 |
|----------|------|
| 无 BKZ 种子 | 初值质量下降，收敛变慢 |
| 无 CP-SAT | $E_\infty$ 停在 44–45，无法进 42 档 |
| 无 G6K（仅 list sieve） | 种子多样性略降，影响有限 |
| `cheby_weight` 降低 | 最大溢出坐标压制不足 |

---

# 4. 总结

本文针对 2026 密码数学挑战赛赛题一（SIS∞），在残差主导建模式 (2) 基础上，提出格种子、字典序局部搜索与 CP-SAT 收尾相结合的三阶段混合框架。主要贡献包括：

1. 将十题三类异构约束统一于残差函数 $r(v)$，避免 $u,v$ 双变量耦合；
2. 设计 Chebyshev 字典序目标 (5) 与阶梯计分 (4) 对齐的搜索与验证体系；
3. 工程化实现 BKZ、G6K/BDGL、Kannan、Wang enumerate-slice 的分层格种子栈；
4. 通过 $O(n)$ 列增量更新 (7) 支撑百维规模下的百万级邻域搜索；
5. 开源可复现求解器（https://github.com/arriverl/sisinf），提供从冒烟到 `--full-max` 的完整评测流水线。

实验验证：标准全量（batch${=}6$，ILP 1\,h）下，已完成四题 $E_\infty=42$–$44$，全维 CP-SAT 为主要压降手段；十题全量报告跑完后将补全表~\ref{tab:full10}。进入得分档仍需更强 BDGL 或更大 Wang 搜索。

**展望**：（1）完成十题 `--full-max` 全量报告并更新实验表；（2）G6K 高维投影筛法参数调优；（3）第三类 lex 收尾中欧氏硬约束；（4）与 fpylll-BKZ 基线的效率—质量权衡分析。

---

# 参考文献

1. M. Ajtai. Generating hard instances of lattice problems. *STOC*, 1996.
2. A. K. Lenstra, H. W. Lenstra Jr., L. Lovász. Factoring polynomials with rational coefficients. *Math. Ann.*, 261:515–534, 1982.
3. D. Micciancio, O. Regev. Lattice-based cryptography. In *Post-Quantum Cryptography*, Springer, 2009.
4. Y. Chen, P. Q. Nguyen. BKZ 2.0: Better lattice security estimates. *ASIACRYPT*, 2011.
5. A. Becker, L. Ducas, N. Gama, M. Tibouchi. New directions in nearest neighbor searching with applications to lattice sieving. *SODA*, 2016.
6. D. Wagner. A generalized birthday problem. *CRYPTO*, 2002.
7. R. Kannan. Minkowski's convex body theorem and integer programming. *Math. Oper. Res.*, 12(3):415–440, 1987.
8. L. Babai. On Lovász's lattice reduction. *Combinatorica*, 6(1):1–13, 1986.
9. T. Achterberg. Constraint Integer Programming. PhD thesis, TU Berlin, 2007.
10. Google OR-Tools. CP-SAT documentation, 2024.
11. The fpylll team. fpylll, 2019.
12. G. Wang, W. Xia, D. Gu. Heuristic algorithm for restricted SVP. *PQCrypto*, 2025. https://eprint.iacr.org/2025/586
13. 2026 全国高校密码数学挑战赛. 赛题一解析与评分规则, 2026.
14. arriverl. *sisinf*. GitHub, 2026. https://github.com/arriverl/sisinf
15. H. H. Hoos, T. Stützle. *Stochastic Local Search*. Morgan Kaufmann, 2004.

---

# 附录

## 附录 A  十题参数一览

| 题号 | 类别 | $n=m$ | $q$ | $\gamma$ | $t\equiv 0$ | 欧氏约束 |
|:----:|:----:|:-----:|:---:|:--------:|:-----------:|:--------:|
| 1 | 一 | 100 | 100 | 15 | 是 | 否 |
| 2 | 二 | 100 | 100 | 15 | 否 | 否 |
| 3 | 一 | 120 | 120 | 16 | 是 | 否 |
| 4 | 二 | 120 | 120 | 16 | 否 | 否 |
| 5 | 三 | 100 | 100 | 16 | 否 | $\|u\|_2^2+\|v\|_2^2<q^2$ |
| 6 | 一 | 140 | 140 | 17 | 是 | 否 |
| 7 | 二 | 140 | 140 | 17 | 否 | 否 |
| 8 | 三 | 120 | 120 | 18 | 否 | $\|u\|_2^2+\|v\|_2^2<q^2$ |
| 9 | 一 | 160 | 160 | 18 | 是 | 否 |
| 10 | 二 | 160 | 160 | 18 | 否 | 否 |

## 附录 B  `SearchConfig` 三类默认差异（摘要）

| 参数 | 类一 | 类二 | 类三 |
|------|:----:|:----:|:----:|
| `use_bkz_seeds` | ✓ | ✗ | ✓ |
| `use_kannan_seeds` | ✗ | ✓ | ✗ |
| `use_restricted_svp_seeds` | ✓ | ✗ | ✓ |
| `use_sieve_seeds` | ✓ | ✗ | ✓ |
| `bkz_beta` | 28–32 | 24–28 | 24–28 |
| `euclid_weight` | 0.5 | 0.6 | 3.0–4.0 |
| `entropy_weight` | 0 | ≤0.2 | 0.45 |
| CP-SAT 模式 | full | full | lex |

## 附录 C  阶梯计分伪代码

```python
def competition_score(gamma, inf_u, inf_v, congruence_ok, norm_sq, q, sis_class):
    if not congruence_ok:
        return 0
    if sis_class == 3 and norm_sq >= q * q:
        return 0
    e_inf = max(inf_u, inf_v)
    if e_inf <= gamma:      return 10
    if e_inf == gamma + 1:  return 8
    if e_inf == gamma + 2:  return 6
    if e_inf == gamma + 3:  return 4
    if e_inf == gamma + 4:  return 2
    return 0
```

## 附录 D  LaTeX 编译说明

```bash
cd docs
xelatex COURSE_PAPER.tex
xelatex COURSE_PAPER.tex
```

字体：正文小四号（12pt）；参考文献与附录五号（10.5pt）。封面学号、姓名等请在 `.tex` 封面页填写。

---

*正式排版版本：`docs/COURSE_PAPER.tex`（XeLaTeX + ctexart）*
