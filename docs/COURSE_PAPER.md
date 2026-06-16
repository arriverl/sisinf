# 暨南大学本科生课程论文（Markdown 版）

> 正式排版请编译 `docs/COURSE_PAPER.tex`（XeLaTeX）。  
> 正文侧重理论与算法。  
> **开源仓库：** [https://github.com/arriverl/sisinf](https://github.com/arriverl/sisinf)

---

**论文题目：** 无穷范数短整数解（SIS∞）的格论基础、启发式搜索与整数规划收尾方法研究

| 项目 | 填写 |
|------|------|
| 学    院 |  |
| 专    业 |  |
| 学生姓名 |  |
| 学    号 |  |
| 课程名称 |  |
| 指导教师 |  |
| 日    期 |    年   月   日 |

---

## 摘要

赛题按官方解析分为**近似 SVP**（第一类）、**近似 CVP**（第二类）、**受限 SVP**（第三类）。与 BKZ 2.0 (Chen & Nguyen, 2011) 的 $L_2$ 目标不同，赛题优化 $L_\infty$；第三类另要求 $\|u\|_2^2+\|v\|_2^2<q^2$，且禁止“$L_2$ 短向量后过滤”(Wang et al., PQCrypto 2025)。

本文**实现**三类差异化种子模块（`lattice_sieve` / `lattice_kannan` / `lattice_restricted_svp`），配合字典序局部搜索与 CP-SAT 收尾，并集成 `run_full_validation.py` 十题验证与阶梯计分。第一类实验可达 $E_\infty=42$–45（$\gamma=15$），当前为 0 分档；第二、三类待服务器全量评测。开源：[https://github.com/arriverl/sisinf](https://github.com/arriverl/sisinf)。

**关键词：** 短整数解；无穷范数；格基约化；最近向量问题；局部搜索；约束规划

---

## 1. 引言

### 1.1 背景

Ajtai (1996) 证明 SIS 平均情况困难性可归约到格最坏情况问题，支撑后量子密码 (Regev, 2009; GPV, 2008)。赛题在式 $Av+u\equiv t\pmod q$ 上施加 $\|u\|_\infty,\|v\|_\infty\le\gamma$，参数 $n=m=q\in\{100,120,140,160\}$，$\gamma\in\{15,16,17,18\}$。

### 1.2 三类小问

| 类 | 题号 | 官方路线 | 本文实现 |
|----|------|----------|----------|
| 一 | 1,3,6,9 | BKZ 2.0 + 筛法 | `lattice_bkz` + `lattice_sieve` + Wang slice + full CP-SAT |
| 二 | 2,4,7,10 | Kannan 嵌入 / CVP | `lattice_kannan` + CVP 提升 + full CP-SAT |
| 三 | 5,8 | 受限 SVP (Wang 2025) | `wang_restricted_svp_v_seeds` + lex CP-SAT + 欧氏抛光 |

**计分（例：题 5，$\gamma=16$）：** 记 $E_\infty=\max(\|u\|_\infty,\|v\|_\infty)$，须同余且 $\|u\|_2^2+\|v\|_2^2<q^2$。$E_\infty\le\gamma$ 得 10 分；$=\gamma+1$ 得 8 分；…；$>\gamma+4$ 得 0 分。

**可行可得分解（$\gamma=15$）：**

| 类 | 文献最优 | 本文实现 | 当前可达 | 得分 |
|----|----------|----------|----------|------|
| 一 | BKZ+筛法 | BKZ + list sieve + Wagner | $E_\infty=42$–45 | 0 分 |
| 二 | Kannan+BKZ | Kannan 嵌入 + CVP 种子 | 待服务器评测 | — |
| 三 | 受限 SVP | 盒内稀疏采样 + lex | 待服务器评测 | — |

### 1.3 可解性直观

- **易解**：$\gamma\ge q/2$ 时中心化自动满足 $L_\infty$；$n,m$ 很小（$<40$）时全维 BKZ/ILP 可精确解。
- **难解**：本题 $\gamma\ll q$、百级维度、随机 $A$——属于文献中的标准困难区间 (Micciancio & Peikert, 2012)。
- **失配**：BKZ 优化 $L_2$，$L_\infty$ 短向量可差 $\Omega(\sqrt{n})$ 量级 (Banaszczyk, 1993)。

### 1.4 本文工作

残差双空间框架、**三类已实现模块**与 `apply_sis_class_defaults` 自动调度、第一类系统实验与平台结论、`sis_scoring` 阶梯计分验证流水线。代码开源：[https://github.com/arriverl/sisinf](https://github.com/arriverl/sisinf)。

---

## 2. 原理或方案设计

### 2.1 公共建模基础

**残差形式：**

$$u = r(v) = \mathrm{center}(t - Av \bmod q)$$

可行性 ⟺ $\|r(v)\|_\infty\le\gamma$ 且 $\|v\|_\infty\le\gamma$。第三类另须 $\|u\|_2^2+\|v\|_2^2<q^2$。

**字典序目标：**

$$V=\#\text{超界坐标},\quad S=\sum\text{溢出量},\quad M=\max\text{溢出}$$

按 $(V,S,M)$ 字典序下降；第三类在 $L_\infty$ 接近可行后再引入 $L_2$ 超额惩罚。

**统一流水线：** `apply_sis_class_defaults` → 类别化种子 → 多重重启局部搜索 → CP-SAT 收尾。

| 阶段 | 第一类 | 第二类 | 第三类 |
|------|--------|--------|--------|
| 实现模块 | `lattice_bkz` + `lattice_sieve` | `lattice_kannan` | `lattice_restricted_svp` |
| 搜索侧重 | 核游走、$u$ 优先 | pull kick、模拉回 | 稀疏化、$L_2$ 惩罚 |
| ILP | full | full | lex + 抛光 |
| 附加校验 | 非平凡 | — | $\|u\|_2^2+\|v\|_2^2<q^2$ |

### 2.2 第一类：近似 SVP（题 1, 3, 6, 9）

官方：BKZ 2.0 + 筛法。本文：BKZ 2.0 + **list sieve**（`lattice_sieve`，BDGL 轻量替代）→ $L_\infty$ 字典序搜索 → full CP-SAT。**可得：** $E_\infty=42$–45，0 分档。

### 2.3 第二类：近似 CVP（题 2, 4, 7, 10）

官方：Kannan 嵌入 + 调参 BKZ。本文：`lattice_kannan` 构造嵌入格并 BKZ，并行 CVP 提升 + 模拉回 + full CP-SAT。无 fpylll 时退化为 CVP 启发式。

### 2.4 第三类：受限 SVP（题 5, 8）

官方：禁止 $L_2$ 短向量后过滤；Wang et al. (2025) enumerate-then-slice + dimension-for-free。本文：`wang_restricted_svp_v_seeds`（BKZ 尾块枚举 → L∞/L₂ slice）+ 稀疏增广 + lex CP-SAT + `euclid_polish`。

### 2.5 共用 CP-SAT 建模

对 $v_j\in[-\gamma,\gamma]$，每行引入 $k_i$ 线性化模同余。三种模式：full（类一、二）、chunk（降维，平台处无效）、lex（类三）。

---

## 3. 程序实现或算法分析

### 3.1 软件模块

| 模块 | 功能 |
|------|------|
| `lattice_bkz` | LLL/BKZ 2.0 多 $\beta$ 种子 |
| `lattice_sieve` | BKZ 后缀 list sieve |
| `lattice_kannan` | Kannan 嵌入 + BKZ |
| `lattice_restricted_svp` | $L_\infty$ 盒稀疏采样 |
| `sis_problem_taxonomy` | 题号分类与 $<q^2$ 校验 |
| `sis_scoring` | 阶梯计分 |
| `run_full_validation.py` | 十题 batch + ILP + JSON 报告 |

验证：`check_algorithms.py` → `run_full_validation.py --batch-rounds 6 --ilp-time-limit 3600`（服务器 + fpylll）。

### 3.2 第一类实验结果（$\gamma=15$，$q=100$）

| 题号 | $n=m$ | 启发式 | ILP 后 | 距可行 |
|------|-------|--------|--------|--------|
| 1 | 100 | 44 | **42** | 27 |
| 3 | 120 | 45 | **43** | 28 |
| 6 | 140 | 45 | **44** | 29 |
| 9 | 160 | 45 | **45** | 30 |

**主要发现：**

- 全维 ILP 首轮约降 1–2，再加时平台（题 1：42@1h=42@2h）
- 分块 12 轮题 1 全部未改进，$\mathrm{max\_over}=27$ 锁死
- 加强 CP 13h 仍 44；小子格 BKZ 后搜索恶化
- 题 1：60% 行 $|u_i|>15$，96% 维 $v\neq 0$——全局协调失败
- 维度越高 ILP 收益越小

### 3.3 与文献可解方案对比

| 方法 | 对本赛题 |
|------|----------|
| LLL/BKZ | 仅 $L_2$ 短；作种子 |
| BKZ + list sieve | 已接入；非完整 BDGL |
| Kannan 嵌入 | 类二已接入；依赖 fpylll |
| 受限 SVP 采样 | 类三已接入 |
| 全维 ILP | **唯一稳定压降**；1h 后平台 |
| 分块 ILP | 平台处无效 |

### 3.4 最终可行可得分解

- **第一类（已验证）：** 同余可行，$E_\infty=42$–45 → **0 分**；突破需完整 BDGL 或更强受限搜索
- **第二类：** 已接入 Kannan；预期 $E_\infty\approx 40$–50，待服务器评测
- **第三类：** 已接入盒采样与 $<q^2$ 校验；双重门槛 $E_\infty$ 与 $L_2<q^2$

---

## 4. 总结

对照官方三类解析与 Chen & Nguyen (2011)、Wang et al. (2025)、Dilithium 规范：本文已实现 BKZ+sieve、Kannan 与受限 SVP 三类种子，当前仅第一类有完整实验数据且为 0 分档。进入得分档（$E_\infty\le\gamma+4$）须升级 sieve 或扩大受限采样规模。

**展望**：十题 `run_full_validation` 报告；list sieve → BDGL；lex 中欧氏硬约束；Micciancio–Peikert 参数解释平台。开源代码：[https://github.com/arriverl/sisinf](https://github.com/arriverl/sisinf)。

---

## 参考文献（节选）

1. Ajtai, STOC 1996  
2. Lenstra–Lenstra–Lovász, Math. Ann. 1982 (LLL)  
3. Chen & Nguyen, ASIACRYPT 2011 (BKZ 2.0)  
4. Becker et al., SODA 2016 (筛法)  
5. Wang et al., PQCrypto 2025 (受限 SVP)  
6. Bai et al., Dilithium Round 3 (NIST)  
7. arriverl/sisinf, GitHub 2026  

完整列表见 `COURSE_PAPER.tex`。
