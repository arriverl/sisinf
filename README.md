# 2026 密码数学挑战赛赛题一：SIS∞ 解题工作区

**开源地址：** [https://github.com/arriverl/sisinf](https://github.com/arriverl/sisinf)

**完整课程论文（暨南大学格式）：**

| 文件 | 说明 |
|------|------|
| [`docs/COURSE_PAPER.md`](docs/COURSE_PAPER.md) | Markdown 完整版（含封面、公式、四章正文、附录） |
| [`docs/COURSE_PAPER.tex`](docs/COURSE_PAPER.tex) | LaTeX 正式排版（XeLaTeX 编译） |

```bash
cd docs && xelatex COURSE_PAPER.tex && xelatex COURSE_PAPER.tex
```

---

## 论文结构（暨南大学本科生课程论文格式）

```
封面（学院、专业、姓名、学号、课程、指导教师、日期）
摘要 + 关键词
目录
├── 1. 引言
├── 2. 原理或方案设计
├── 3. 程序实现或算法分析
├── 4. 总结
├── 参考文献（五号）
└── 附录（五号）
```

详细内容见 [`docs/COURSE_PAPER.md`](docs/COURSE_PAPER.md)。

---

## 1. 引言（摘要）

赛题求 $Av+u\equiv t\pmod q$ 且 $\|u\|_\infty,\|v\|_\infty\le\gamma$，$n=m\in\{100,120,140,160\}$，$\gamma\in\{15,16,17,18\}$。十题分三类：

| 类 | 题号 | 路线 |
|:--:|:----:|:-----|
| 一 | 1,3,6,9 | 近似 SVP：BKZ + 筛法 |
| 二 | 2,4,7,10 | 近似 CVP：Kannan 嵌入 |
| 三 | 5,8 | 受限 SVP + $\|u\|_2^2+\|v\|_2^2<q^2$ |

官方**阶梯计分**：$E_\infty=\max(\|u\|_\infty,\|v\|_\infty)$，按偏移给 0–10 分。

---

## 2. 原理或方案设计（核心公式）

### 2.1 残差主导建模

$$u = r(v) = \mathrm{center}(t - Av \bmod q)$$

可行性 ⟺ $\|r(v)\|_\infty\le\gamma \land \|v\|_\infty\le\gamma$

### 2.2 Chebyshev 字典序

$$M=\max\text{溢出},\quad V=\text{违规坐标数},\quad S=\text{溢出量和}$$

字典序 $(M,V,S)$，与阶梯计分对齐。

### 2.3 三阶段流水线

```
题号识别 → 格种子(BKZ/G6K/Kannan/Wang) → 字典序局部搜索 → CP-SAT收尾
```

### 2.4 七大创新点

残差主导建模 · Chebyshev 字典序 · 三类自动调度 · 格种子分层栈 · $O(n)$ 列增量 · lex 双约束收尾 · 可行可得分解

---

## 3. 程序实现与运行

### 3.1 目录结构

```
scripts/
├── solve_sisinf.py    # 主求解器（SearchConfig、局部搜索）
├── sis_common.py      # 题号分类、阶梯计分、full-max 参数、mod-q 核基
├── lattice_seeds.py   # BKZ / G6K / sieve / Kannan / Wang 格种子
├── sis_heuristics.py  # Wagner、u 优先算子等启发式
├── sis_finish.py      # CP-SAT 收尾 + execute_finish 管线
├── sis_run.py         # 十题全量验证 + 按类 batch（主运行入口）
├── sis_check.py       # 环境自检（fpylll / G6K / ILP / 冒烟 / 算法）
└── install_g6k.sh     # G6K 安装
```

### 3.2 环境安装（Linux 推荐）

```bash
conda activate cwh
cd /home/cwh/sisinf
conda install -c conda-forge fpylll -y
pip install numpy ortools "Cython>=3.0" cysignals
bash scripts/install_g6k.sh
python3 scripts/sis_check.py all
python3 scripts/sis_check.py g6k
```

### 3.3 运行命令

| 场景 | 命令 |
|------|------|
| 冒烟 | `python3 scripts/sis_check.py smoke` |
| 快速十题（1–3h） | `python3 scripts/sis_run.py --quick` |
| **标准全量（推荐）** | `python3 scripts/sis_run.py --batch-rounds 6 --ilp-time-limit 3600` |
| 论文全量 | `python3 scripts/sis_run.py --full-max --batch-rounds 24 --ilp-time-limit 14400` |
| 分三类 | `python3 scripts/sis_run.py batch --class {1,2,3} --full-max --max-rounds 24` |
| 单题 | `python3 scripts/sis_run.py --problems 1 --batch-rounds 6 --ilp-time-limit 3600` |

报告：`results/full_validation/full_validation_report.json`

### 3.4 实验结果（标准全量 `batch=6`, `ilp=3600s`）

| 题号 | 类 | $\gamma$ | 启发式 | ILP 后 | 得分 | 状态 |
|:----:|:--:|:--------:|:------:|:------:|:----:|:----:|
| 1 | 一 | 15 | 44 | **42** | 0 | ✓ |
| 2 | 二 | 15 | 42 | **42** | 0 | ✓ |
| 3 | 一 | 16 | 45 | **43** | 0 | ✓ |
| 4 | 二 | 16 | 45 | **44** | 0 | ✓ |
| 5–10 | — | — | — | — | — | 跑批中 |

完整十题表见 [`docs/COURSE_PAPER.md`](docs/COURSE_PAPER.md) §3.7。

---

## 4. 总结

残差主导 + 三类差异化 + 格种子/搜索/CP-SAT 混合框架已开源。第一类 $E_\infty$ 可达 42–45；进入得分档需更强 BDGL 或更大 Wang 受限搜索。

---

## Windows 本地注意

推荐使用独立虚拟环境避免 ortools/protobuf 冲突：

```powershell
cd sisinf_challenge2026
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

---

## 输入格式

```json
[{"id": 1, "n": 100, "m": 100, "q": 100, "gamma": 15, "A": [[...]], "t": [0, ...]}]
```

## 提交前自检

- 同余：$Av+u\equiv t\pmod q$
- 无穷范数：$\|u\|_\infty,\|v\|_\infty\le\gamma$
- 第三类：$\|u\|_2^2+\|v\|_2^2<q^2$
- 齐次题：$u,v$ 非全零

---

*完整公式推导、算法伪代码、复杂度分析、附录参数表见 [`docs/COURSE_PAPER.md`](docs/COURSE_PAPER.md)。*
