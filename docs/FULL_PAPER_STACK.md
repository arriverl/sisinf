# 论文全量方案（--full-max）

> 不考虑算力上限时的完整文献路线与服务器命令。实现开关：`--full-max`。

## 一、服务器运行命令（最新）

### 0. 一次性环境（Linux）

```bash
cd /path/to/sisinf_challenge2026
conda create -n sisinf python=3.11 -y && conda activate sisinf
conda install -c conda-forge fpylll -y
pip install numpy ortools
bash scripts/install_g6k.sh          # G6K BDGL2 真筛法
python3 scripts/check_algorithms.py
python3 scripts/check_g6k.py
```

### 1. 冒烟 / 自检

```bash
python3 scripts/smoke_algorithms.py
python3 scripts/check_algorithms.py
```

### 2. 标准全量验证（生产推荐）

```bash
python3 scripts/run_full_validation.py \
  --batch-rounds 6 \
  --ilp-time-limit 3600 \
  --output-dir results/full_validation
```

### 3. 论文全量拉满（`--full-max`）

```bash
bash scripts/run_server_max.sh
# 或等价：
python3 scripts/run_full_validation.py \
  --full-max \
  --batch-rounds 24 \
  --ilp-time-limit 14400 \
  --output-dir results/full_max_validation
```

### 4. 分三类长跑

```bash
python3 scripts/run_class_batch.py --class 1 --full-max --max-rounds 24 --seed 20260603
python3 scripts/run_class_batch.py --class 2 --full-max --max-rounds 24 --seed 20260603
python3 scripts/run_class_batch.py --class 3 --full-max --max-rounds 24 --seed 20260603
```

### 5. 单题调试

```bash
python3 scripts/run_full_validation.py --problems 5 --full-max --batch-rounds 12 --ilp-time-limit 7200
```

---

## 二、三类论文全量方案对照

| 类 | 题号 | 文献主线 | 本仓库模块（full-max） |
|----|------|----------|------------------------|
| **一** | 1,3,6,9 | Chen–Nguyen **BKZ 2.0** (ASIACRYPT 2011) + Becker **BDGL 筛法** (SODA 2016) + Wang **L∞ restricted slice** (PQCrypto 2025) + Wagner 子系统 | `lattice_bkz` → `lattice_g6k` (bdgl2) → `lattice_sieve` → `lattice_restricted_svp` → 核游走 → **full CP-SAT** |
| **二** | 2,4,7,10 | **Kannan 嵌入** (MOR 1987) + BKZ 2.0 + Babai/CVP 提升 + Dilithium 式 L∞ 截断 | `lattice_kannan` (β≥52) + CVP lift + 模拉回 → **full CP-SAT** |
| **三** | 5,8 | Wang **enumerate-then-slice** + **dimension-for-free** + BDGL 近似 SVP 列表 + 欧氏上界 | `lattice_g6k` 列表 → `wang_restricted_svp_v_seeds` (pool 8192) → 字典序搜索 → **lex CP-SAT** → `euclid_polish` |

---

## 三、`--full-max` 关键参数（拉满）

| 参数 | 类一 | 类二 | 类三 |
|------|------|------|------|
| `bkz_beta` | ≥56 | ≥52 | ≥56 |
| `use_g6k_sieve` | ✓ bdgl2 | — | ✓ |
| `g6k_saturation_ratio` | 0.95 | — | 0.95 |
| `g6k_max_lift_vectors` | 2048 | — | 2048 |
| `wang_enum_pool_size` | 4096 | — | 8192 |
| `restarts` | ≥160 | ≥160 | ≥160 |
| `timeout_sec`/restart | 7200s | 7200s | 7200s |
| ILP 收尾 | full 4h | full 4h | lex 4h + 抛光 |

---

## 四、专用库清单

| 库 | 用途 | 安装 |
|----|------|------|
| **fpylll** | LLL / BKZ 2.0 | `conda install -c conda-forge fpylll` |
| **g6k** | BDGL2 / Gauss 真筛法 | `bash scripts/install_g6k.sh` |
| **ortools** | CP-SAT 收尾 | `pip install ortools` |
| numpy | 数值 | `pip install numpy` |

G6K 不可用时自动回退 `list sieve` + Wang 尾块枚举，但**不等价**于论文全量 BDGL。

---

## 五、算力参考（拉满量级，仅供规划）

| 维度 | 单题 batch (24轮×2h) | 单题 ILP (4h) | 十题合计（粗估） |
|------|----------------------|---------------|------------------|
| n=m=100 | ~48h CPU | 4h | ~3–7 天（8–16 核） |
| n=m=160 | ~48h+ | 4h | ~1–2 周 |

建议：`OMP_NUM_THREADS=16`，类一/二/三分机器并行。
