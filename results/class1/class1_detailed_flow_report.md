# 第一类赛题（1、3、6、9）详细解题流程报告

生成时间：2026-05-31T10:45:52

## 总体框架（README：Dual-Space + Entropy IIR-CLS）

| 阶段 | 内容 | 第一类扩展 |
|------|------|------------|
| 建模 | r(v)=Center(t−Av)，u=r(v) | t≡0 齐次；拒绝 u=v=0 |
| 格空间 | Ajtai 嵌入 + BKZ/LLL | β≈28–32，组合短向量种子 |
| Dual 空间 | pull/投影/稀疏/随机候选 | 无 CVP lift |
| 模 q 核 | v←clip(v+Kd) 保持 u | SymPy 或素数域高斯 |
| 残差精修 | 坐标下降+Pair+CP+Kick | Chebyshev 分层目标 |

**可行解数量：0/4**

## 小问 1（n=m=100, q=100, γ=15）

### 0_建模
残差 r(v)=Center(t-Av)；u=r(v)；在 v∈[-γ,γ]^m 上使 |r|_∞≤γ；齐次须 u,v 非全零
```json
{
  "residual_at_v0": {
    "violations": 0,
    "overflow_sum": 0,
    "max_overflow": 0,
    "inf_u": 0,
    "inf_v": 0
  }
}
```

### 1_BKZ格种子
```json
{
  "enabled": true,
  "seeds": 0,
  "beta": 28,
  "combo_depth": 5,
  "elapsed_sec": 0.0
}
```

### 2_模q核
```json
{
  "dim_m": 100,
  "basis_cols": 3,
  "kernel_walk_every": 20,
  "elapsed_sec": 2.768
}
```

### 3_Dual空间候选
```json
{
  "num_candidates": 24,
  "top_scored_preview": [
    {
      "violations": 0,
      "overflow_sum": 0,
      "max_overflow": 0,
      "inf_v": 0
    },
    {
      "violations": 67,
      "overflow_sum": 1138,
      "max_overflow": 33,
      "inf_v": 15
    },
    {
      "violations": 71,
      "overflow_sum": 1230,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 72,
      "overflow_sum": 1193,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 72,
      "overflow_sum": 1289,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 59,
      "overflow_sum": 1115,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 60,
      "overflow_sum": 1135,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 63,
      "overflow_sum": 1125,
      "max_overflow": 35,
      "inf_v": 15
    }
  ],
  "elapsed_sec": 0.306
}
```

### 4_配置摘要
```json
{
  "restarts": 16,
  "iters": 6000,
  "timeout_sec": 360.0,
  "bkz_beta": 28,
  "kernel_walk_every": 20,
  "pair_relief_every": 24,
  "block_cp_every": 80,
  "euclid_weight": 0.5,
  "entropy_weight": 0.15
}
```

### 5_局部搜索
```json
{
  "elapsed_sec": 5778.762,
  "success": false,
  "verify": {
    "congruence_ok": 1,
    "inf_u": 43,
    "inf_v": 15,
    "norm_sq": 78873,
    "norm_req_ok": 1,
    "nontrivial_ok": 1
  },
  "meta": {
    "restart": 2,
    "steps": 6000,
    "violations": 64,
    "overflow_sum": 1049,
    "max_overflow": 28,
    "feasible": 0,
    "congruence_ok": 1,
    "inf_u": 43,
    "inf_v": 15,
    "norm_sq": 78873,
    "norm_req_ok": 1,
    "nontrivial_ok": 1,
    "energy": 64002185.314539716,
    "entropy": 3.272803804040776,
    "dual_candidates": 24
  },
  "final_residual_stats": {
    "violations": 64,
    "overflow_sum": 1049,
    "max_overflow": 28,
    "inf_u": 43,
    "inf_v": 15
  }
}
```

**结果**：success=False，inf_u=43，inf_v=15，congruence=1

## 小问 3（n=m=120, q=100, γ=15）

### 0_建模
残差 r(v)=Center(t-Av)；u=r(v)；在 v∈[-γ,γ]^m 上使 |r|_∞≤γ；齐次须 u,v 非全零
```json
{
  "residual_at_v0": {
    "violations": 0,
    "overflow_sum": 0,
    "max_overflow": 0,
    "inf_u": 0,
    "inf_v": 0
  }
}
```

### 1_BKZ格种子
```json
{
  "enabled": true,
  "seeds": 0,
  "beta": 28,
  "combo_depth": 5,
  "elapsed_sec": 0.0
}
```

### 2_模q核
```json
{
  "dim_m": 120,
  "basis_cols": 32,
  "kernel_walk_every": 20,
  "elapsed_sec": 3.904
}
```

### 3_Dual空间候选
```json
{
  "num_candidates": 24,
  "top_scored_preview": [
    {
      "violations": 0,
      "overflow_sum": 0,
      "max_overflow": 0,
      "inf_v": 0
    },
    {
      "violations": 82,
      "overflow_sum": 1480,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 88,
      "overflow_sum": 1529,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 90,
      "overflow_sum": 1593,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 67,
      "overflow_sum": 1175,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 71,
      "overflow_sum": 1240,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 78,
      "overflow_sum": 1475,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 79,
      "overflow_sum": 1300,
      "max_overflow": 35,
      "inf_v": 15
    }
  ],
  "elapsed_sec": 0.003
}
```

### 4_配置摘要
```json
{
  "restarts": 16,
  "iters": 6000,
  "timeout_sec": 360.0,
  "bkz_beta": 28,
  "kernel_walk_every": 20,
  "pair_relief_every": 24,
  "block_cp_every": 80,
  "euclid_weight": 0.5,
  "entropy_weight": 0.15
}
```

### 5_局部搜索
```json
{
  "elapsed_sec": 12565.37,
  "success": false,
  "verify": {
    "congruence_ok": 1,
    "inf_u": 45,
    "inf_v": 15,
    "norm_sq": 85114,
    "norm_req_ok": 1,
    "nontrivial_ok": 1
  },
  "meta": {
    "restart": 3,
    "steps": 6000,
    "violations": 78,
    "overflow_sum": 1228,
    "max_overflow": 30,
    "feasible": 0,
    "congruence_ok": 1,
    "inf_u": 45,
    "inf_v": 15,
    "norm_sq": 85114,
    "norm_req_ok": 1,
    "nontrivial_ok": 1,
    "energy": 78002445.05415273,
    "entropy": 3.0112970665288916,
    "dual_candidates": 24
  },
  "final_residual_stats": {
    "violations": 78,
    "overflow_sum": 1228,
    "max_overflow": 30,
    "inf_u": 45,
    "inf_v": 15
  }
}
```

**结果**：success=False，inf_u=45，inf_v=15，congruence=1

## 小问 6（n=m=140, q=100, γ=15）

### 0_建模
残差 r(v)=Center(t-Av)；u=r(v)；在 v∈[-γ,γ]^m 上使 |r|_∞≤γ；齐次须 u,v 非全零
```json
{
  "residual_at_v0": {
    "violations": 0,
    "overflow_sum": 0,
    "max_overflow": 0,
    "inf_u": 0,
    "inf_v": 0
  }
}
```

### 1_BKZ格种子
```json
{
  "enabled": true,
  "seeds": 0,
  "beta": 28,
  "combo_depth": 5,
  "elapsed_sec": 0.0
}
```

### 2_模q核
```json
{
  "dim_m": 140,
  "basis_cols": 3,
  "kernel_walk_every": 20,
  "elapsed_sec": 2509.836
}
```

### 3_Dual空间候选
```json
{
  "num_candidates": 24,
  "top_scored_preview": [
    {
      "violations": 0,
      "overflow_sum": 0,
      "max_overflow": 0,
      "inf_v": 0
    },
    {
      "violations": 96,
      "overflow_sum": 1721,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 96,
      "overflow_sum": 1812,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 97,
      "overflow_sum": 1555,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 100,
      "overflow_sum": 1799,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 83,
      "overflow_sum": 1610,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 86,
      "overflow_sum": 1750,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 87,
      "overflow_sum": 1615,
      "max_overflow": 35,
      "inf_v": 15
    }
  ],
  "elapsed_sec": 0.034
}
```

### 4_配置摘要
```json
{
  "restarts": 16,
  "iters": 6000,
  "timeout_sec": 360.0,
  "bkz_beta": 28,
  "kernel_walk_every": 20,
  "pair_relief_every": 24,
  "block_cp_every": 80,
  "euclid_weight": 0.5,
  "entropy_weight": 0.15
}
```

### 5_局部搜索
```json
{
  "elapsed_sec": 11806.391,
  "success": false,
  "verify": {
    "congruence_ok": 1,
    "inf_u": 46,
    "inf_v": 15,
    "norm_sq": 97393,
    "norm_req_ok": 1,
    "nontrivial_ok": 1
  },
  "meta": {
    "restart": 14,
    "steps": 6000,
    "violations": 80,
    "overflow_sum": 1319,
    "max_overflow": 31,
    "feasible": 0,
    "congruence_ok": 1,
    "inf_u": 46,
    "inf_v": 15,
    "norm_sq": 97393,
    "norm_req_ok": 1,
    "nontrivial_ok": 1,
    "energy": 80002576.49926563,
    "entropy": 3.4764583128110944,
    "dual_candidates": 24
  },
  "final_residual_stats": {
    "violations": 80,
    "overflow_sum": 1319,
    "max_overflow": 31,
    "inf_u": 46,
    "inf_v": 15
  }
}
```

**结果**：success=False，inf_u=46，inf_v=15，congruence=1

## 小问 9（n=m=160, q=100, γ=15）

### 0_建模
残差 r(v)=Center(t-Av)；u=r(v)；在 v∈[-γ,γ]^m 上使 |r|_∞≤γ；齐次须 u,v 非全零
```json
{
  "residual_at_v0": {
    "violations": 0,
    "overflow_sum": 0,
    "max_overflow": 0,
    "inf_u": 0,
    "inf_v": 0
  }
}
```

### 1_BKZ格种子
```json
{
  "enabled": true,
  "seeds": 0,
  "beta": 28,
  "combo_depth": 5,
  "elapsed_sec": 0.0
}
```

### 2_模q核
```json
{
  "dim_m": 160,
  "basis_cols": 19,
  "kernel_walk_every": 20,
  "elapsed_sec": 19016.096
}
```

### 3_Dual空间候选
```json
{
  "num_candidates": 24,
  "top_scored_preview": [
    {
      "violations": 0,
      "overflow_sum": 0,
      "max_overflow": 0,
      "inf_v": 0
    },
    {
      "violations": 106,
      "overflow_sum": 1878,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 115,
      "overflow_sum": 1999,
      "max_overflow": 34,
      "inf_v": 15
    },
    {
      "violations": 95,
      "overflow_sum": 1668,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 97,
      "overflow_sum": 1955,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 100,
      "overflow_sum": 1985,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 102,
      "overflow_sum": 1655,
      "max_overflow": 35,
      "inf_v": 15
    },
    {
      "violations": 102,
      "overflow_sum": 1801,
      "max_overflow": 35,
      "inf_v": 15
    }
  ],
  "elapsed_sec": 0.012
}
```

### 4_配置摘要
```json
{
  "restarts": 16,
  "iters": 6000,
  "timeout_sec": 360.0,
  "bkz_beta": 28,
  "kernel_walk_every": 20,
  "pair_relief_every": 24,
  "block_cp_every": 80,
  "euclid_weight": 0.5,
  "entropy_weight": 0.15
}
```

### 5_局部搜索
```json
{
  "elapsed_sec": 7822.556,
  "success": false,
  "verify": {
    "congruence_ok": 1,
    "inf_u": 46,
    "inf_v": 10,
    "norm_sq": 125377,
    "norm_req_ok": 1,
    "nontrivial_ok": 1
  },
  "meta": {
    "restart": 0,
    "steps": 6000,
    "violations": 111,
    "overflow_sum": 1851,
    "max_overflow": 31,
    "feasible": 0,
    "congruence_ok": 1,
    "inf_u": 46,
    "inf_v": 10,
    "norm_sq": 125377,
    "norm_req_ok": 1,
    "nontrivial_ok": 1,
    "energy": 111003109.20844634,
    "entropy": 2.0207154830989342,
    "dual_candidates": 24
  },
  "final_residual_stats": {
    "violations": 111,
    "overflow_sum": 1851,
    "max_overflow": 31,
    "inf_u": 46,
    "inf_v": 10
  }
}
```

**结果**：success=False，inf_u=46，inf_v=10，congruence=1
