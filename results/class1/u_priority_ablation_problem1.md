# problem1 u 优先消融（A/B/C）

| 组 | inf_u | inf_v | violations | max_overflow | 耗时(s) | success |
|----|-------|-------|------------|--------------|---------|---------|
| A_baseline | 46 | 10 | 57 | 31 | 126.93 | False |
| B_cover32x24 | 45 | 15 | 61 | 30 | 129.75 | False |
| C_cover40x32 | 45 | 15 | 65 | 30 | 131.21 | False |

**当前最优组**：`B_cover32x24`（inf_u=45）