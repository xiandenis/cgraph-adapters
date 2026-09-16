# cloud.stat_outlier · Usage

| 接 | 调 | 看 |
| --- | --- | --- |
| `in: cloud{xyz}` → `out` | `mean_k`/`std_mul`；`min_points` 剩余点数下界（空云） | 门禁 SOR；MAP body 改 min_points 不重跑聚类 |

门禁实现（内核验收）。用例 12 的「过滤体」。
