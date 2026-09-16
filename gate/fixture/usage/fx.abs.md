# fx.abs

| 接 | 调 | 看 |
| :--- | :--- | :--- |
| `in: json`（整数）→ `out: json`。 | 无参数。非整数或 `INT64_MIN` 溢出失败。 | `out = \|int64(in)\|`。`abs(5)` 与 `abs(-5)` 的 `output_digest` 全等（64 hex）。 |
