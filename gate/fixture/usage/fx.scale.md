# fx.scale

| 接 | 调 | 看 |
| :--- | :--- | :--- |
| `x: int` → `y: int`。 | `k: int`，**domain `k > 0`**。`k≤0` 在 execute 前拒绝，不写成功槽。 | `y = x * k`。改 `k` 会改 `node_key`。Capability / CostHint 不进身份。 |
