# fx.gemm

| 接 | 调 | 看 |
| :--- | :--- | :--- |
| `a`=im2col，`b`=flatten kernel → `out`。 | 无参。 | `out.data[i] = a.data[i] * b.data[0]`。 |
