# fx.dup

| 接 | 调 | 看 |
| :--- | :--- | :--- |
| `in: json`。下游可只连 `full` 或再补连 `head`。 | 无参数。尊重 `requested_outputs`。 | `full=in`。`head` 是字典序最小键的一对；空对象则为 `{}`。两口不得互相当 hit。 |
