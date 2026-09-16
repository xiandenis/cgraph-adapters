# fx.filter2d

| 接 | 调 | 看 |
| :--- | :--- | :--- |
| `image`, `kernel` ndarray JSON → `out`。 | P0：1×1 核、pad=0。 | `out.data[i] = image.data[i] * kernel.data[0]`。 |
