# fx.const_matrix / fx.matvec / OP-C linalg

**JSON convention (nested row-major lists):**

| kind | form |
| :--- | :--- |
| vector | `[x, y, ...]` |
| matrix | `[[r0c0, r0c1, ...], ...]` |
| vector_array | `[[...], [...], ...]` |

`fx.const_matrix` also accepts `{ "shape": [rows, cols], "data": [row-major flat] }` and normalizes to nested lists.

| op | connect |
| :--- | :--- |
| `fx.const_matrix` / `fx.const_vector` / `fx.const_vector_array` | `params.value` → `out` |
| `fx.matvec` | `A`, `v` → `out` |
| `fx.vadd` / `fx.vsub` | `a`, `b` → `out` |
| `fx.l2sq` | `in` → scalar `out` |
| `fx.sum_reduce` | number list `in` → scalar `out` |
| `fx.quadratic_form` | `A`, `p` → `pᵀ A p` |
| `fx.det` / `fx.trace` / `fx.transpose` | matrix `in` → `out` |
