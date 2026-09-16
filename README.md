# cgraph-adapters

**Operator adapter layer** for cgraph:

| Path | Role |
| --- | --- |
| `gate/fixture` | Kernel-acceptance `fx.*` fixtures (not production) |
| `gate/cloud` | Gate-level cloud stubs (not production) |
| `native/` | First-party production implementations |
| `vendor/rtr/` | Thin wrappers over third-party `real_time_registration` (never vendor the source tree) |

## Build

Requires an installed `cgraph` package (`find_package(cgraph)`):

```bat
cmake -S . -B build -DCMAKE_PREFIX_PATH=..\cgraph-kernel\install
cmake --build build
```

Dynlib plugins are written to `build/operators/`. Point the kernel at them:

```bat
set CGRAPH_OPERATOR_PATH=%CD%\build\operators
```

## Remotes

- https://github.com/xiandenis/cgraph-adapters.git
