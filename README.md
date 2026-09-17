# cgraph-adapters

**Operator adapter layer** for cgraph:

| Path | Role |
| --- | --- |
| `gate/fixture` | Kernel-acceptance `fx.*` fixtures (not production) |
| `native/` | First-party production implementations |
| `vendor/rtr/` | Thin wrappers over third-party `real_time_registration` (never vendor the source tree) |

Gate-level `cloud.*` stubs were removed; production point-cloud ops land under `native/`.

## Build

Requires an installed `cgraph` package (`find_package(cgraph)`). From the **LidarWorkFlow2** workspace root (out-of-tree `build/cgraph-adapters`, shared `install/`):

```bat
cmake -S cgraph-adapters -B build\cgraph-adapters ^
  -DCMAKE_PREFIX_PATH=%CD%\install -DCMAKE_INSTALL_PREFIX=%CD%\install
cmake --build build\cgraph-adapters --config Release
```

`CGRAPH_ADAPTERS_WITH_RTR` defaults ON and `find_package(real_time_registration)` (sync with `sync-rtr-install.cmd` first). Plugin `cgraph_rtr` is written next to the fixture plugin.

Dynlib plugins are written to `build/cgraph-adapters/operators/` (or in-tree `build/operators/`). Point the kernel at them:

```bat
set CGRAPH_OPERATOR_PATH=%CD%\build\cgraph-adapters\operators
```

## Remotes

- https://github.com/xiandenis/cgraph-adapters.git
