# 上手指南 — cgraph-adapters

面向：写/改算子插件、接 RTR、跟内核 ABI 的工程师。

## 前置

| 依赖 | 说明 |
| --- | --- |
| 已安装的 **cgraph** | `find_package(cgraph CONFIG REQUIRED)`，通常来自工作空间 `install/` |
| vcpkg + Eigen | `vcpkg.json` / `Eigen3` |
| （默认开）RTR 预编译 | `find_package(real_time_registration)`；**不要**把 RTR 源码放进本仓 |
| CMake ≥ 3.24、C++17 | Windows 主路径 |

## 构建（工作空间推荐）

在 **LidarWorkFlow2 根**：

```bat
REM 1) kernel 已 install 到 %CD%\install
REM 2) 同步 RTR 预编译树（默认源 D:\work_space\real_time_registration\install_party）
sync-rtr-install.cmd

cmake -S cgraph-adapters -B build\cgraph-adapters ^
  -DCMAKE_PREFIX_PATH=%CD%\install ^
  -DCMAKE_INSTALL_PREFIX=%CD%\install ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build\cgraph-adapters --config Release
```

若 RTR 的 Config 还依赖 PCL/OpenCV/PDAL，把编 RTR 时用的 vcpkg 安装树也放进 `CMAKE_PREFIX_PATH`（分号分隔）。

关闭 RTR 包装（仅 fixture）：

```bat
cmake ... -DCGRAPH_ADAPTERS_WITH_RTR=OFF
```

## 产物与环境变量

| 产物 | 路径 |
| --- | --- |
| fixture 插件 | `build\cgraph-adapters\operators\cgraph_gate_fixture.dll` |
| RTR 插件 | `build\cgraph-adapters\operators\cgraph_rtr.dll` |
| RTR Type Pack | 同目录 `cgraph-types.yaml`（POST_BUILD 拷贝） |

```bat
set CGRAPH_OPERATOR_PATH=%CD%\build\cgraph-adapters\operators
```

Studio / 内核扫描该目录加载 Process；Type Pack YAML 由内核目录扫描安装（非 Process）。

## 只 clone 本仓时

1. 另装一份 kernel 到某 prefix（或拿同事的 `install/`）。  
2. 另备 RTR `install_party` 或等价 prefix。  
3. `CMAKE_PREFIX_PATH=<kernel-prefix>;<rtr-prefix>;<extra-vcpkg>`。  
4. 仍建议最终与团队统一到工作空间布局，避免多份 prefix 漂移。

## 测试（RTR 开时）

```bat
ctest --test-dir build\cgraph-adapters -C Release --output-on-failure
```

典型目标：`cgraph_rtr_test_json` / `unpack` / `rough_fine` / `wire` / `plugin_smoke`。

## 常见踩坑

1. **`find_package(cgraph)` 失败**：kernel 未 install，或 `CMAKE_PREFIX_PATH` 不对。  
2. **Catalog 没有 `rtr.*`**：`cgraph_rtr.dll` 的依赖 DLL（PCL/TBB/…）不在 `PATH`；工作空间用 `run-studio.cmd` 会补 vcpkg `bin`，或设 `CGRAPH_RUNTIME_PATH`。  
3. **Eigen 主版本与 RTR Config 冲突**：本仓 `cmake/rtr-dep-shim/` 会写入 CMake redirects；勿删。  
4. **把 RTR `.cpp` 拷进 `vendor/rtr/`**：禁止；只允许薄封装与 `catalog/cgraph-types.yaml`。

## 下一步

- 分层边界 → [ARCHITECTURE.md](ARCHITECTURE.md)  
- 加新算子 / 提交 → [CONTRIBUTING.md](CONTRIBUTING.md)
