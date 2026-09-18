# cgraph-adapters

**算子适配层**：把领域实现打成 cgraph 可加载的动态库插件（Process + 可选 Type Pack）。

| 路径 | 角色 |
| --- | --- |
| `gate/fixture` | 内核验收用 `fx.*`（**非生产**） |
| `native/` | 自研生产算子（可空，按需加） |
| `vendor/rtr/` | 对预编译 `real_time_registration` 的**薄封装**（禁止 vendoring 算法源码） |

## 5 分钟上手（工作空间）

先装好 kernel 到统一 `install/`，再同步 RTR 预编译包：

```bat
REM 在 LidarWorkFlow2 根
cmake --build build\cgraph-kernel --config Release
cmake --install build\cgraph-kernel --prefix install
sync-rtr-install.cmd

cmake -S cgraph-adapters -B build\cgraph-adapters ^
  -DCMAKE_PREFIX_PATH=%CD%\install -DCMAKE_INSTALL_PREFIX=%CD%\install ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build\cgraph-adapters --config Release
set CGRAPH_OPERATOR_PATH=%CD%\build\cgraph-adapters\operators
```

细节见 **[docs/dev/GETTING_STARTED.md](docs/dev/GETTING_STARTED.md)**。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/dev/GETTING_STARTED.md](docs/dev/GETTING_STARTED.md) | 依赖、构建、RTR、测试 |
| [docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md) | fixture / native / RTR 边界 |
| [docs/dev/CONTRIBUTING.md](docs/dev/CONTRIBUTING.md) | 新算子、Type Pack、提交 |

RTR 薄封装说明亦见 [vendor/rtr/README.md](vendor/rtr/README.md)。

## Remotes

| 名 | URL |
| --- | --- |
| `origin`（默认） | http://192.168.110.124/realone/desktop/lidarprocess/cgraph-adapters.git |
| `github`（可选） | https://github.com/xiandenis/cgraph-adapters.git |

## 姊妹仓

- [cgraph-kernel](http://192.168.110.124/realone/desktop/lidarprocess/cgraph-kernel) — `find_package(cgraph)`  
- [cgraph-studio](http://192.168.110.124/realone/desktop/lidarprocess/cgraph-studio) — 通过 `CGRAPH_OPERATOR_PATH` 加载插件
