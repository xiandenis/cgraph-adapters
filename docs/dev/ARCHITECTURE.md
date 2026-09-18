# 架构地图 — cgraph-adapters

## 职责边界

| 本仓做 | 本仓不做 |
| --- | --- |
| 实现 `cgraph::Process` 插件 | 改写内核对象模型来迁就算子 |
| Type Pack sidecar（YAML）与端口 TypeId+Semantic | 用 `make_port(..., "json")` / 裸 dtype 当类型 |
| 链接预编译 RTR / 其它第三方库 | 把第三方算法源码树 submodule 进仓 |
| gate fixture 支撑内核/Studio 旅程验收 | 把 `fx.*` 当成产品算子对外承诺 |

## 目录

```
cgraph-adapters/
  gate/fixture/       fx.* 验收夹具（静态库 + *_plugin → operators/）
  native/             自研生产算子（按需添加目标）
  vendor/rtr/         RTR 薄封装：src/、include/、catalog/cgraph-types.yaml、tests/
  cmake/rtr-dep-shim/ Eigen 版本重定向，满足 RTR Config 的 find_dependency
  CMakeLists.txt      插件 OUTPUT → ${CMAKE_BINARY_DIR}/operators
```

## 插件契约（读码要点）

1. **注册**：`plugin_register.cpp` 导出内核约定的注册入口，填 `op_id`、端口、execute。  
2. **端口**：每个 Port 声明 TypeId + Semantic；与 Catalog / Pack 一致。  
3. **载荷**：执行路径走 DataObject ABI，不要把业务状态藏进无类型 json 主载荷。  
4. **副作用**：遵守 EffectSpec；失败路径不得绕过 Commit Barrier 语义（由内核 enforce）。  

RTR 第一刀算子（见 `vendor/rtr/README.md`）：

| op_id | 角色 |
| --- | --- |
| `rtr.rough_global_reg` | 粗全局配准 |
| `rtr.fine_registration` | 精配准 |
| `rtr.align_result.unpack` | 拆 AlignResult 字段 |

## 与姊妹仓

```text
install(cgraph) ──find_package──► cgraph-adapters ──operators/*.dll──► Studio / Runtime
                                      │
                                      └── Type Pack YAML 同目录，供 Catalog 扫描
```

内核 bump ABI 后：先拿到新 `install/`，再重建本仓插件，再跑 studio 旅程。
