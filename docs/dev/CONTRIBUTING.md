# 贡献指南 — cgraph-adapters

## 远程与分支

- 默认：`origin` → 内网 GitLab  
- 可选：`github` → GitHub  

```bat
git push -u origin HEAD
```

## 新增算子（最小路径）

1. 选定落点：`gate/fixture`（仅验收） / `native/`（自研生产） / `vendor/<lib>/`（薄封装第三方）。  
2. 实现 Process：端口 TypeId+Semantic；execute 用 DataObject。  
3. 若引入新类型：加 Type Pack YAML，并保证 codec / Catalog 可安装。  
4. CMake：编进对应 library，SHARED plugin 输出到 `operators/`。  
5. 加 C++ 测或 studio journey（生产算子至少有一条可复现路径）。  
6. 本仓提交；若依赖新 kernel ABI，姊妹仓须已 install 对应版本。

## 改 RTR 包装

- 只改 `vendor/rtr/` 封装与测试。  
- 升级算法：在外部 RTR 仓发版 → 刷新 `install_party` → `sync-rtr-install.cmd` → 再编本仓。  
- 禁止 clone RTR 源码进工作空间或本仓。

## 跟内核清扫

内核去掉 dtype/json 类型轴之后，本仓端口与 fixture **必须**对齐 TypeId+Semantic；不要保留双轨。

## 提交检查

- [ ] `CMAKE_PREFIX_PATH` 指向当前 kernel install  
- [ ] Release 编过；RTR 开时 `ctest` 相关测通过  
- [ ] 未引入 RTR/算法源码树  
- [ ] 插件与 `cgraph-types.yaml`（若有）同目录可被扫描
