# 旧生产文件清理记录

2026-10-29 已按以下精确清单删除 **42 个既有生产文件**：modules/operators 下 27 个，extensions/operators/standard 下 15 个。选定的新领域实现见 [Operators.md](Operators.md)，来源实现归 frontend。

**当前状态：已完成。** 删除前确认全部文件与已读取的基线一致，并建立临时归档、逐文件核验归档内容；再次核对后仅删除清单中的普通文件，没有递归删除目录。随后通过 rmdir 移除 9 个已空旧目录；modules/operators 已不存在，standard 目录只保留说明与原测试。

以下不包含测试、fixture、规则文件、兄弟插件、用户数据或第三方文件。保留 extensions/operators/standard/README.md 与 tests/OperatorsTest.cpp；保留所有其他已有测试及 contracts 中单份 OpFusionStandardTests 目标登记。清理期间未修改生产源码或构建配置。

## modules/operators：27 个

```text
modules/operators/CMakeLists.txt
modules/operators/README.md
modules/operators/include/opfusion/BuiltinOperators.h
modules/operators/include/opfusion/operators/Add.h
modules/operators/include/opfusion/operators/Concat.h
modules/operators/include/opfusion/operators/Conv2D.h
modules/operators/include/opfusion/operators/MatMul.h
modules/operators/include/opfusion/operators/MaxPool2D.h
modules/operators/include/opfusion/operators/Mul.h
modules/operators/include/opfusion/operators/ReLU.h
modules/operators/include/opfusion/operators/Resize.h
modules/operators/include/opfusion/operators/Sigmoid.h
modules/operators/include/opfusion/operators/Slice.h
modules/operators/src/Add.cpp
modules/operators/src/Concat.cpp
modules/operators/src/Conv2D.cpp
modules/operators/src/MatMul.cpp
modules/operators/src/MaxPool2D.cpp
modules/operators/src/Mul.cpp
modules/operators/src/ReLU.cpp
modules/operators/src/Resize.cpp
modules/operators/src/Sigmoid.cpp
modules/operators/src/Slice.cpp
modules/operators/src/SemanticChecks.h
modules/operators/src/SemanticChecks.cpp
modules/operators/src/SourceFields.h
modules/operators/src/SourceFields.cpp
```

## extensions/operators/standard：15 个

```text
extensions/operators/standard/CMakeLists.txt
extensions/operators/standard/plugin.toml
extensions/operators/standard/src/Operators.cpp
extensions/operators/standard/src/SourceBinding.cpp
extensions/operators/standard/include/opfusion/standard/Operators.h
extensions/operators/standard/include/opfusion/standard/Add.h
extensions/operators/standard/include/opfusion/standard/Concat.h
extensions/operators/standard/include/opfusion/standard/Conv2D.h
extensions/operators/standard/include/opfusion/standard/MatMul.h
extensions/operators/standard/include/opfusion/standard/MaxPool2D.h
extensions/operators/standard/include/opfusion/standard/Mul.h
extensions/operators/standard/include/opfusion/standard/ReLU.h
extensions/operators/standard/include/opfusion/standard/Resize.h
extensions/operators/standard/include/opfusion/standard/Sigmoid.h
extensions/operators/standard/include/opfusion/standard/Slice.h
```

## 清理结果与遗留工作

modules/operators 已移除，根构建不再引用该模块；旧 standard 的 manifest、生产 CMake、头和源均已删除。插件默认值保持 ON，发现规则未改，也没有按旧包名称排除。静态核对 OpFusionStandardTests 只在 contracts 中登记一份，仍引用原测试源，不再存在旧生产 CMake 导致的同名目标重复。

清理没有自动解决旧测试的头路径、命名空间和 API 迁移。测试源与目标保留未改，后续须单独恢复测试迁移；不提供伪兼容头，也不删除/禁用测试来掩盖。本轮只做文件/引用/摘要等静态检查，未配置、编译、运行 CLI 或测试，不能宣称默认构建或行为验收通过。
