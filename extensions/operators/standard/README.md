# standard 历史测试说明

基础算子的生产实现归 [contracts](../../../modules/contracts/docs/Operators.md)，[同名纯领域头](../../../modules/contracts/include/opfusion/operators/)和[源](../../../modules/contracts/src/operators/)由 OpFusionContracts 提供。来源工厂独立归 frontend；apps 通过公共入口依次装配语义和绑定，不调用旧 opfusionRegisterStandard。

## 生产清理边界

2026-10-29 已删除本目录的 **15 个旧生产文件**，以及 modules/operators 的 27 个旧文件。清理时保留全部测试源，逐文件范围见 [LegacyCleanup.md](../../../modules/contracts/docs/LegacyCleanup.md)。本次迁移不恢复旧生产包或兼容头。

- 本目录不再包含 plugin.toml、生产 CMakeLists.txt、include 或 src，不再作为生产插件被发现。
- 插件选项仍默认 ON；没有关闭发现、添加 standard 名称例外或伪造新 API manifest。
- [OperatorsTest.cpp](tests/OperatorsTest.cpp) 迁移为 contracts 十个具体类的纯语义测试，使用 `opfusion/operators/` 和 `opfusion/registry/` 公共头、`opfusion::operators` 命名空间；不引入 frontend 依赖。
- 原来源 fixture 和 sourceTests 的全部用例意图迁至 [frontend/BenchmarkTest.cpp](../../../modules/frontend/tests/BenchmarkTest.cpp)：端口顺序/重复引用、九个有界来源绑定、严格字段/策略/类型及未知来源反例由来源模块负责。
- 纯语义测试保留原卷积、逐元素、切片、连接、重采样、池化和 MatMul 正反例；物理 Layout 误传由编译期类型断言拒绝，逻辑轴 AxisOrder 的全部值及非法枚举仍有运行期反例。正常签名同时核对当前索引协议，空维、标量和空归约不丢失。
- OpFusionStandardTests 仍由 contracts/CMakeLists.txt 单份登记并只链接 contracts；没有删除或禁用目标。本说明不是构建结果。

## 版本与待验证状态

按功能目录分层后的 C++ contractVersion 是 **4.0.0**；注册/工厂及 Operator 虚接口的 operatorApiVersion 保持 **3.0.0**。Registration 与 SourceBinding 分离，数学语义仍为 `opfusion:<ClassName>@1.0.0`，来源仍为受限 benchmark-json/1。纯语义登记调用 [registerBuiltinOperators](../../../modules/contracts/include/opfusion/registry/BuiltinOperators.h)，不安装来源绑定。

集成阶段已实际完成默认选项配置、完整构建及 **6/6 CTest 目标**，其中 OpFusionStandardTests 通过且仍仅链接 contracts；迁出的来源解析用例在 FrontendBenchmarkJson 中通过。frontend 的逗号小数 C locale 子用例因环境缺少 locale 报告 SKIP。新 CLI 已核对十语义/九绑定及八图端口依赖，但不证明 kernel、设备执行或性能；实际结果与限制见 [CodeQuality.md](../../../modules/contracts/docs/CodeQuality.md)。
