# C++ 整改状态

当前 ONNX/算子迁移采用 contract 5.0.0、operator API 4.0.0，算子身份不含版本；本页以下为此前阶段记录，不是本次验收结论。当前能力、命令和限制见 [根 README](../../../README.md)、[算子语义](Operators.md) 和 [frontend](../../frontend/README.md)。


本页记录截至 **2026-10-29** 的限定集成状态，不是全仓规范认证或硬件证据。本轮依据用户批准的范围实施设备资源协议、公共头/源功能归组、全部自有源码中文注释，以及直接相关旧测试迁移；允许本地配置、编译和测试，不下载依赖、不执行硬件工作。已有未提交工作保留，没有提交或推送。

## 本轮落地

1. 旧平铺公共头与对应实现按 common / graph / registry / platform / model / plan 归组，operators 保持原位；35 个既有文件迁移，调用方、构建登记和文档同步更新，不保留旧转发头。
2. 从平台拆出 Validation、Resource、Device 和 Boundary。Device 按值拥有描述，只定义 requiredResources / clone；CpuDevice 和 DeclaredDevice 直接派生，不承担探测、SDK、编译或执行。
3. Platform 独占多态设备，复制时逐个 clone，拒绝空实例、重复身份、空 clone 或变更 clone 身份；避免对象切片。派生规则必须遵守不可变纯查询契约，const 本身不保证深度不可变或线程安全。
4. 物理库存、设备最低规则、实际请求与 Boundary 动态限额分开。Candidate / 本地转换接入统一校验，Registers/Thread/Block 不能代替整任务 Compute，不自动补请求或重复登记资源活动。
5. 分数 CPU 配额保持原值；只有明确 cores 单位才与核心配额比较。同节点重叠 affinity 要求共同显式共享 Compute 约束；未知共享关系保持 Unresolved，已有约束未请求为 Invalid，不宣称完整 CPU 拓扑/隔离证明。
6. contracts / frontend / apps 的自有头源、测试和 CMake 补中文文件职责、公开契约及非显然实现理由；说明借用、单位、未知状态、证据及生命周期，不逐行翻译或机械造管理类。
7. 旧 Core / Model / standard / frontend / CLI 测试迁移到当前公开接口，保留反例并新增设备资源回归。旧 standard 的来源用例移到 frontend，纯语义目标仍只链接 contracts。CLI fixture 不依赖已删除的生产 manifest，不以整套 skip 掩盖缺数据。

公共头路径和 Device/Platform 所有权协议为破坏性变化，**contractVersion=4.0.0**；Operator 虚函数、注册/来源工厂协议未变，**operatorApiVersion=3.0.0**。数学语义仍为 1.0.0，benchmark-json/1、manifest schema、CLI JSON schema 及 cpp-operators/3、registry/cpp-static/3 身份格式标记不变。新路径与源码重新计算内容身份，静态调用者需重新编译，不承诺旧源码或二进制兼容。

## Plan 文件职责

本次建模一致性修正已完成全新临时目录构建与回归，详情见本节末；下面的迁移验证记录是历史背景。源码新增 InputsAvailable/OutputsDelivered 准确时刻、虚拟 Sink 零成本完成和累计资源偏序上界校验；回归继续保留 Simulation，不升级为设备证据。contractVersion 保持 4.0.0，无新增字段/单位变化，构建内容身份已随源码重新生成。

Plan 相关公开头在 [plan/Plan.h](../include/opfusion/plan/Plan.h)，所有内部校验集中于 [src/plan](../src/plan/)：

- Plan：阶段编排与派生任务图。
- PlanValidation：本次校验的索引、事件偏序、选择、分配和来源/交付。
- PlanTaskValidation：端口、区域任务、适配阶段与复用。
- PlanMemoryValidation：全体读者/传输区间、工作区和既有别名规则。
- PlanMemory：唯一 Allocation 的容量核验，不是内存管理器。
- PlanResourceValidation：给定时间表容量、成对排他复用顺序与任务峰值的偏序保守上界；上界超限保留 Unresolved。
- PlanConditions：数值验收声明与计时覆盖，不产生误差观测或实测时间。

输出启动前分配、完整区域输出可见、工作区全任务预留、全部读取/传输完成关系及唯一分配计费保持；偏序上界超限不证明全局不可行。除新设备资源接线外，本轮不扩展为上一轮建模审查中所有独立问题的修复。

### 本次建模一致性修正验证（2026-10-29）

- 经用户明确授权，在全新临时目录使用 GCC 11.4.0 / C++17 / Ninja，以 BUILD_TESTING=ON、opfusionEnableOperatorPlugins=ON 完成配置和全部 66 项构建步骤；apps 通过 OpFusion::System 链接既有实现。
- 6/6 CTest 目标通过；ConditionalModelAndPlan 为 390 项检查，BenchmarkIntegration 为 15 个黑盒测试。frontend 的 CommaDecimalLocale 子用例因系统缺少 locale 报告 SKIP，不代表已覆盖。
- 在独立临时对照中，分别将修改前的 PlanConditions、PlanValidation、PlanResourceValidation 对象链接到新增回归，其余实现保持本次版本。三个对照均以预期诊断缺失而失败，分别对应延后计时起点、虚拟汇延迟和仅两两检查的累计容量缺口；未替换工作区文件。
- 初次回归发现私有资源测试夹具缺少 EventOrder 的祖先闭包；补齐任务内部及跨任务的传递关系后，重新构建并完成上述全量回归，没有放宽通过条件。
- 实际构建、可执行产物及对照日志均位于临时目录；未修改旧构建缓存、治理提示词、范围策略或原始研究/benchmark 数据。没有下载依赖，没有设备执行/性能测量。

## 实际验证

### 基线与工具链

- 重构前在独立目录配置成功；完整构建在旧 CoreTest/ModelTest 编译阶段失败，包括旧 Registration、缺失 Operator 纯虚方法、PackingKind 和旧 Device 聚合接口。未运行基线 CTest。
- 迁移后使用另一全新目录，保持 **BUILD_TESTING=ON / opfusionEnableOperatorPlugins=ON**，以 Linux x86-64 / GCC 11.4.0 / C++17 / CMake 3.22.1 / Ninja 1.10.1 完成配置和全部目标编译；Python 3.10.12 仅驱动测试。
- 本轮不复用、修改或清理仓库旧 build 缓存，也不把历史二进制当作当前结果。

### 回归与贯穿

| 核验 | 真实结果 |
|---|---|
| ContractsCore | 通过：图端口、注册/绑定、冻结与工厂边界 |
| ConditionalModelAndPlan | 通过：九映射算术、精确未知集合、模拟不能生产准入、事件/缓冲/资源反例 |
| DeviceResourceContracts | 通过：派生查询、多态 clone/快照、规则覆盖、资源归属、CPU 配额/共享 |
| OpFusionStandardTests | 通过：保留的纯语义正反例与类型化轴/索引关系 |
| FrontendBenchmarkJson | 通过：中立来源、九工厂、严格输入与错误边界；缺少逗号小数 C locale，相关子用例明确 SKIP |
| BenchmarkIntegration | 通过：15 个 Python 黑盒测试，含八图导入与可达 manifest 负例，无 benchmark 缺失跳过 |
| 公开头自包含 | contracts / frontend 共 30 个头逐个独立通过 C++17 语法检查 |
| 新 CLI 注册目录 | 10 个语义、9 个来源绑定，MatMul 无绑定，Registry 身份重复查询一致 |
| 新 CLI 八图导入 | 264 节点、415 张量、301 条完整有序端口边；重新从端口推导的边多重集完全相同 |

CTest **6/6 目标通过、0 目标失败**不意味着 locale 子用例已覆盖，更不意味着设备、数值 kernel 或性能已验证。所有模拟记录仍为 Simulation；结构断言只允许明确列出的未知原因，另断言模拟不能进入生产候选/设备域或令完整 Plan 为 Valid。九映射的 2.90 ms 最小值只是给定假设的算术，不是设备时延。

导入后的 execution=not_established、candidate_search=not_searched、cost=null；平台、参数内容与 Boundary 仍为 not_provided，没有默认 CPU 或零成本回退。

复现构建参数可使用下列命令；build-contract4 表示新目录，已存在时另取新名，勿覆盖历史缓存：

```sh
cmake -S . -B build-contract4 -G Ninja -DBUILD_TESTING=ON -DopfusionEnableOperatorPlugins=ON
cmake --build build-contract4
ctest --test-dir build-contract4 --output-on-failure
build-contract4/apps/opfusion operators --json
build-contract4/apps/opfusion inspect benchmark/resnet/resnet50-layer1.json --json
```

## 既有清理与保护边界

此前已清理旧 modules/operators 和 standard 的 42 个生产文件，精确历史见 [LegacyCleanup.md](LegacyCleanup.md)。本轮未恢复旧包或改写该清单；插件发现仍默认 ON，不添加旧包名称排除。已有测试目标未删除或禁用。

本轮保存了实施前文件字节摘要、未跟踪自有源码副本、Git 状态及工作区/暂存区差异，用于核对全部改动及迁移两端，不只检查 tracked diff。治理/所有权、所有 AGENTS/CLAUDE、第三方、原始 benchmark/研究输入、fixture、既有 .gitignore/.clang-format/.clang-tidy 不属于写入范围；旧生成物不手工迁移。最终摘要与范围检查结果以本轮集成报告为准，不把历史保护统计挪作当前结果。

## 未验证与未完成

- clang-format / clang-tidy 的目标版本为 **18.1.8**，环境未安装、未运行；没有下载替代工具、修改配置或关闭检查，不声称工具检查通过。部分既有长行与整体风格仍可后续整理。
- TensorId/OperatorId/EdgeId、DeviceId 等仍为整数/字符串别名，没有编译期身份隔离；本轮未扩成全对象强 ID 迁移。
- 设备规则是显式建模声明，不是硬件探测/资源隔离证明；新增资源偏序上界是保守充分检查，不搜索或证明所有可行调度，可能要求补充资源顺序。
- InputsAvailable/OutputsDelivered 准确计时和虚拟汇时刻已在本次修正中收紧并通过上述回归。Stage 证据范围、Handoff 环境绑定、同值别名/复用规则、其他计时模式、全图数值属性传递仍未一并修复；测试通过不表示建模已无遗漏。
- 尚无 ONNX、完整 ModelPackage/Program、区域/候选/规划搜索、真实后端/Buffer/Run/Worker、硬件正确性/性能、观测真实性解析、自定义 packing 校验 provider、控制流、动态形状或跨区域流式执行。

更完整的字段与语义见 [CppModel.md](CppModel.md)，基础算子见 [Operators.md](Operators.md)。
