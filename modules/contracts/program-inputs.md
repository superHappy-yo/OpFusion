# 模型入口、程序方案与验证政策

状态：0.2.0 草案，未实现。本文定义入口对象与跨编译器边界，不提供现成 API、序列化库或运行命令。

## 模型与任务入口

| 对象 | 必需内容 | 生产与消费 |
|---|---|---|
| ModelPackage | package_id、格式/版本、模型与权重的内容摘要和制品引用、输入输出签名、源算子域/opset、导出器版本/选项、来源记录 | apps 校验后交 frontend |
| OptimizationRequest | request_id、package_id、shape 特化条件、numeric_policy、优化目标、搜索/编译/测量预算、验证政策引用、部署环境引用 | apps 分解为各研究阶段请求 |
| RunRequest | request_id、冻结的 plan_id、实际输入绑定、输入/权重驻留、输出目标和运行限制 | apps 校验后交 runtime |

输入签名声明名称、端口顺序、dtype、逻辑轴与形状；权重引用不等于模型运行时输入。DeploymentContext 的定义以 [执行计划契约](execution-plan.md) 为准，输入驻留/输出目标/容量等必须和 RunRequest 一致，不能在执行时悄悄改写计划前提。

ONNX 为第一模型格式；torch.export、StableHLO 作为以后按需求增加的前端。首版只接受经验证可规范化为静态形状、纯值语义、无控制流 DAG 的子集。框架导出成功不等于本系统支持，符号维度也不等于成本模型已获得实际维度。未知算子、语义版本、状态或控制流返回结构化诊断。

frontend 依据扩展提供的格式映射生成 GraphSpec，并保留 SourceMap：源节点/端口、域/opset、导出模型摘要、规范化规则及版本、目标节点/值的多对多对应。源 Conv 带 bias 时，按明确政策拆为 Conv 与 BiasAdd 或保留支持的规范语义；不能重复加 bias，不能假定导出器没有做优化。

验证/校准/测试数据作为独立引用和用途登记；不能将测试标签隐含放入优化请求。预处理、tokenizer、生成循环及 KV cache 管理不因模型格式支持而自动纳入；需要时明确其图接口和状态政策。

## ProgramSpec

ProgramSpec 描述 codegen 选出的一个具体程序方案，不等于已编译、可执行或已测性能的实现。

| 字段 | 含义 |
|---|---|
| program_id、schema_version | 不可变方案标识与结构版本 |
| graph_id、region_id、interface_id | 语义和必须保持的边界；引用须来自同一冻结上下文 |
| registry_snapshot_id、generator_id/version | 确切扩展解析结果与生成规则来源 |
| representation_kind/version、source_ref | 源表示类型/版本和内容摘要引用；例如某锁定版本的张量 IR、Triton 源程序、库调用描述 |
| compiler_id/version、target_constraints | 可接受的编译提供者及目标架构、数值模式、ABI 条件 |
| structure_descriptor、options | 已选择的 tile、物化、内核分解等摘要，以及有版本 schema 的后端选项；未知项明确标记 |
| numeric_policy、boundary_signature | 保持原接口的数值、输出集合、表示和所有权政策 |

contracts 不承载活的 MLIR context、TVM IRModule、Triton AST、Python Tensor 或设备句柄。后端内部对象通过适配器构造；跨进程使用可校验制品引用。结构参数不强制所有架构共用一套 tile/线程字段，后端特有选项必须有命名空间、版本和验证，不能改变未声明的公共语义。

首版 ProgramSpec 覆盖一个完整固定区域；独立实现扩展生成该区域的完整可序列化方案，或先通过合法分解复用已有生成器。不承诺任意插件的单算子 lowering 可以拼接成融合内核；细粒度 IR 片段的组合协议尚未定义。

## BackendCompileRequest 与编译结果

BackendCompileRequest 包含 request_id、program_id/方案引用、profile_id、registry_snapshot_id、已解析 compiler provider、编译选项、目标运行 ABI、工作区约束、编译预算与缓存条件。与外部 OptimizationRequest 是不同层次的请求。

编译结果返回 ArtifactRef、编译日志引用、工具链版本、资源估计及来源、实际内核/调用结构（若可观测）、接口核查结果和预算消耗。编译成功不自动升级为 verified；无法报告的内部资源须为 unknown，后续按保守测量政策处理。

缓存键至少覆盖源内容、语义/接口、生成规则和编译器版本、目标架构、选项、数值政策、ABI 与相关插件内容身份。禁止仅以算子名或设备名称复用制品。缓存命中记录来源，不伪造本轮编译时间。

## ValidationPolicy 与 MeasurementProtocol

| 对象 | 内容 |
|---|---|
| ValidationPolicy | policy_id/version、oracle_ref、输入域/测试用例引用、比较方法和数值容差、特殊值政策、接口/资源检查、证据用途 |
| MeasurementProtocol | protocol_id/version、计时边界/单位/时钟域、热身和重复、同步、驻留、线程/并发、缓存处理、失败样本政策、测量预算和 data_role |

oracle_ref 指向固定版本的语义参考提供者或已验证参考程序；与需要优化的 seed_candidate_id 不同。参考路径的实现、测试覆盖与限制必须可追溯，不能用新实现自身的输出作为唯一正确性依据。测试通过只说明声明范围内通过。生产与调优流程中的参考提供者必须通过 C++ 可用的端口执行；Python 独立参考只用于测试，不能因这里需要 oracle 就被引入生产调用链。

apps 提供生产/调优任务的明确政策；experiments 可以构造独立评价政策，但生产模块不依赖 experiments。runtime 的受控执行服务实现 MeasurePort，复用正式执行的 run/task/attempt、缓冲、取消和同步生命周期；backends/扩展只提供设备操作与计时原语。单机测量可以使用本地 runtime，不要求启动远程服务。

MeasurementRecord、EvidenceRecord、CostRecord 的唯一定义见 [候选与成本契约](candidates-costs.md)。测量成功、正确性通过和预测可用分别记录，任何一项都不能替代另一项。

## 未来验收

- 同一 ONNX 导出文件的规范化图、版本和来源可复现；Conv 的 bias 不重复，Add 的广播轴明确。
- 模型参数、真实输入和校准数据不混淆；动态形状和未支持控制流有明确诊断。
- 两个后端的程序方案通过相同区域/接口约束，但保留不同的内部表示与参数。
- 更换 compiler、target、插件内容、数值政策或接口后不能误命中旧产物缓存。
- pending 实现能进入受控验证，生产计划拒绝 pending；测量不另建一套无身份的执行路径。
