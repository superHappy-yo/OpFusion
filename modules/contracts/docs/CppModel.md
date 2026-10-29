# C++ 领域与条件模型

本页描述当前 C++17 对象及纯校验实现，不是未来规格的完成声明。生产 contracts 只依赖标准库：不探测硬件，不执行设备工作，不搜索区域、候选或调度。公共 **contractVersion=6.1.0**，**operatorApiVersion=5.0.0**；算子身份不含语义版本，来源格式版本独立。公共头路径与 Device/Platform 所有权协议为显式破坏性变化，不保留旧路径转发头，调用方须重新编译。文末记录本轮实际回归，测试通过不构成硬件、数值或全部建模完备性的证明。

## 文件与分层

| 公共头 | 对象与责任 |
|---|---|
| [common/Validation.h](../include/opfusion/common/Validation.h)、[Error.h](../include/opfusion/common/Error.h) | 三态诊断与稳定错误边界 |
| [graph/Axis.h](../include/opfusion/graph/Axis.h)、[Tensor.h](../include/opfusion/graph/Tensor.h) | 逻辑轴、签名、值角色、参数内容引用、数值要求 |
| [graph/Operator.h](../include/opfusion/graph/Operator.h)、[IndexSemantics.h](../include/opfusion/graph/IndexSemantics.h) | 公共行为与索引访问；基础具体类位于 operators 子目录 |
| [registry/Registry.h](../include/opfusion/registry/Registry.h) | 独立的语义登记、来源绑定、通用匹配与冻结目录，不解析具体来源 schema |
| [graph/DAG.h](../include/opfusion/graph/DAG.h) | 只含逻辑算子的图、有序端口边、可选显式模型接口 |
| [platform/Device.h](../include/opfusion/platform/Device.h)、[Resource.h](../include/opfusion/platform/Resource.h)、[Platform.h](../include/opfusion/platform/Platform.h) | 纯设备资源协议、多态所有权及唯一物理库存 |
| [platform/Boundary.h](../include/opfusion/platform/Boundary.h) | 一次部署的驻留/交付/动态容量/计时条件 |
| [model/Model.h](../include/opfusion/model/Model.h) | 区域、划分、表示、实现、配置、候选、成本、交接、证据与组合数值验收 |
| [plan/Plan.h](../include/opfusion/plan/Plan.h) | 手工选择、任务/事件、物理分配与绑定、全区域工作区预留 |

时间为 `double` 微秒，物理字节/步长/偏移为 `Bytes`，padding 为每个逻辑轴的额外元素总数。`nullopt` 表示未知，不能改写成零；ID 的领域语义是不透明身份，不从名称推断设备或语义。

本次修正已有计时/资源语义的实现一致性，不增加公共字段、不改变单位，contractVersion 保持 4.0.0；构建内容身份随源码重算。本次已在全新临时目录完成完整构建和 6/6 CTest，模型校验为 390 项检查；三个旧实现对照均被相应新增回归拒绝。其余迁移验证数据先于本次改动；本次仍无硬件证据。

**既有类型债务：** 当前 TensorId/OperatorId/EdgeId 仍为整数 alias，RegionId/DeviceId/PoolId/CandidateId 等仍为字符串 alias，C++ 类型系统尚不能隔离不同身份。本轮未扩大为全对象强类型 ID 迁移，不能宣称已满足新 C++ 规范的强类型身份要求；后续需连同公开接口、序列化边界及测试统一迁移。

`Validation` 的 Valid / Invalid / Unresolved 分别表示在显式假设下通过、已知违反、必要信息不足。两类问题并存时状态为 Invalid，问题列表仍保留。构造的 DAG/Platform/Region/Partition/Model 拒绝已知结构错误，但 Model 可以保留未知探索记录。Plan 接受手工记录并通过 `validate()` 返回问题。

## 逻辑图、值与统一 Operator

Tensor 保存 shape、dtype、可未知的 `LogicalAxes`、`ValueKind` 和可选 `ParameterRef`，不携物理 stride、设备或 Allocation。角色为 Unknown / Input / Parameter / OperatorResult；参数引用给出 artifactId/member/contentDigest，不内嵌权重。参数内容引用缺失时不能据角色断言已具备执行数据。

`DAG` 可接收有序 `modelInputs()` 与完整 `modelOutputs()`。输入必须唯一、无 producer 且不是参数；已声明的 Input 不能遗漏于显式接口。真实 producer 结果收敛为 OperatorResult，参数或输入不能同时被节点生产。只有来源未声明角色时才保持 Unknown；不能按名字或广播形状猜权重。外部依赖和 terminal outputs 是推断视图，不替代显式完整接口。

`AxisKind` 表达 Batch/Channel/Height/Width/OutputChannel/InputChannel/Row/Column 等逻辑含义，也可未知；`Axis.label` 保留非空的轴身份约束。`AxisOrigin` 区分 Declared/Derived/Mixed/Unknown。已知轴冲突拒绝，不静默转置 shape。DAG 在发布前经公共访问关系做单调收敛，不知道具体类名。

逻辑 `AxisOrder` 与物理 `Layout` 是两个 enum class。前者用于算子的 NCHW/OIHW 等逻辑约定；后者属于候选存储表示。来源 strides 不能自动提供这两者的证明。

Operator 公共协议：

- 非虚 `validate(inputs)` 统一核对签名数量、静态维度、轴、dtype/数值要求，再调用受保护的 `validateAttributes()`。
- `infer(inputs)` 仅据输入及具体属性推导输出；声明输出只供 DAG 事后比较。
- `traits()` 保留条件性与未知，`indexSemantics(inputs)` 提供迭代、端口访问和归约关系。
- 父类构造函数不调用虚钩子，不解析 JSON、不保存 Registry、不执行设备。

索引关系是半开域及可选取模、间接整数 lookup 的受检表达式，迭代器非负、除数为正。支持广播、Concat 分段、Resize floor-div、Conv/Pool 窗口与越界填充值；不是通用编译器 IR。输出在非归约域上解释，Sum 初值为零、Maximum 初值为浮点负无穷，因此 K=0 的 MatMul 仍写出零和结果。Unknown/Unsupported 必须有原因且不提供伪造的部分关系。访问关系校验不等于证明任意新插件的数学正确性或融合合法性。

具体算子和“增加同名头/源即可登记”的构建约定见本模块的 [算子语义](Operators.md)。十个基础类位于 `include/opfusion/operators` 与 `src/operators`，不是独立业务模块或默认扩展包。实现统一行为不自动提供未知来源格式、数值 kernel 或设备能力。

### 语义登记与来源绑定

API 3 的 `Registration` 只保存 `SemanticIdentity`；独立 `SourceBinding` 保存目标语义、`SourceSelector` 集合与 `OperatorFactory`。`Registry::add` 先登记语义，`bind` 只绑定已登记语义，并检查冻结状态、完整身份、非空工厂/选择器及绑定内部和跨绑定的歧义。未指定 operation 的选择器匹配该 format/type 的全部 operation，因此不能与同范围内的具体 operation 绑定重叠。

`create` 要求冻结目录，只做通用选择及返回对象的非空/语义身份检查，不解析具体来源字段。`registrations()` 与 `sourceBindings()` 分别提供只读查询，不重复维护同一份选择器或工厂。工厂必须同步消费借用的 SourceNode，不得保存其引用；冻结 Registry 不自动冻结工厂捕获的外部可变状态。

[内置语义入口](../include/opfusion/registry/BuiltinOperators.h) 由 contracts 提供；[内置来源入口](../../frontend/include/opfusion/frontend/BuiltinSourceBindings.h) 与 JSON/ONNX 工厂归 frontend。apps 显式按“语义 → 来源绑定 → 可选插件 → 冻结”装配；纯领域类不包含 Registry、来源 schema 或来源字段解析。`contractVersion=6.1.0`，`operatorApiVersion=5.0.0`，算子语义不再版本化；registry/cpp-static/4 是当前身份格式标记，内容摘要随本轮源码重新生成，不机械替换数字。

## 平台、实现与环境

DeviceDescription 为纯值描述，`DeviceKind` 分类为 Unknown/Cpu/Gpu/Accelerator/Other；架构、型号、节点、实例、配置及 backend/provider 身份保持开放。Device 父类按值拥有描述，仅提供 `requiredResources(platform, work)` 和 `clone()`，不加入 SDK、执行句柄或发现器。CpuDevice 与 DeclaredDevice 直接派生；同规则新型号复用声明类，新行为直接继承 Device，不修改 Model/Plan 名称分支。

Platform 接管 `vector<unique_ptr<const Device>>`，拒绝 null 和重复 ID，查找返回借用的 const Device&。复制时逐个 clone，检查结果非空且身份未改，避免切片；赋值禁止。派生类必须独立保存不可变规则，不捕获外部可变配置，const 本身不提供深度不可变或线程安全保证。Model 继续共享 shared_ptr<const Platform>，候选/阶段仍按设备 ID 引用。

资源分四层：Platform 唯一库存保存 Resource/MemoryPool 的单位、容量、归属和共享；Device 只保存最低资源 ID/量；ExecutionConfig.resources / Stage.resources 是具体实际请求；Boundary 是本次动态限额及 IO/计时。工作区和服务时间仍来自 Cost/Stage。`minimumUses=nullopt` 表示规则未知；已知空规则与未知不同，不能代表有效 Compute。规则只验证，不自动补请求，不重复产生 Plan 活动。

Candidate 使用 DeviceWork::Compute，本地 Convert/Repack/Quantize/Dequantize 使用 LocalConversion，经统一 validateExecutionResourceUses 核验。实际量须覆盖已知最低量，且双方必须含 Device 作用域 Compute；Registers/Copy/Thread/Block 不能靠非空表绕过。未知单位/容量/归属保持 Unresolved，已知缺漏、非法量、错归属或超容量为 Invalid。Transfer 保留 Link.resources，Identity/Synchronize 不统一强加 Compute。

CPU 用 CpuAllocation 显式提供 quotaCores、affinity、NUMA 适用性与节点。分数配额不取整，slots 不默认为 cores；只有规范库存明确 unit=cores 才比较请求与配额。已知零配额/空 affinity 不能执行，公共描述和执行核验对直接 Device 子类也生效。同 nodeId 的 affinity 交集缺少共同显式共享 Compute 约束时为 cpu_sharing_unknown / Unresolved；已有共享约束未请求时为 cpu_shared_resource_missing / Invalid。这只防止显式重叠实例被当作独占，不证明全部 CPU 拓扑或并发隔离。

BackendEnvironment 提供 backend/version/ABI/runtimeVersion、driver 适用性及版本。Implementation 提供 registry、精确 provider id/version/digest、后端版本、制品引用和所需环境。validateExecution 核对所选设备、后端、ABI/runtime/driver；缺失为 Unresolved，不探测设备补默认值。当前按显式版本精确匹配，不声称已有复杂兼容范围解析器或制品加载器。

共享物理池只能用一个 ID，物理容量与动态 available 容量分离。Resource 指定单位、作用域、容量、独占 device 或显式 sharedDevices/sharingRef，也可关联 pool；设备请求必须满足所有权和 pool 可访问性。Cost.scopedResources 仍独立检查且不能重复配置请求；最低规则不复制库存或动态限额。

## 冻结快照

先构造 DAG、Platform、Boundary，再用 Builder 填表并发布 const Model/Plan：

```cpp
opfusion::Model::Builder records;
records.id = "snapshot-2";
records.dag = frozenDag;
records.platform = frozenPlatform;
records.boundary = explicitBoundary;
records.numericPolicy = explicitNumericPolicy;
// 显式填充区域、划分、表示、实现、候选、成本、交接和证据。
auto model = std::make_shared<const opfusion::Model>(std::move(records));

opfusion::Plan::Builder supplied;
supplied.id = "manual-plan-1";
supplied.model = model;
// 显式填充 partition、selections、tasks、events、allocations、bindings。
const opfusion::Plan plan(std::move(supplied));
const auto result = plan.validate();
```

访问器返回 const 引用，发布对象不支持赋值覆盖；新结果创建新身份/快照。调用方仍须放弃外部可变别名并保证 ID/内容持久化；`shared_ptr<const T>` 不是内存沙箱，也不会自动建立跨进程版本存储。

## 区域、完整候选与条件成本

Region 从原始有序端口关系派生稳定、按值去重的边界，保留所有外部消费者需要的结果和全部模型输出。原图 Add(U,U) 的两个使用不丢失，区域接口可以只传一次 U。模型输出未知时 terminal fallback 不是完整输出证明，Model/Plan 返回 `model_outputs_unknown`；模型输入身份/顺序未定时返回 `model_inputs_unknown`。

Partition 检查完整覆盖、互斥、成员与图身份、商图无环。原图无环不代表任意区域收缩都无环。

`candidatesFor(region)` 保留全部登记项，`eligibleCandidatesFor(region)` 只接受 Supported、Verified 且所选依赖校验完整的候选；`deviceDomain(region)` 从该集合派生。完整 q 包含 device、implementation、execution config 与按端口表示，不从不同候选拼时延/工作区。无候选、Unknown、BudgetExhausted 不证明全局无解。

候选校验传播所选设备、实现提供者、后端环境、配置、实际资源、参数内容和边界表示的必要条件。计划不会把无关探索项的 Unresolved 全局搬到选中路径，但已知结构错误仍拒绝，所选 Candidate/Cost/Handoff/Stage 各自重新核对。

Cost 必须精确绑定 candidate/platform/boundary/config。服务时间从输入已驻留就绪到完整区域输出就绪，包含内部调用/转换，排除排队与区域外交接。每个输出 ready offset 首版等于服务时间；计划不混用不同统计口径。

Analytical/Predicted/Measured 来源独立于执行验证；measured 需要观测引用，其他来源不得伪填实测样本。`workspaceBytesByPool` 是所选 Cost 的条件属性，完整空表才表示已知零工作区；未知不是零。引用未由实际观测服务解析核验，不因此成为实机证据。

## 物理表示与数值政策

Representation 独立保存存储 dtype、Layout、字节 strides、轴元素 padding、alignment、PackingKind、quantparams、actualBytes、pool 和 backend context；没有 SDK 句柄。

- Layout 的 RowMajor/ColumnMajor 按逻辑轴位置解释；NCHW/NHWC/OIHW/HWIO 通过逻辑 AxisKind 找物理排列。命名布局缺轴信息为 Unresolved，已知冲突为 Invalid；不重排逻辑 shape。
- `PackingKind::Dense` 的 `requiredBytes()` 以 dtype/stride/padding 计算防溢出的物理跨度，检查元素重叠与声明顺序。actualBytes 必须覆盖跨度，可以大于逻辑字节数；真实空张量可以为零。
- `PackingKind::Custom` 需要开放 packingFormat、validationRef 和实际大小。当前没有自定义格式校验 provider，即使给出引用仍为 `packing_check_unavailable`，不猜压缩比或伪装已证明兼容。
- 未量化也要显式 `QuantizationKind::None`；per-tensor/per-axis 校验 scale、zero point、轴、数量和整数范围。缺参数是未知。

NumericalRequirements 包含 allowedTypes、computeType、accumulationType、allowApproximation 和误差界。ExecutionConfig 的 Arithmetic 声明计算类型、累加类型及 Strict/ReducedPrecision。显式算子要求不能被配置覆盖，归约的累加类型缺失为未知；未知 reduction traits 不能证明累加政策已满足。

改变存储 dtype、**同 dtype 量化**、计算/累加模式变化均须核对局部与全局政策。近似的正式选择还需要 NumericAcceptance：绑定准确 graph/platform/boundary/policy/partition，以及完整 candidate/handoff/boundaryStage 组合、方法、用例/数据范围、观测引用和误差。任意非空 validationRefs 不算验收；Plan 要求集合精确匹配实际选择。

局部 Verified 候选/交接与近似整图验收要求 Passed + RealDevice 的适用记录。Simulation/Analytical 仍可记录，但不能升级为正式执行证据；本模块没有运行这些测量或核验外部观测真实性。整图误差不能证明更严格的逐算子误差；缺独立局部观测模型时保守为 `operator_numeric_evidence_unknown`。

## 适配器与边界交付

Handoff 绑定原始 EdgeId 和准确两端 CandidateID。每个非 Identity Stage 需要 implementation/executionDevice，复用实现/环境/制品条件校验；缺失时仍为 Unresolved，路径和时间不冒充可执行提供者。

阶段端点必须与所选候选匹配，中间引用连续。raw Transfer 使用正确方向 Link、真实表示字节和共享资源，不暗中改变 dtype/布局/量化；本地 Convert/Repack/Quantize/Dequantize 不隐藏跨池移动。Identity 要求同物理表示、backend context、allocation/offset，零时间仍有依赖。Synchronize 是显式有时延的无复制阶段。

`Handoff::serviceTimeUs()` 只求阶段服务时间和，未知阶段则整体未知。当前阶段连续串行，排队只能在整项任务启动前表达；不支持阶段间等待资源，也不能把等待塞入固定服务时间。边界 IO 使用 Model.boundaryStages 和 BoundaryStage task；原始 DAG 不添加设备、Source/Sink 或复制节点。

Boundary 明确输入/参数驻留、完整输出目标、动态容量与 TimingScope。计时起止是类型化语义及真实事件引用，校验应覆盖的区域/交接/边界适配，以及实际输入可用或全部输出交付。BeforeInitialization 或 includesInitialization 目前因无初始化任务而 Unresolved，不能默认初始化成本为零。

InputsAvailable 必须准确等于全部声明 SourceReady 的最晚时刻，不能跳过输入可用后的等待；无来源时为 0。OutputsDelivered 必须等于全部要求 SinkComplete 的最晚时刻，无输出时等于起点。SinkComplete 是零成本虚拟汇，时刻等于目标绑定 ready，额外交付须由显式任务承担。准确时刻不能替代已有的因果依赖检查；任一时间未知时继续保留未知，不取部分 max 值。

## 事件、复用与容量

Plan 每个选中区域恰好一个候选/成本与 Region task，跨区域依赖必须有精确两端的已验证 Handoff。事件校验 kind/owner/引用/无环/顺序/预测时间，包括 SourceReady、Allocate、TaskStart/Complete、SinkComplete、Release 及可选专用 TimingStart/Finish。时间戳不替代事件依赖，预测时间不是跨主机执行命令。

1. 输入绑定有真实 ready 事件；输出到达指定位置/表示后 Sink 才完成，不以最后一个设备 kernel 结束代替交付。
2. 边界输出、私有中间缓冲和工作区在任务开始前分配。Workspace 首版全区域/适配任务保守预留，引用实际 Allocation，不重复计费。
3. release 等待全部读取、传输、同步与最终交付。字节区间保存**全部**使用开始/完成事件；不同值/工作区重叠时，前者全部完成必须先于后者全部开始，不能按预测时间挑一个 last reader。等价同值、同表示、同位置别名可保留。
4. 容量仅按唯一 Allocation 计费。每个请求独立检查 min(物理容量,动态可用)，零时长 allocate/release 也不能跳过大小检查；初始全部 preexisting 分配同时计入。
5. 每个分配事件求偏序保守占用上界：除非证明已释放或尚未分配，其他分配都计入。上界超限为 `pool_capacity_order_unproven` / Unresolved，不是所有调度都不可行的证明；不穷举或搜索时间表。
6. 另检查给定完整预测时间表，确切超额为 Invalid。不能仅凭该表不重叠就通过偏序容量条件。
7. 资源核对所选请求的物理/动态容量，检查预测时间冲突及排他复用顺序，并对每个任务启动切点求累计资源的偏序保守上界。同一任务连续阶段取局部峰值，不能重复累加；其他可能存活任务的峰值合计超限时为 resource_capacity_order_unproven / Unresolved，不宣称所有时间表不可行。线程/块作用域只作局部约束，不声称有实测干扰模型。

`Task.reuse` 只能引用已完成交接的等价物理副本，是 dependency-only join，不重执行或重复收费；仍核对 producer、值、表示、实际位置与就绪依赖。不能从未完成副本推导零成本通信。

`Plan::validate()` 只编排阶段：模型/选择 → 事件偏序 → 分配/绑定/Source → 任务 → 精确交接 → Sink → 工作区/别名 → 资源/容量 → 事件所有权 → 数值/计时 → 派生任务图。事件有环时停止后续依赖偏序的分析。TaskNode 的默认 kind 为显式 Unknown，校验拒绝未知或非法枚举；Edge、TaskArc 等公开记录的整数成员有确定初值，但零值不冒充有效对象身份。

内部文件按职责拆分：PlanValidation 处理索引、事件及边界；PlanTaskValidation 处理区域任务、适配阶段与复用交接；PlanMemoryValidation 保存全部使用事件并检查工作区/别名；PlanResourceValidation 检查给定请求、排他资源顺序与累计资源保守上界。PlanIndex/EventOrder/PlanUsage 仅借用冻结记录或保存本次校验的派生结果，不写回计划，不引入公共管理器或全局状态。

容量、数值及计时继续复用 PlanMemory、PlanConditions；实现/表示检查仍由 ExecutionConditions、NumericalValidation、RepresentationChecks 提供。这些文件不新增对其他业务模块的依赖，也不执行设备或优化器工作。

## 研究字段对应

| 研究/旧草案字段 | 当前对象 |
|---|---|
| Spec_v / OperatorSpec | DAG 中的具体 Operator、输入输出 Tensor、requirements 与 indexSemantics |
| processors_v / D_v | `Model::deviceDomain(region)`，从完整已验证候选派生 |
| C_v / (d,k,theta) | Candidate + Implementation + ExecutionConfig，原登记池另行保留 |
| rho_in / rho_out | Candidate 的按端口 Representation，完整区域输出可见 |
| t_v(q) | Cost.serviceTimeUs、统计/来源/条件，排队独立 |
| Work_v(q) / Cache_v | Cost.workspaceBytesByPool 与 Plan.Workspace，依候选且分池 |
| data_size_e | Representation.actualBytes / Stage.transferBytes，不固定为逻辑 dtype 字节 |
| p_e / AdapterSpec | Handoff + 有序 Stage，绑定精确 Edge 与两端 q |
| HardwareProfile / DeploymentContext | Platform / Boundary，物理与动态容量分离 |
| ExecutionPlan / BufferSpec | Plan / Allocation + TensorBinding，不是 runtime Buffer |
| 虚拟 Source / Sink | 派生 TaskGraph + BoundaryBinding，不污染原始算子 DAG |

这些旧名称不是已有序列化记录；本页不宣称有兼容迁移器。

## 验证与未实现部分

本轮已获用户授权，在两个独立新目录分别记录基线与迁移后结果。基线配置成功、旧 CoreTest/ModelTest 编译失败，未运行基线测试；迁移后保持 BUILD_TESTING=ON / opfusionEnableOperatorPlugins=ON，以 GCC 11.4.0 / C++17 完成配置及完整构建，**6/6 CTest 目标通过**，CLI 黑盒含 15 个测试。逗号小数 C locale 子用例因环境缺少 locale 明确 SKIP，不计为该 locale 的覆盖。

contracts/frontend 共 **30 个公开头**分别通过独立 C++17 语法检查。新 CLI 核对十语义、九绑定及 MatMul 无绑定；八图只读导入共 **264 节点、415 张量、301 条完整有序端口边**。端口重新派生的边多重集与输出完全一致，执行仍 not_established，搜索 not_searched，成本未知。完整结果见 [整改状态](CodeQuality.md)；clang-format/tidy 未安装、未运行，没有硬件验证。

旧测试已迁移而非删除/禁用；standard 的来源用例归 frontend，纯语义目标仍仅链接 contracts。模拟始终保留 Simulation，结构测试只允许明确的未知原因，并独立断言不能产生生产候选/设备域或完整 Valid Plan。旧生产包未恢复，历史清单见 [清理记录](LegacyCleanup.md)。

本次新增输入就绪后的等待、非零来源时刻、多个输出目标、延后计时终点/虚拟汇、未知来源时间，以及两两可容纳但合计超额、分数请求和任务内部连续阶段的回归，已随模型目标通过。Stage/Handoff 证据范围、同值别名/复用、全图数值属性传递及其他计时模式仍不属于全面修复声明；测试通过不证明所有建模约束闭合。

仍未实现：完整 ModelPackage/来源摘要、ONNX、Program/编译器、区域/候选/调度搜索、真实后端/Buffer/Run/Worker、硬件正确性和性能测量、原始 MeasurementRecord 存储/解析/真实性核验、局部数值观测模型、自定义 packing 校验 provider、任意后端 options schema、稳定序列化/快照持久化、原地写入、控制流、动态形状、跨区域流式输出、阶段内排队与精细内部 workspace 时分复用。当前 Valid 仅是在显式条件与引用成立前提下的保守模型结论。


结构轴公共查询见 [AxisModel.h](../include/opfusion/graph/AxisModel.h)。`buildAxisModels(DAG)` 从静态签名和已验证索引关系构造坐标范围及逐 piece 来源，记录直接、拆分、合并、重索引、间接读取、元数据和生成关系；不按算子或张量名称分派。`TensorAxisModel::validate` 核对图/registry、端口、维度、表达式与结构来源，拒绝篡改。未知提供者的索引关系仍保持 Unknown，不以签名齐全冒充关系已知。标量无轴，命名轴为独立可选注释，既有 Tensor 命名约束不变。新增接口令 contract 升至 6.1.0，operator API 保持 5.0.0；CLI inspection/4 改变轴输出格式，调用方需要迁移。
