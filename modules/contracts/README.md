# contracts — C++ 公共领域模型

本模块提供 C++17 领域对象、22 个算子语义和纯不变量，只依赖标准库，不探测或执行硬件，不依赖 frontend/apps 或格式解析库。

公共 **contractVersion=6.1.0**，**operatorApiVersion=5.0.0**。算子身份仅有 namespace/name，删除语义版本与旧二维类型别名；Conv/MaxPool 按一至三维空间参数建模，MatMul 支持批维广播。具体类持有类型化属性，公共只读属性投影、实际广播和静态整数求值见 [算子语义](docs/Operators.md)。来源格式、插件包、后端与契约版本独立保留。

SourceBinding 可提供一对多 SourceNormalizer，返回真实算子、新张量及说明；Registry 仍只按 selector 查询。TimingPolicy 在部署时声明口径，TimingScope 在 Plan 中引用具体事件，两者不混同。Dense 表示即使 stride 未知，也不能声明比全部逻辑元素更小的 actualBytes；精确物理跨度仍由 stride/padding 校验。

当前验证与完整模型准备命令见 [根 README](../../README.md)。ND 数值索引回归仅是测试解释器，不是设备后端；模型导入不证明执行支持。历史 CodeQuality/LegacyCleanup 记录保留其当时背景。

## 按功能查找文件

公共头和实现同时归组，命名空间仍为 opfusion；没有旧平铺头转发层，调用方必须更新 include 并重新编译。

| 目录 | 公共入口与职责 | 实现 |
|---|---|---|
| common | [Error.h](include/opfusion/common/Error.h)、[Validation.h](include/opfusion/common/Validation.h)：错误和三态诊断 | [src/common](src/common/) |
| graph | [DAG.h](include/opfusion/graph/DAG.h)、[Operator.h](include/opfusion/graph/Operator.h)、[Tensor.h](include/opfusion/graph/Tensor.h)、Axis、IndexSemantics：逻辑图、端口、值及访问 | [src/graph](src/graph/) |
| registry | [Registry.h](include/opfusion/registry/Registry.h)、Source、BuiltinOperators：语义与来源绑定分开、显式装配冻结 | [src/registry](src/registry/) |
| platform | [Device.h](include/opfusion/platform/Device.h)、[Platform.h](include/opfusion/platform/Platform.h)、[Resource.h](include/opfusion/platform/Resource.h)、[Boundary.h](include/opfusion/platform/Boundary.h)：设备协议、唯一库存和部署条件 | [src/platform](src/platform/) |
| model | [Model.h](include/opfusion/model/Model.h)：区域/划分、表示、实现、候选、成本、交接与证据 | [src/model](src/model/) |
| plan | [Plan.h](include/opfusion/plan/Plan.h)：给定选择的事件、分配、绑定与纯校验 | [src/plan](src/plan/) |
| operators | [22 个具体类](include/opfusion/operators/)：类型化属性、input-only 推导、traits 与索引语义 | [src/operators](src/operators/) |

小型关联记录与枚举留在所属概念头中，不为每个 struct 机械建文件。自有头源、测试及 CMake 均有中文职责说明；公共 API 使用中文 ///，关键实现解释单位、未知状态、所有权、生命周期及保守结论，不逐行翻译代码。

## 设备父类与四层资源责任

Device 仅提供两个可替换行为：`requiredResources(platform, work)` 和 `clone()`；按值拥有 DeviceDescription，只允许只读访问。新增不同行为可直接继承 Device，不修改 Model/Plan 的型号名单，不增加 SDK 或执行职责。

- [CpuDevice](include/opfusion/platform/CpuDevice.h) 保存显式规则并检查 core 配额；分数配额不取整，slots 不默认为 cores。
- [DeclaredDevice](include/opfusion/platform/DeclaredDevice.h) 用声明规则表达 GPU/加速器等建模实例，不是硬件执行器。同规则新型号可复用该类。
- Platform 独占 `vector<unique_ptr<const Device>>`，拒绝空设备与重复身份；复制时逐个 clone，避免对象切片。派生规则必须按值拥有，不共享外部可变配置；const 本身不是线程安全保证。

| 信息 | 唯一责任方 |
|---|---|
| 资源身份、单位、物理容量、归属/共享、物理池 | Platform 的 Resource / MemoryPool 库存 |
| Compute / LocalConversion 必须声明的资源和最低量 | Device 规则；未知保留 nullopt / Unresolved |
| 某个完整候选/阶段的实际请求 | ExecutionConfig.resources / Stage.resources |
| 本次动态可用量、输入驻留、输出交付和计时 | Boundary |

工作区和服务时间仍是 Cost/Stage 的条件属性，不固化为设备常量。规则只验证，不向实际请求补资源，也不生成第二份资源活动。Candidate 和本地转换统一调用 validateExecutionResourceUses：已知要求被遗漏为 Invalid；Compute 必须是 Device 作用域，非空 Registers/Copy/Thread/Block 表不能替代。Transfer 仍覆盖显式 Link.resources，Identity/Synchronize 不统一强加 Compute。

CPU 的公共 quota/affinity/NUMA 不变量不依赖派生类型。同节点 affinity 有交集时，缺少共同显式共享 Compute 约束为 Unresolved；已有共享约束却未实际请求为 Invalid。这是最小共享核验，不是全部拓扑或并发隔离证明。

## PlanMemory 等文件是什么

[Plan.cpp](src/plan/Plan.cpp) 的 Plan::validate 负责阶段编排。以下均为同目录私有校验实现，不是同名内存管理类，不分配设备缓冲：

| 文件 | 实际职责 |
|---|---|
| PlanValidation.cpp/.h | 本次调用的借用索引、事件偏序、选择、分配、来源/交付及事件归属 |
| PlanTaskValidation.cpp | 区域任务、阶段端口、交接/复用及实际资源活动 |
| PlanMemoryValidation.cpp | 全部读者/传输存活区间、全任务工作区与既有别名规则 |
| PlanMemory.cpp/.h | 唯一 Allocation 的初始占用、偏序保守上界和给定时间表容量 |
| PlanResourceValidation.cpp | 给定时间表容量、成对排他顺序和任务峰值的偏序保守上界；超上界为未知，不搜索调度 |
| PlanConditions.cpp/.h | 精确数值组合的验收声明及计时覆盖 |

保持输出启动前分配、完整区域输出就绪、工作区全任务保留、全部读者/传输完成前不得释放及唯一 Allocation 计费。偏序保守上界超限为 Unresolved，不证明所有调度不可行。Plan 是给定记录的校验器，不是优化器或运行时。

## 注册与证据边界

Registration 只登记语义；SourceBinding 独立保存 semantic、selectors 和 factory。Registry::bind 要求目标先存在，拒绝冻结后修改和通配歧义；失败不发布部分绑定。frontend 提供来源工厂，contracts 不反向依赖 frontend。

封闭状态使用 enum class；AxisOrder 与物理 Layout 独立，开放设备型号/提供者不引入中央枚举。Validation 区分 Valid / Invalid / Unresolved。Simulation 不升级为 RealDevice；结构一致不等于候选进入生产设备域或计划完整 Valid。更新发布新快照，不原地改写已引用记录。

## 与早期规格的关系

以下规格保留共享语义及未来系统背景，旧 *Spec / *Record 名称不是当前 C++ 类型或已有序列化格式：

- [semantic-graph.md](semantic-graph.md)：逻辑图、值、端口与数值要求。
- [candidates-costs.md](candidates-costs.md)：区域、表示、候选、成本与证据。
- [execution-plan.md](execution-plan.md)：部署、交接、事件和内存。
- [backend-ports.md](backend-ports.md)：平台及未来编译/执行端口。
- [extensions.md](extensions.md)、[program-inputs.md](program-inputs.md)：扩展与入口规范。
- [worker-protocol.md](worker-protocol.md)：尚未实现的分布式任务与重试。

本轮不修改治理提示词或所有权策略。

结构轴模型与可选语义名称分开；公开 `buildAxisModels` 提供坐标和生产来源，不把缺少业务名称算作缺少坐标关系。详见 [C++ 对象](docs/CppModel.md)。
