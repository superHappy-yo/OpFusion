# 硬件描述与公共端口契约

状态：0.2.0 草案，未实现。端口是行为边界，不是已经存在的 Python 类、RPC 或命令。实现语言映射应保留字段含义和失败语义。

## HardwareProfile

`HardwareProfile` 是不可变硬件/软件能力快照，包含 `profile_id`、`registry_snapshot_id`、设备表、有向互联、后端绑定及采集条件。apps 显式启动经 backends 组织的探测，由对应设备扩展执行；contracts 只定义结果。

| 内容 | 必需描述 |
|---|---|
| DeviceSpec | device_id、node_id、type、arch，以及实例和运行配置标识 |
| ComputeEngine | 引擎类别、运算/数值模式、吞吐及测得/标称来源 |
| MemoryPool | pool_id、physical_capacity_bytes、管理方式、可访问设备；共享物理池采用同一 ID |
| ScopedResource | 寄存器、显式片上存储、线程等作用域和限制；缓存不自动等价于硬容量约束 |
| ExecutionCapability | 计算/复制资源、可并发条件、完整输出事件能力 |
| LinkSpec | 有向端点、共享资源、路径约束、带宽及启动时延来源 |
| BackendBinding | backend_id、编译/执行/探测提供者的精确 ID/版本/摘要、后端/库/驱动版本、关联设备和产物 ABI |

动态 available_capacity_bytes 不覆盖物理容量，进入 DeploymentContext。CPU/GPU/NPU/FPGA 使用有类型扩展字段；未知、不适用和真实零值必须区分。新设备不能仅增加一个类别标签后就宣称可运行。

## 公共端口与职责

| 端口 | 输入 → 输出 | 典型实现模块 |
|---|---|---|
| CapabilityPort | 设备/后端请求 → HardwareProfile；区域+接口+配置 → 支持判断与原因 | backends |
| ImplementationCatalogPort | 区域+设备+接口 → 已登记实现描述及适用条件 | backends；apps 汇总新登记项 |
| CompilePort | BackendCompileRequest（引用 ProgramSpec）→ 制品、编译日志、资源报告或失败原因 | backends 路由到编译扩展 |
| ExecutePort | 通过摘要/架构/接口等加载前检查的制品、缓冲绑定、执行目的和上下文 → 后端操作句柄、完成/错误事件 | backends |
| MeasurePort | 候选、ValidationPolicy 与 MeasurementProtocol → 成本样本、资源和正确性证据 | runtime 受控测量服务；backends/扩展提供计时与设备原语 |
| PredictPort | 候选描述、特征、profile 与校准快照 → 带来源和适用域的预测 CostRecord | cost_model |
| ArtifactStorePort | 内容/引用 → 可校验 ArtifactRef；引用 → 取得、验证或缺失原因 | runtime 的存储适配 |
| SnapshotStorePort | 完整记录集合 → 新不可变快照；snapshot_id → 一致读取 | runtime 的存储适配，由 apps 注入 |

核心算法只接收契约对象和注入的端口，不自行构造具体设备驱动、runtime 服务或数据库客户端。apps 是装配具体实现的入口；端口返回数据，不回调导入 fusion/planner/codegen 形成业务依赖环。

具体算子、设备、编译器和实现通过 [扩展契约](extensions.md) 提供。backends 保留通用编译/执行路由，不为每个扩展新增中心注册项；实际工厂由 apps 加载并注入。ProgramSpec、BackendCompileRequest、验证与测量政策见 [入口与程序契约](program-inputs.md)。

CompilePort 负责把给定程序方案翻译/编译到后端。codegen 负责决定和搜索程序方案，例如 tiling、物化和内核分解；不能把整个结构搜索藏进一个无预算的后端调用。已有库程序可以直接登记，不能因此伪造一次编译。

ExecutePort 提供设备级执行原语；runtime 拥有任务状态、请求幂等、资源生命周期、取消和结果归属。设备后端不能自行跨 worker 重规划或启动新的运行尝试。

MeasurePort 必须显式取得测量预算和输入/数值验证政策，记录实际重复值、计时边界与版本。预测器不能通过隐蔽实测获取测试标签；编译与测量消耗均应反馈预算统计。测试与正式运行的操作入口可复用，但证据用途必须区分。

MeasurePort 复用 runtime 的 run/task/attempt、缓冲、取消与同步生命周期，不在后端另建不受管理的测量执行器。apps 提供生产/调优政策，experiments 仅提供独立评价政策和分析；本地验证无需远程服务。

首次正确性验证允许 MeasurePort 在受控验证任务中执行 `execution_status=pending` 的候选；执行目的显式标为 `validation`，仍检查制品内容、架构、接口和资源。只有取得要求的证据后才能登记 verified。正式 ExecutionPlan 的执行目的为 `production`，只接受 verified 候选/适配器。这避免“必须先执行才能验证、必须已验证才能执行”的启动循环；加载前检查不等同于候选数值验证。

ArtifactStorePort 和 SnapshotStorePort 的本地实现可先于分布式服务落地；不要求每次离线建模启动远程 runtime。不可变发布应验证引用完整性，不能让 planner 读取到半写入快照。具体持久化后端由实现 ADR 决定，不把数据库类型泄漏到算法接口。

## 失败结果与验证边界

所有端口返回结构化成功/失败结果，不以负时延、零工作区、空制品伪装失败。能力查询的 `supported` 只表明规则允许尝试，只有实际验证证据才能使 CandidateSpec 进入 verified。

需要明确的原因包括：版本/架构不匹配、未支持算子或表示、资源不足、编译失败、数值验证失败、预算耗尽、制品缺失/摘要不符、执行失败和状态未知。允许重试应附条件，不能把所有错误统一自动重试。

首次实现应提供端口一致性测试和至少一个真实可用后端验证；mock 可用于协议测试，但必须标为模拟，不能登记为真实设备性能证据。
