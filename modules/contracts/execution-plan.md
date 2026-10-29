# 部署环境与执行计划契约

状态：0.2.0 是早期规格版本，当前 C++ 领域记录与纯校验见 [CppModel.md](docs/CppModel.md)，公共 contractVersion 为 4.0.0；planner 和 runtime 尚未实现。planner 产生计划，runtime 执行已验证计划；runtime 不在执行中静默改变区域划分或设备放置。

## DeploymentContext

| 字段 | 含义 |
|---|---|
| deployment_context_id、profile_id | 环境快照及匹配硬件/后端描述 |
| initial_residency | 输入/参数已有的 tensor_id、设备或内存池、表示、数据引用和可用状态 |
| output_requirements | 模型输出必须到达的位置与表示 |
| available_capacity_bytes_by_pool | 本次计划可用容量，不等于设备物理总容量 |
| execution_limits | CPU 线程、资源保留、计算/复制并发及共享资源约束 |
| timing_scope | 计时起止边界，是否包含输入上传、输出回传和初始化 |

模型图与局部成本不能推断这些条件，调用方必须显式提供。持久参数已经驻留只免去重复上传，不免除内存占用。首版固定资源保留；资源环境变化时应重新验证计划，不能将未知竞争当作精确模型。

## AdapterSpec 与连接条件

`AdapterSpec` 包含 `adapter_id`、源/目标设备或可访问内存池、源/目标 RepresentationSpec、路径及共享链路资源、后端实现/制品引用、数值语义、服务时延来源、工作区和事件要求；还须含 `execution_status`、`evidence_refs`、`profile_id`、backend 版本及验证适用条件。状态采用 pending/verified/failed；未验证规则可用于探索，但正式计划的 Join 见证只接受符合条件的 verified 适配器。适配器不是 region 候选，不能无说明地复用 CandidateSpec。

`Join(producer_candidate, adapter, consumer_candidate, tensor_id)` 只有在三者的表示、设备访问性、数值和接口都匹配时成立。恒等适配器可以没有数据复制，但仍保留依赖同步。已在区域内部完成的转换不得在适配器中重复计费。

相邻兼容不证明全图选择一致；planner 必须选择同一组候选和适配器见证。一个张量是否复用一次移动，由目标表示、已有副本和存活期共同决定，不能按边数直接生成复制任务。

## ExecutionPlan

| 字段 | 含义 |
|---|---|
| schema_version、plan_id、plan_digest | 不可变计划及规范化内容摘要；规范化算法由首次实现 ADR 明确 |
| snapshot_id、deployment_context_id | 精确绑定候选/成本和部署环境 |
| partition_id | 选中的完整划分 |
| selected_candidate_ids | 每个区域恰好选择一个完整候选 |
| tasks | 区域计算或适配器执行任务；包含 task_id、候选/适配器引用及目标 worker/device |
| events、dependencies | 计算、适配、分配和释放事件，以及完整就绪和资源顺序依赖 |
| buffers、bindings | 物理缓冲描述及逻辑张量/端口与缓冲的绑定 |
| predicted_timing | 同一条件下的预测开始/完成时间，可缺省；不作为跨主机墙钟指令 |
| validation | 覆盖、接口、依赖、资源和容量检查结果及引用 |

任务必须使用匹配的已验证实现，运行时制品可用性仍需工作端确认。预测时间不是实际结果；执行应由事件和资源可用性驱动。请求重新规划返回新 plan_id，而非修改执行中的任务含义。

完整推理完成的条件是所有模型输出在 output_requirements 指定的位置、表示下可见，不能只取最后一个设备内核完成事件。

### 当前 C++ 计时与资源校验

`InputsAvailable` 的起点必须等于全部已声明 SourceReady 的最晚时刻；没有来源时使用计划原点 0。不能把输入就绪后、首个任务启动前的等待从指标中截掉。`OutputsDelivered` 的终点必须等于全部要求 SinkComplete 的最晚时刻；没有要求输出时等于计时起点。虚拟 Sink 是零成本边界，完成时刻等于目标 TensorBinding 的 ready 时刻；额外交付或同步必须先表示为任务，不隐含塞入 Sink。

上述数值相等要求与事件依赖同时成立；相同时间戳不能替代 Source→Start、Ready→Sink 或 Sink→Finish 的因果关系。未知时间保持 Unresolved，不从已知子集求最大值冒充完整边界。

资源校验分别检查给定时间表、两活动排他复用，以及累计用量的事件偏序保守上界。上界按每任务在同一资源上的局部峰值计量；在各任务启动切点，排除已由依赖证明完成或尚未启动的其他任务，累加其余任务峰值。上界超限返回 `resource_capacity_order_unproven` / Unresolved，不能据此宣称所有调度不可行。独立分支总请求不超过容量时无需人为串行。

当前 Handoff 的 Stage 连续串行执行，排队仅能发生在整项交接任务启动前；中间缓冲与工作区保守存活到整项任务完成。它不表达阶段间资源等待。需要 GPU→主机后等待另一链路的场景，须先扩展阶段事件及其生命周期协议，不能靠增大某个 Stage 的固定服务时间隐藏排队。本次未增加阶段事件，也未把该简化声明为一般调度模型的完整实现。

## 物理缓冲与内存保守模型

`BufferSpec` 记录 `buffer_id`、`pool_id`、`size_bytes`、对齐要求、分配事件和释放事件。`TensorBinding` 关联 tensor_id、表示、buffer_id 和 offset_bytes；共享物理内存池使用同一 pool_id，避免跨设备重复计算同一分配。

首版规则：

1. 调用方提供的边界输出缓冲在区域启动前完成分配和绑定。
2. 区域私有 workspace 在整个区域执行区间保守预留；作为 BufferSpec 计入后不得再加一次峰值。
3. 适配器目标缓冲在适配器启动前分配，其私有 workspace 同样全程预留。
4. 缓冲释放必须晚于全部读取、相关复制和设备操作完成；输出就绪不等于可以释放。
5. 输入只读，不允许未声明的跨区域原地覆盖；内部缓冲复用由后端验证。
6. 各时刻按唯一物理分配累加占用，不得超过 available_capacity_bytes_by_pool。

这是保守充分条件，可能拒绝内部峰值错开的实际可行计划。若后续引入阶段分配/释放和提前输出事件，需要升级相应接口及验证，不能仅修改 planner 的估算公式。

## 最小计划例子

选择 Conv–BiasAdd–ReLU 全融合 R123 在 GPU：根据 initial_residency 准备 X、W、b → 分配 Y 与 workspace → 执行已验证候选 → 区域完成后 Y 可见 → 按 output_requirements 必要时回传 Y → 所有读取结束后释放非持久缓冲。

若仅融合 Conv+BiasAdd，R12 输出 T2 给独立 ReLU；若 ReLU 在另一设备，还必须选择已验证的 T2 适配器。不能从“两个区域分别有实现”推断整体可连接。

## 失败与版本边界

planner 可返回可行计划、已证明违反约束的原因，或预算内尚未找到计划。后者不等于数学上不可行。runtime 的设备失联、资源不足或执行失败返回运行事件；apps 决定是否以新环境、新快照和新 attempt 重新规划，禁止在旧计划 ID 下无记录换设备或实现。
