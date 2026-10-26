# 工作端控制与执行事件契约

状态：0.2.0 草案，未实现；传输库、RPC 编码和持久化机制待实现阶段确定。这里规定逻辑语义，不声称存在 exactly-once 执行或跨主机精确时钟。

## ArtifactRef 与数据平面

`ArtifactRef` 包含 `digest_algorithm`、`digest`、`size_bytes`、`media_type`、`locator` 和必要的兼容性元数据。首版选择 SHA-256 内容摘要；locator 只负责定位，内容身份由摘要决定。相同名字或路径不证明内容一致。

程序制品还须与 implementation_id、backend_id、架构、编译选项和接口绑定；工作端在加载前核对适用条件。计划的 digest 需要稳定的规范化编码，不能直接对语言对象的默认字符串哈希。

控制消息只传计划、制品与缓冲引用、状态及小型元数据。模型参数、输入输出张量和程序二进制走数据/制品通道，不内嵌在心跳、状态事件或调度消息中。日志不替代张量交付协议。

## 身份与请求信封

| 字段 | 含义 |
|---|---|
| schema_version | 控制契约版本 |
| request_id | 单个控制操作的幂等标识 |
| run_id | 一次逻辑推理/验证运行 |
| attempt_id、attempt_epoch | 协调端为本次尝试分配的标识和同 run 下递增代次 |
| plan_id、plan_digest、snapshot_id | 绑定不可变计划及其候选条件 |
| task_id | 计划内的具体任务；任务身份由 run/attempt/task 共同限定 |
| worker_id | 目标或上报工作端 |

字段不适用的消息可以省略，但不能以空字符串充当合法引用。重复 request_id 必须对应同一规范化请求；同 ID 携带不同内容应拒绝。

## 最小控制操作

| 操作 | 作用与响应 |
|---|---|
| DescribeWorker | 返回工作端身份、可用设备/后端 profile 引用及当前能力；动态可用容量单独报告 |
| ActivateAttempt | 安装 run 对应的当前 attempt_epoch 和计划绑定；较旧代次拒绝，相同内容重复安装可返回原结果 |
| PrepareTask | 核对代次、计划、制品、接口、输入与资源预留；成功返回 prepared 或失败原因，不执行计算 |
| StartTask | 在依赖满足后执行已准备任务，返回已接收或当前状态；完成由事件报告 |
| GetTaskStatus | 查询任务状态及已发布事件序号，供超时/断连后对账 |
| CancelAttempt | 请求取消该尝试；取消请求不等于设备计算已停止 |
| ReleaseResources | 在无在途操作和未完成读取后释放该尝试拥有的资源；重复释放返回已释放 |

这些操作仅可处理工作端已识别的后端与制品，不提供任意 shell 执行接口。具体 RPC 的批处理与消息形态可以优化，但必须保持上述语义。

## Task 状态与事件

正常状态路径为 `accepted → prepared → running → succeeded`。任务也可以进入 `rejected`、`failed`、`cancel_requested`、`cancelled` 或 `unknown`。

- `accepted` 只说明请求通过基本校验，不表示已执行。
- `prepared` 表示制品、接口和资源准备完成；调用方输出缓冲已分配绑定。
- `running` 表示后端已接受执行，不等于输出可见。
- `succeeded` 必须在后端完成并满足输出可见性、完整性及交付记录要求后发布。
- `cancel_requested` 允许转为 `cancelled`、`succeeded`、`failed` 或 `unknown`，不得提前释放可能仍使用的缓冲。
- `unknown` 表示无法确认是否执行或完成，不能当成失败零耗时，也不能无条件重新提交相同任务。

事件至少包含身份信封、`event_seq`、状态、错误/结果引用、输出缓冲或制品引用、工作端本地持续时间及资源观测。event_seq 在同一任务尝试内递增；协调端去重，并拒绝使状态倒退的过期事件。跨主机时间戳不能用作总时延的唯一证据。

## 幂等、重试与旧结果隔离

工作端在幂等记录存续期内对同 request_id 返回相同接收结果或当前状态，不能重复启动。进程重启丢失记录时，应恢复记录或报告 unknown；不能默认为从未执行。

协调端仅接纳当前 run 的活动 attempt_epoch 的结果；工作端仅接收已安装代次的执行命令。旧事件可以归档，但不能提交为新尝试输出。输出、缓冲和结果索引包含 attempt 身份，避免旧写入覆盖新结果。

代次隔离保证结果归属和提交规则，不保证失联设备上的旧计算立即停止。重试前应确认可安全重试、资源隔离和输入身份，显式创建新 attempt；不得宣称 exactly-once。初版遇到 worker 宕机可以终止/报告未知，由 apps 决定重新运行，不要求透明迁移正在执行的内核。

完整 run 成功由协调端确认全部所需输出的最终交付；不是收到任意一个 task 的 succeeded 就完成。取消、回收和失联处理均不得突破执行计划的缓冲生命周期要求。

## ExecutionReport：完整执行尝试的交付

由 runtime coordinator 汇总，apps 决定后续优化/重规划，experiments 通过公开运行入口读取。不在报告中原地修改计划或成本快照。

| 字段 | 含义 |
|---|---|
| report_id、schema_version | 不可变报告标识和契约版本 |
| run_id、attempt_id、attempt_epoch | 对应逻辑运行和具体尝试 |
| plan_id、plan_digest、snapshot_id、deployment_context_id | 此次执行实际绑定的完整条件 |
| outcome | succeeded、failed、cancelled 或 unknown；只能在全部要求的输出交付完成后记 succeeded |
| task_results、event_refs | 各任务终态/未知状态及事件记录，保留失败和未完成任务 |
| delivered_outputs | tensor_id、目标位置、表示及可验证的交付引用；不得只记录生产端内核完成 |
| elapsed_us、timing_scope | 协调端同一单调时钟下的运行范围耗时；另保留 worker 本地持续时间，不混减跨主机时钟 |
| measurement_refs、correctness_evidence_refs | 指向 candidates-costs.md 定义的 MeasurementRecord 和数值/接口证据 |
| errors、resource_disposition | 错误，以及资源已释放、仍在使用或状态未知的记录 |

部分输出和资源状态未知必须保留其状态；报告失败不等于所有设备操作已停止。新尝试生成新报告，旧报告不能由后到达事件改写成另一个尝试的成功。
