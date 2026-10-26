# contracts：公共数据与端口契约

状态：`schema_version = 0.1.0` 的设计草案。当前只有规范，没有 DTO、序列化器、RPC、验证器或可调用 API 实现。首次实现须将本规范转化为明确类型和契约测试，不得把文档示例当作已运行结果。

本模块只拥有跨模块共享的数据含义、约束、错误语义及端口定义。不能依赖其他业务模块、设备驱动、深度学习框架、网络库或数据库。`frontend`、`backends`、`cost_model`、`fusion`、`planner`、`codegen`、`runtime` 依赖这里的公开契约，由 `apps` 注入具体实现；不能为方便调用而反向导入它们。

## 读取顺序与所有权

| 文档 | 唯一定义的内容 | 主要生产者与消费者 |
|---|---|---|
| [semantic-graph.md](semantic-graph.md) | GraphSpec、算子、逻辑张量、端口依赖和数值政策 | frontend → fusion、cost_model、codegen |
| [candidates-costs.md](candidates-costs.md) | 区域、接口、实现候选、原始 MeasurementRecord、成本、证据和不可变快照 | fusion、codegen、cost_model → planner |
| [execution-plan.md](execution-plan.md) | 部署环境、适配器、执行计划、物理缓冲 | planner → runtime |
| [worker-protocol.md](worker-protocol.md) | 制品引用、控制消息、任务状态、重试、事件归属和 ExecutionReport | runtime 协调端 ↔ 工作端 |
| [backend-ports.md](backend-ports.md) | 硬件描述、能力查询和编译、运行、测量、预测、存储端口 | backends/runtime/cost_model 实现，apps 装配 |

公共字段发生改变时，修改此处唯一规范；其他模块只能引用，不能复制一份“兼容结构”后自行演进。普通模块任务可在本模块之外的自身 `proposals/` 提出需求，但不能顺便修改公共契约。跨模块变更流程见 [AI 工作流](../../docs/ai/workflow.md)。

## 通用约定

- 标识字段使用明确的 `graph_id`、`candidate_id` 等名称；ID 是不透明稳定标识，不从字符串拼接猜测对象关系。跨记录关系须显式引用并校验存在性。
- 序列化顶层对象携带 `schema_version`；嵌套字段继承所属对象版本。当前字段表是逻辑 schema，具体编码格式和语言类型尚未选定。
- 持续时间和就绪偏移使用 `_us`，内存和数据量使用 `_bytes`。计数使用整数；时间估计可为非负浮点数。`null/unknown` 不等于零。
- 本地持续时间由单调时钟测量；跨主机时间戳仅用于记录，不直接相减推导通信时延。展示日期使用带时区的时间格式。
- `schema_version` 表示数据结构兼容性；`snapshot_id` 表示一次冻结的数据集合；实现、后端、硬件记录的版本分别说明执行条件。不能用一个“version”字段代替所有含义。
- 不可变语义、接口、成本和快照的变更都产生新记录。记录可以被标记不再推荐，但已经被计划引用的内容不得原地覆盖。
- shape、dtype、单位、端口和错误代码不得静默转换。未知的必需字段或不兼容版本应拒绝；可选扩展须声明可忽略范围，不能影响核心语义。
- 失败记录至少说明 `code`、`message`、关联对象及是否允许重试；“不支持”“验证失败”“超预算”“尚未找到”“执行状态未知”必须区分。
- 契约不携带任意 shell 命令作为执行入口。设备程序通过受支持后端和制品引用执行；大张量、参数与程序二进制通过制品/数据通道传递。

## 首次实现的验收基线

契约测试应覆盖：端口重复使用、错误 shape、广播轴、区域输出保留、分区后成环、未知成本、布局不兼容、快照引用失效、工作区重复计费、旧 attempt 事件和重复请求。以 Conv–BiasAdd–ReLU 为最小贯穿例子，同时添加至少一个带外部消费者的分支图。

这里只规定测试应验证的性质；当前仓库尚无测试命令，不得在报告中将这些要求写成“测试已通过”。
