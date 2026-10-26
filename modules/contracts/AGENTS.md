# contracts：模块 AI 规则

本文件补充根 `AGENTS.md`。开始任务前先读根规则、本模块 README 和 `modules/contracts/semantic-graph.md`、`modules/contracts/candidates-costs.md`、`modules/contracts/execution-plan.md`、`modules/contracts/worker-protocol.md`、`modules/contracts/backend-ports.md`。

## 职责与边界

- 你负责：维护共享数据对象、端口、单位、版本、状态和纯不变量；不承担任何算法或设备行为。
- 默认可写：`modules/contracts/` 中与任务有关的实现、`tests/`、README、`docs/` 和 `proposals/`。
- 默认只读：其他模块、`apps/`、全局配置及治理文件。本模块契约规格仅在任务明确属于契约维护时可改。
- 所有 `AGENTS.md`、`CLAUDE.md`、范围/所有权策略即使在本目录也受保护；不得用修改规则来扩大当前任务。
- 禁止导入其他业务模块、硬件 SDK、模型框架、网络/数据库库；禁止提供设备 I/O、预测、搜索或执行实现。
- 导入边界：无业务模块依赖；仅允许纯数据和验证所必需的轻量基础依赖。

## 输入与交付

- 输入：建模规格及经协调的接口变更请求。
- 输出：Graph/Tensor、Hardware、Candidate/Cost、ExecutionPlan、Worker 协议和公共端口规范。
- 贯穿例子：Conv 不含 bias；BiasAdd 沿通道轴 1 广播；原图为 3 节点、2 条带端口边。
- 公共接口不足时先写本模块 proposal；不要制造影子字段或修改邻居来适配自己。用户已授权的跨模块变更由集成任务明确拆分，不重复索要授权。

## 怎么实现与验证

- 当前没有业务代码。未来实现按实际任务放在本模块 `src/`，测试放在本模块 `tests/`；包名、依赖和真实命令由 M1 的集成任务建立，不能自行创造全仓构建体系。
- 先写明输入输出、失败语义和最小可验证行为；优先简单实现，不引入尚无使用场景的策略/基类。
- 重点验证：端口与唯一生产者、shape/dtype、商图无环、字段单位、版本兼容、未知值/失败状态；按每个消费方进行契约核对。
- 完成前对照任务 allowlist 检查全部 diff、未跟踪文件及重命名两端；报告真实验证，不虚构硬件执行或测试通过。
