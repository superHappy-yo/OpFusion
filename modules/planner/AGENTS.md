# planner：模块 AI 规则

本文件补充根 `AGENTS.md`。开始任务前先读根规则、本模块 README 和 `modules/contracts/execution-plan.md`、`modules/contracts/candidates-costs.md`、`modules/contracts/backend-ports.md`。

## 职责与边界

- 你负责：在固定候选快照和部署环境下，联合选择完整划分、实现、适配器、设备、事件顺序及物理缓冲。
- 默认可写：`modules/planner/` 中与任务有关的实现、`tests/`、README、`docs/` 和 `proposals/`。
- 默认只读：其他模块、`apps/`、全局配置及治理文件。公共 contracts 规格与类型也只读。
- 所有 `AGENTS.md`、`CLAUDE.md`、范围/所有权策略即使在本目录也受保护；不得用修改规则来扩大当前任务。
- 禁止编译、设备执行、改候选时延、重新定义语义、静默生成新实现；不得把局部合法当成完整计划可行。
- 导入边界：仅依赖 contracts 的公开类型和端口；其他能力由 apps 注入。禁止导入邻居模块的具体实现。

## 输入与交付

- 输入：CandidateSnapshot、DeploymentContext、兼容关系和资源规则。
- 输出：ExecutionPlan、计划检查报告或明确的失败/未找到原因。
- 贯穿例子：同一 Conv 图在 X 位于 CPU 或 GPU、Y 要求回主机时，必须分别计入适配和交付成本。
- 公共接口不足时先写本模块 proposal；不要制造影子字段或修改邻居来适配自己。用户已授权的跨模块变更由集成任务明确拆分，不重复索要授权。

## 怎么实现与验证

- 当前没有业务代码。未来实现按实际任务放在本模块 `src/`，测试放在本模块 `tests/`；包名、依赖和真实命令由 M1 的集成任务建立，不能自行创造全仓构建体系。
- 先写明输入输出、失败语义和最小可验证行为；优先简单实现，不引入尚无使用场景的策略/基类。
- 重点验证：初始驻留和最终输出位置；Join 兼容性；跨主机依赖；输出缓冲启动前分配；全程工作区预留；重复副本不重复计费。
- 完成前对照任务 allowlist 检查全部 diff、未跟踪文件及重命名两端；报告真实验证，不虚构硬件执行或测试通过。

## 扩展与技术选型规则

先读 modules/contracts/extensions.md、program-inputs.md 和 docs/architecture/technology.md。

- 只消费固定候选、能力、接口与资源约束；不得按新设备型号名添加放置分支或绕过能力查询。
- 探索区域不能作为 verified 候选直接调度；没有可行候选时给出诊断，由 apps 决定生成/测量/重规划。
- 计划绑定图/profile/候选的 registry 快照身份，不能在执行前静默解析为最新插件。
