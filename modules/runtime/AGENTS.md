# runtime：模块 AI 规则

本文件补充根 `AGENTS.md`。开始任务前先读根规则、本模块 README 和 `modules/contracts/worker-protocol.md`、`modules/contracts/execution-plan.md`、`modules/contracts/backend-ports.md`。

## 职责与边界

- 你负责：按已验证的固定计划协调 worker，管理真实执行、产物、传输、事件和缓冲的生命周期。
- 默认可写：`modules/runtime/` 中与任务有关的实现、`tests/`、README、`docs/` 和 `proposals/`。
- 默认只读：其他模块、`apps/`、全局配置及治理文件。公共 contracts 规格与类型也只读。
- 所有 `AGENTS.md`、`CLAUDE.md`、范围/所有权策略即使在本目录也受保护；不得用修改规则来扩大当前任务。
- 禁止内部调用 planner/fusion/codegen 偷换计划；不得信任过期 attempt 事件或将断连视为成功；不承诺恰好一次执行。
- 导入边界：仅依赖 contracts 的公开类型和端口；其他能力由 apps 注入。禁止导入邻居模块的具体实现。

## 输入与交付

- 输入：ExecutionPlan、ArtifactRef、部署配置、后端执行端口。
- 输出：ExecutionReport、任务/传输事件和原始 MeasurementRecord。
- 贯穿例子：先确认 X/W/b 就绪和 Y 缓冲已分配，再执行 R123；只有正确 attempt 的完成事件才能使 Y 可用。
- 公共接口不足时先写本模块 proposal；不要制造影子字段或修改邻居来适配自己。用户已授权的跨模块变更由集成任务明确拆分，不重复索要授权。

## 怎么实现与验证

- 当前没有业务代码。未来实现按实际任务放在本模块 `src/`，测试放在本模块 `tests/`；包名、依赖和真实命令由 M1 的集成任务建立，不能自行创造全仓构建体系。
- 先写明输入输出、失败语义和最小可验证行为；优先简单实现，不引入尚无使用场景的策略/基类。
- 重点验证：重复提交、状态迁移、过期事件、取消、断连、worker 故障、资源释放、产物 digest/版本不匹配；控制消息与张量数据分离。
- 完成前对照任务 allowlist 检查全部 diff、未跟踪文件及重命名两端；报告真实验证，不虚构硬件执行或测试通过。
