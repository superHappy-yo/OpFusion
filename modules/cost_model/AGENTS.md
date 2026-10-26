# cost_model：模块 AI 规则

本文件补充根 `AGENTS.md`。开始任务前先读根规则、本模块 README 和 `modules/contracts/candidates-costs.md`、`modules/contracts/semantic-graph.md`、`modules/contracts/backend-ports.md`。

## 职责与边界

- 你负责：构建语义/边界/实现/硬件条件特征，预测完整区域代价、输出不确定性，并依据独立观测校准。
- 默认可写：`modules/cost_model/` 中与任务有关的实现、`tests/`、README、`docs/` 和 `proposals/`。
- 默认只读：其他模块、`apps/`、全局配置及治理文件。公共 contracts 规格与类型也只读。
- 所有 `AGENTS.md`、`CLAUDE.md`、范围/所有权策略即使在本目录也受保护；不得用修改规则来扩大当前任务。
- 禁止直接操作硬件、生成融合划分、调度任务、替换观测真值；不得偷用留出标签或把未知值当零。
- 导入边界：仅依赖 contracts 的公开类型和端口；其他能力由 apps 注入。禁止导入邻居模块的具体实现。

## 输入与交付

- 输入：候选、接口、设备 profile、执行条件、已有 MeasurementRecord。
- 输出：带条件和来源的预测 CostRecord、模型版本及校准请求。
- 贯穿例子：对 Conv-BiasAdd-ReLU，区分内部复用与仍物化 T1/T2 的实现；理论字节节省不能直接写成时延收益。
- 公共接口不足时先写本模块 proposal；不要制造影子字段或修改邻居来适配自己。用户已授权的跨模块变更由集成任务明确拆分，不重复索要授权。

## 怎么实现与验证

- 当前没有业务代码。未来实现按实际任务放在本模块 `src/`，测试放在本模块 `tests/`；包名、依赖和真实命令由 M1 的集成任务建立，不能自行创造全仓构建体系。
- 先写明输入输出、失败语义和最小可验证行为；优先简单实现，不引入尚无使用场景的策略/基类。
- 重点验证：同一语义不同边界/实现/设备的记录不会混键；预测和实测分离；相同候选与预算下验证收益符号、排序及 regret。
- 完成前对照任务 allowlist 检查全部 diff、未跟踪文件及重命名两端；报告真实验证，不虚构硬件执行或测试通过。

## 扩展与技术选型规则

先读 modules/contracts/extensions.md、program-inputs.md 和 docs/architecture/technology.md。

- 使用扩展提供的有版本语义与通用特征、实际能力快照；禁止新增算子或设备名条件分支作为接入要求。
- 未见算子、未知特征/资源或超出校准域时返回 unknown/明确适用域及校准请求，不用默认0成本或标称峰值冒充实测。
- 插件提供的分析特征必须可追溯；新的特征 schema 或训练政策需要明确演进，不允许借私有字段偷读评价标签。
