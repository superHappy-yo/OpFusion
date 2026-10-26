# fusion：模块 AI 规则

本文件补充根 `AGENTS.md`。开始任务前先读根规则、本模块 README 和 `modules/contracts/semantic-graph.md`、`modules/contracts/candidates-costs.md`、`modules/contracts/backend-ports.md`。

## 职责与边界

- 你负责：从原始图推导边界，生成合法完整划分，查询已有实现与代价，按有限预算保留候选图。
- 默认可写：`modules/fusion/` 中与任务有关的实现、`tests/`、README、`docs/` 和 `proposals/`。
- 默认只读：其他模块、`apps/`、全局配置及治理文件。公共 contracts 规格与类型也只读。
- 所有 `AGENTS.md`、`CLAUDE.md`、范围/所有权策略即使在本目录也受保护；不得用修改规则来扩大当前任务。
- 禁止最终设备映射、开始时刻选择、RPC 或执行；不搜索任意 tile/线程/内核结构；不在原图中加入设备节点。
- 导入边界：仅依赖 contracts 的公开类型和端口；其他能力由 apps 注入。禁止导入邻居模块的具体实现。

## 输入与交付

- 输入：GraphSpec、能力/实现/成本端口、候选和测量预算。
- 输出：不可变 CandidateSnapshot、接口及兼容关系引用、搜索轨迹。
- 贯穿例子：输出 Π0、Π12、Π23、Π123 中有依据的备选；只有后端支持且验证符合接口，才登记相应可执行候选。
- 公共接口不足时先写本模块 proposal；不要制造影子字段或修改邻居来适配自己。用户已授权的跨模块变更由集成任务明确拆分，不重复索要授权。

## 怎么实现与验证

- 当前没有业务代码。未来实现按实际任务放在本模块 `src/`，测试放在本模块 `tests/`；包名、依赖和真实命令由 M1 的集成任务建立，不能自行创造全仓构建体系。
- 先写明输入输出、失败语义和最小可验证行为；优先简单实现，不引入尚无使用场景的策略/基类。
- 重点验证：完整覆盖且互斥、商图无环、外部输出保留、参考候选；pending 不伪装 verified；同候选池/预算测试保留策略。
- 完成前对照任务 allowlist 检查全部 diff、未跟踪文件及重命名两端；报告真实验证，不虚构硬件执行或测试通过。

## 扩展与技术选型规则

先读 modules/contracts/extensions.md、program-inputs.md 和 docs/architecture/technology.md。

- 通过 OperatorSemanticPort 的条件 traits 和实现查询判断合法区域，不因新算子名增加特殊分支；未知属性保留边界，不能当作任意可融合。
- 可保留合法但暂无实现的探索区域/接口供 apps 请求 codegen；与可供生产规划的 verified 候选分开。
- 一个单算子有实现不证明整个融合区域有实现；检查全部边界输出和精确语义/接口。
