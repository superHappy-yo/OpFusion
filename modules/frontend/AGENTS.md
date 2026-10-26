# frontend：模块 AI 规则

本文件补充根 `AGENTS.md`。开始任务前先读根规则、本模块 README 和 `modules/contracts/semantic-graph.md`。

## 职责与边界

- 你负责：将外部模型转换为设备无关的 GraphSpec/TensorSpec，保存有序端口、轴含义和数值政策。
- 默认可写：`modules/frontend/` 中与任务有关的实现、`tests/`、README、`docs/` 和 `proposals/`。
- 默认只读：其他模块、`apps/`、全局配置及治理文件。公共 contracts 规格与类型也只读。
- 所有 `AGENTS.md`、`CLAUDE.md`、范围/所有权策略即使在本目录也受保护；不得用修改规则来扩大当前任务。
- 禁止选择设备、执行融合搜索、预测时延、调用硬件或静默量化；不得丢弃外部消费者、复用端口或原始来源。
- 导入边界：仅依赖 contracts 的公开类型和端口；其他能力由 apps 注入。禁止导入邻居模块的具体实现。

## 输入与交付

- 输入：模型文件/上游图、静态 shape、参数引用、数值政策。
- 输出：规范化图、张量表、来源映射与诊断。
- 贯穿例子：将明确无偏置的 Conv、按通道 BiasAdd 和 ReLU 登记为三节点；前端 Conv 已带 bias 时避免再加一次。
- 公共接口不足时先写本模块 proposal；不要制造影子字段或修改邻居来适配自己。用户已授权的跨模块变更由集成任务明确拆分，不重复索要授权。

## 怎么实现与验证

- 当前没有业务代码。未来实现按实际任务放在本模块 `src/`，测试放在本模块 `tests/`；包名、依赖和真实命令由 M1 的集成任务建立，不能自行创造全仓构建体系。
- 先写明输入输出、失败语义和最小可验证行为；优先简单实现，不引入尚无使用场景的策略/基类。
- 重点验证：重复操作数 Add(U,U)、多输出/多消费者、缺 shape、未知算子、非法广播；规范化前后数值语义一致。
- 完成前对照任务 allowlist 检查全部 diff、未跟踪文件及重命名两端；报告真实验证，不虚构硬件执行或测试通过。

## 扩展与技术选型规则

先读 modules/contracts/extensions.md、program-inputs.md 和 docs/architecture/technology.md。

- 首版接收 ONNX ModelPackage，输出带 OperatorKey、SourceMap 和 registry_snapshot_id 的图；导出图与规范化图分别记录。
- 既有源格式内的新算子通过扩展的 frontend binding/语义端口接入；禁止新增算子名分支或修改中心算子列表。未知语义、shape 或不支持控制流明确诊断。
- Conv 自带 bias 的规范化必须有政策与数值依据，不重复加入 bias，不假定导出器保留源结构。
