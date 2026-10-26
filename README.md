# OpFusion

面向异构设备的图级融合、分布式映射调度与设备内程序优化研究系统。

当前状态：**文档与 AI 协作骨架**。已定义模块边界、数据契约、运行角色、任务模板和验收要求；尚无业务实现、可执行入口、自动测试或 CI 门禁。所有标为 API、策略或协议的内容目前均为实现规格。

## 从哪里开始

1. AI 先读 [AGENTS.md](AGENTS.md)，再读目标目录的 `AGENTS.md` 和 `README.md`。`CLAUDE.md` 只引用同级规则，避免维护两套指令。
2. 人先读 [系统架构](docs/architecture/system.md)、[依赖规则](docs/architecture/dependency-rules.md) 和 [设计决策](docs/architecture/decisions.md)。
3. 实现前选用 [单模块任务](docs/ai/module-task.md) 或 [集成任务](docs/ai/integration-task.md)，列明允许修改路径和验收条件。
4. 按 [实施路线](docs/roadmap/implementation.md) 从最小可验证链路推进；不要直接开始全平台、全算子的系统实现。
5. 新增算子/设备先读 [扩展入口](extensions/README.md)、[扩展契约](modules/contracts/extensions.md) 和 [扩展任务模板](docs/ai/extension-task.md)。默认只增加一个扩展包，不修改核心模块。

本次完成的检查与限制见 [初始化核查记录](docs/verification/bootstrap.md)。

## 模块地图

| 目录 | 职责 | 研究对应 |
|---|---|---|
| [contracts](modules/contracts/README.md) | 图、张量、设备、候选、代价、计划和协议的公共契约 | 公共基础 |
| [frontend](modules/frontend/README.md) | 模型导入、语义规范化、端口和张量关联 | 输入基础 |
| [backends](modules/backends/README.md) | 设备能力、默认实现、编译/运行/转换原语 | 硬件适配 |
| [cost_model](modules/cost_model/README.md) | 条件特征、代价预测、校准和不确定性 | 研究一 |
| [fusion](modules/fusion/README.md) | 合法区域、完整划分、候选保留 | 研究一 |
| [planner](modules/planner/README.md) | 划分选择、设备映射、事件与缓冲计划 | 研究二 |
| [codegen](modules/codegen/README.md) | 固定区域/设备/接口下的内部程序搜索 | 研究三 |
| [runtime](modules/runtime/README.md) | Coordinator/Worker、执行事件、传输与产物生命周期 | 执行基础 |
| [experiments](modules/experiments/README.md) | 基线、测量协议、数据划分、正确性与效果评估 | 证据基础 |
| [apps](apps/README.md) | 唯一装配层：选择具体组件、预算、快照和反馈流程 | 集成 |

这是模块化单仓。Coordinator 与 Worker 是部署角色，不要求把每个研究模块做成网络服务。算法模块只通过公共数据和端口协作，由 `apps` 注入实现。

`extensions/` 是算子、设备、编译器和实现规则的扩展所有权区，不新增研究模块。约定由 apps 启动发现包内 manifest，再通过小接口、组合和不可变 registry 注入现有模块；不依赖深继承或核心中的算子/设备名称分支。当前仅完成机制规格，尚无插件加载器。

## 首个贯穿示例

```text
v1: Conv2D(X,W) -> T1
v2: BiasAdd(T1,b) -> T2
v3: ReLU(T2) -> Y

v1 --T1--> v2 --T2--> v3
```

Conv 不含 bias；X 为 `[1,16,32,32]`，W 为 `[32,16,3,3]`，b 为 `[32]`，偏置沿通道轴广播；stride=1、padding=1、dilation=1、groups=1，全 FP32。原始图只有三个算子节点和两条端口依赖边。融合区域输入 X/W/b，输出 Y；存在外部消费者时须保留相应中间输出。

## 当前工程选择

- 默认使用 Python 实现领域逻辑、算法和控制流程；设备热点优先调用成熟后端，必要时增加受限的原生实现。
- 第一模型入口为 ONNX 模型包，连同 shape/数值/预算要求和部署上下文进入内部 GraphSpec；后续通过 frontend 扩展支持 torch.export、StableHLO。
- 研究三优先验证 TVM 张量程序/调度路线，Triton 作为 GPU 后端选项；MLIR/IREE 为按需求评估的替代路径，当前未安装或验证任何工具链。
- 采用端口与适配器、显式依赖注入、可替换策略、不可变版本快照和运行时状态机。
- RPC 库、部署工具和具体编译后端在相应里程碑前通过 ADR 确定；当前不添加占位依赖或虚假的启动命令。
- 不将图上融合直接等同于单内核或性能收益；不将预测、模拟和实机观测混用。

模块提示词规定默认修改边界，但不是文件系统沙箱。[边界强制机制规格](docs/ai/enforcement.md) 区分了提示词、未来 CI 检查与系统权限；当前未启用后两者。仓库未添加许可证，许可证选择另行决定。

本轮选型理由、模块修订与接入边界见 [输入、后端与语言选择](docs/architecture/technology.md)；检查结果见 [扩展设计核查](docs/verification/extensions.md)。
