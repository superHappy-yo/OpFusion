# 输入、后端、语言与可扩展性选择

状态：2026-10-26 的设计决策与待验证技术路线。模块边界和扩展原则采用；具体编译工具链需通过 PoC 后锁定版本。当前仓库没有业务代码、插件发现器或设备运行证据。

## 1. 保留模块，完善扩展边界

保留九模块与 apps。研究一是 cost_model/fusion，研究二是 planner，研究三是 codegen。extensions 是贡献内容的所有权区，不是第十个研究模块，也不是独立网络服务。

backends 内部按通用编译路由、设备操作与库调用适配机制分工；具体算子、型号、厂商 SDK、编译器和实现规则放入独立扩展包。一个设备可用多个编译器，一个编译器也可服务多个设备，不能用一棵 Device 继承树把二者绑定。

codegen 选择和搜索结构，backends/编译扩展把明确方案编译成产物；runtime 管理执行生命周期。候选注册、编译缓存和测量不改变这些归属。

## 2. 输入格式与内部表示

采用 ModelPackage＋OptimizationRequest＋DeploymentContext 进入优化流程；运行使用冻结计划和 RunRequest。详细字段以 [入口契约](../../modules/contracts/program-inputs.md) 为准。

```text
ONNX（首版） / torch.export、StableHLO（以后按需）
                    ↓ frontend + 格式/算子映射贡献
GraphSpec + TensorSpec + SourceMap + RegistrySnapshot 引用
                    ↓ 三项研究通过区域/接口/候选契约协作
执行计划 + 后端产物 + 证据
                    ↓ runtime + 设备适配贡献
CPU / GPU / 以后有明确适配的设备
```

ONNX 是交换输入；内部算法消费 GraphSpec，不直接处理各框架对象。统一表示复用明确的算子语义，不凭算子名称猜测含义。ONNX 允许实现使用自己的内部表示。[ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html)

torch.export 保留 PyTorch 运算和形状约束，但并不保证捕获任意 Python 程序；StableHLO 是框架与编译器之间的高层算子可移植层。首版不同时建立多条前端转换链。[torch.export](https://docs.pytorch.org/docs/stable/export.html)、[StableHLO](https://openxla.org/stablehlo)

首版为静态 shape、无控制流、纯张量值 DAG。动态 shape、自定义算子、状态及量化/稀疏支持取决于明确的语义和后端能力，不由文件扩展名保证。LLM 可从固定形状的一次 prefill/decode 开始，KV cache 必须显式进入状态/输入输出约定。

ONNX Conv 可自带 bias；规范化三节点 Conv→BiasAdd→ReLU 必须有政策和来源映射。导出器和运行后端均可能提前优化，研究输入需冻结摘要、导出配置、规范化版本和算子统计；后端优化需记录，不能归为自研融合贡献。[ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html)、[ORT 图优化](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

## 3. 代码生成选型

| 选项 | 在本系统的位置 | 当前决策 |
|---|---|---|
| TVM 张量程序与调度路线 | 表达循环、tile、存储、线程映射，生成 CPU/GPU 目标程序 | 研究三优先 PoC；具体 API 和依赖锁定后使用 |
| Triton | GPU 内核生成/编译提供者 | 可选 GPU 后端与对照，不作为全设备统一执行层 |
| MLIR Linalg/Transform/LLVM | 可定制的中间表示与编译变换基础设施 | 有必要深入控制变换时评估；不从零同时建设完整编译栈 |
| IREE | MLIR 生态中的编译与运行平台 | 偏重既有跨平台部署时作为替代主线评估 |
| 厂商库/SDK | 已实现算子/子图、运行与设备接口 | 参考实现、强基线和设备专用适配 |

TVM 提供张量程序调度、目标代码生成和外部库接入路径，符合研究三要控制内部结构的需要；这只是选型理由，不是本项目性能证明。[TVM 架构](https://tvm.apache.org/docs/arch/)、[代码生成](https://tvm.apache.org/docs/arch/codegen.html)

Triton 主仓兼容说明当前列 NVIDIA/AMD GPU，CPU 仍标开发中；独立设备后端需单独核实版本和支持范围。IREE 列 CPU、CUDA、ROCm、Vulkan、Metal 等稳定配置。二者均不能直接证明任意 NPU 可运行。[Triton](https://github.com/triton-lang/triton#compatibility)、[IREE](https://iree.dev/guides/deployment-configurations/)

统一的是 ProgramSpec/CompilePort/ExecutePort 和区域边界，不是要求所有设备共用一个 kernel DSL 或相同 tile 参数。TVM/IREE 等只处理明确的区域和接口；其默认全图融合/分区不得覆盖 OpFusion 的研究一、二。所有内部调优、转换和编译成本登记预算，无法观测的结构保留 unknown。

## 4. 语言选择

| 部分 | 默认选择 |
|---|---|
| 领域对象、图分析、预测、融合、调度、结构搜索 | Python；以明确类型和窄协议隔离实现 |
| 应用装配、Coordinator、实验流程 | Python |
| Worker 的状态/请求逻辑 | 首版 Python，复用成熟原生设备 API |
| 计算、编译器内部、设备热点 | 复用原生实现；有具体需要或测量依据时写 C++ |
| 跨语言二进制接口 | 按需使用 C ABI，明确所有权和错误语义 |

Python 负责搜索/生成方案不代表用 Python 逐元素执行张量计算。应分开测量离线导入/编译/调优、计划生成、在线提交、设备计算/通信和完整请求；确认主机开销或部署要求后再迁移热点，不预先维护两套系统。

## 5. 新增算子与设备的扩展方式

采用开放扩展的小 Protocol/必要的 ABC、不可变描述、组合、实例级注册表和依赖注入。不是每个算子都必须写一个类：仅含参数/组合分解的扩展可以主要由数据表达；行为扩展实现相应小接口。不要用深继承、自动类扫描或全局装饰器副作用注册。

新增包在 `extensions/<kind>/<plugin_id>/` 内提供 manifest 和所需贡献，apps 在启动阶段统一发现，无须修改核心导入列表。算子可以提供语义、shape 校验、trait、前端映射、参考实现和可选分解/lowering；新增设备型号可以复用已存在的探测/编译/运行 ID，全新架构还需要适配器与实机验证。已有算子增加一个目标实现可独立放在 implementations 包，不修改原算子或设备包。

RegistrySnapshot 冻结描述、版本、依赖解析与内容身份；设备 session、stream、buffer 不放在其中，其生命周期由 runtime 管理。发现、语义校验、支持判定、编译成功、执行验证和成本测量分别记录。契约详见 [扩展规范](../../modules/contracts/extensions.md)。

“无需修改核心”适用于现有接口和语义覆盖的扩展。控制流、可变状态、新事件语义等超出契约的能力需显式演进，不靠扩展私有字段隐瞒。未知 trait 禁用相关优化，未知成本保持 unknown，不把新算子自动当作 elementwise 或默认可融合。

## 6. 新实现探索与验证路线

TuningRequest 区分必需的语义参考与可选 seed_candidate。apps 可以对 fusion 保留的 pending 区域发起探索，不要求 planner 先选中它。逐算子组合可作为参考路径；新程序验证后进入新快照，再参与正式规划。

首次工具链 PoC 固定 Conv→BiasAdd→ReLU 与 MatMul→BiasAdd→GELU 两类区域，在 CPU 和首个真实 GPU 上验证；分别检查只输出 Y 与保留外部消费者所需中间值。记录正确性、编译/调优成本、工作区、内核数和端到端耗时。通过后选定一个主要生成后端；不要同时建设 TVM、完整自研 MLIR 栈和多套运行时。

扩展机制的验收另外要求：仅新增扩展目录即可被发现，核心 diff 为空；独立实现包能给旧算子增加目标支持；重复 ID、映射冲突、依赖环和不兼容版本有明确诊断；旧计划不静默绑定新插件。以上均为未来测试要求，当前未运行。
