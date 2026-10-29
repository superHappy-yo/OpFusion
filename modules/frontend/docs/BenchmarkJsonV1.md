# benchmark-json/1 导入约定

`benchmark-json/1` 是本仓库 benchmark JSON 的受限来源 profile，不是 ONNX 导入器、通用 cuDNN JSON 标准或可执行模型格式。导入只建立逻辑张量表、有序逻辑算子和端口依赖；不执行模型、设备探测、融合、候选搜索或性能测量。

## 来源与图身份

- 系统统一使用 `opfusion::frontend::ModelImporter`，公开头为 [`opfusion/frontend/ModelImporter.h`](../include/opfusion/frontend/ModelImporter.h)。调用 `importer.importFile(path, ModelFormat::JsonGraph)` 或 `importer.importText(text, ModelFormat::JsonGraph, sourceName)`；本 profile 是其中一种来源格式，不是系统入口。
- 具体 JSON 适配位于模块内部 `src/JsonGraphImporter.cpp`。apps 不直接调用该适配，不读取 benchmark 目录来决定模型。
- 构造 ModelImporter 时提供已由装配层冻结的 `registry`，导入器保存其值快照，图记录该 registry 的身份，不借用原目录对象。工厂捕获的借用状态仍须由调用方保证寿命与并发安全，复制快照不深度冻结外部状态。通用读取器不发现或隐式登记算子包；内置来源绑定由 frontend 独立文件提供，apps 在冻结前显式调用 [公共绑定入口](../include/opfusion/frontend/BuiltinSourceBindings.h)。
- 每次成功导入产生新的随机 128 位 opaque token，以 32 位十六进制字符串表示；编码使用 classic locale，不受进程全局 C++ locale 的数字分组影响。该 token 不是文件名、内容哈希或跨导入缓存键；同一文本也得到新的图身份。随机源失败时明确报错，不以来源名作回退。
- `sourceName` 是运行时诊断中的来源标签，不参与图身份。仓库中的测试只使用可分享的中立数据，不记录本机路径或权重内容。
- 失败抛出 `opfusion::Error`，不返回部分图。成功结果的 `profile` 固定为 `benchmark-json/1`。

## 文档与张量

顶层必须是对象，恰有以下八个字段：`name`、`nodes`、`tensors`、`compute_data_type`、`intermediate_data_type`、`io_data_type`、`is_override_shape_enabled`、`min_required_engine_api_version`。当前 profile 只接受静态 FP32：compute、intermediate、IO 及每个 tensor 的数据类型为来源字符串 `float`，不执行隐式类型转换或量化。`is_override_shape_enabled` 必须为布尔 `false`；`min_required_engine_api_version` 必须恰有整数成员 `major:1`、`minor:0`、`patch:0`。该版本对象不是字符串，也不是已安装运行引擎的证据。

- `nodes`、`tensors` 都必须是数组；输入 JSON 不能有重复对象键或 raw NUL 字节。合法字符串中的转义 `\u0000` 仍作为来源数据保留，不能利用 raw NUL 让解析器忽略尾随内容。profile 之外的顶层或 tensor 字段明确拒绝，避免默默丢弃 initializer、显式模型输出或其他未支持语义。
- tensor 的 `uid` 是逻辑值 ID，不是数组下标。全部声明 tensor 都进入 DAG 和 `tensorOrigins`，包括未被节点引用的 tensor。
- `name` 保留来源名；`dims` 保留来源逻辑维度顺序。含 `nhwc` 的图名不触发轴重排；来源的 NCHW/KCRS 逻辑维度不改写为物理布局。
- `strides` 是有符号 64 位整数数组；`alignment` 是非负整数提示；`virtual` 和 `is_runtime_pass_by_value` 是布尔提示。它们只进入 `TensorOrigin`。可缺省的提示没有值时保持 `nullopt`，缺省 stride 为空数组。
- 提示的 stride/alignment 单位、分配和别名语义没有建立；不能用它们建立 Representation、Buffer、驻留、生命周期或工作区结论。`virtual=true` 也不代表可以删除值或忽略外部消费者。

## 节点与扩展边界

每个节点必须是对象，具有字符串 `type`、`name`。frontend 原样递归保存完整节点到 `SourceNode.fields`，并设置以下通用元数据。

| 字段 | 来源 |
|---|---|
| `format` | `benchmark-json/1` |
| `type` | 节点的 `type`，不按算子名称解释 |
| `operation` | 优先取存在的 `inputs.operation` 字符串，否则取顶层 `resample_mode` 字符串；不判断节点属于哪种算子 |
| `name`、`index` | 来源节点名及从 0 开始的数组位置 |
| `fields` | 完整节点对象，包括未被 frontend 解读的属性和 null |

来源端口、可选参数、属性、数值政策和 source binding 的语义由 `Registry.create(source)` 选择的独立 factory 负责。通用 JsonGraphImporter 不包含具体算子头、逐算子 switch、dynamic cast 或中央算子清单；frontend 内部 JsonGraphBindings 的各工厂只包含 contracts 的对应公共具体类头，严格来源字段检查不进入领域类。允许节点没有 `inputs`/`outputs` 而使用扁平的 root 字段；`resample_mode` 只作为通用选择标签，不让 frontend 解读 root ports。factory 返回的节点 ID 必须为 `source.index + 1`，而不是来源名或 tensor UID。

null 作为来源值保留给 factory；可选 null 端口不应由 factory 变成 tensor 引用。图的边只由已构造 Operator 的有序输入输出推导，frontend 不扫描原始 JSON 猜边。相同值出现在两个输入端口时，保留两次使用；多输出和多消费者同样不去重。

内置绑定分别位于 `src/JsonGraphBindings/<Name>Binding.h/.cpp`：Add、Concat、Conv、MaxPool、Mul、ReLU、Resize、Sigmoid、Slice。CMake 按同名文件约定生成绑定入口，SourceFields 仅共享严格字段转换，不参加发现；MatMul 没有本格式绑定。来源支持子集没有扩大，具体属性约束见 [算子语义](../../contracts/docs/Operators.md)。新增来源绑定不要求修改通用读取器或中央名称清单。

## 数值与类型严格性

`SourceValue` 仅包含 null、bool、`int64_t`、有限 double、字符串、递归数组和对象。JSON 布尔不转成整数，字符串不转成数值，`1.0` 不转成整数属性。

- 整数属性必须落在 `INT64_MIN..INT64_MAX`；正数即使先由 JSON 库解析为 unsigned，也不能截断或绕回。
- 超过 JSON 库整数范围的整数字面量不能退化成 double。SAX 回调按“可选负号加 ASCII 数字”的整数形态检查 token，明确拒绝这类整数溢出；不依赖可能被库按 locale 本地化的小数点，也不修改进程全局 locale。
- double 必须有限；`1e400` 等溢出、NaN/Infinity 非 JSON token 都失败。此 profile 不承诺十进制到 double 的精确表示，也不额外拒绝有限的下溢舍入结果。
- `uid`、`alignment` 等无符号字段不接受负数、bool 或小数；`dims`、`strides` 不接受 bool、小数或数值字符串。
- JSON 最大容器嵌套深度为 256。其他来源属性的具体类型/范围由 factory 校验，不以未知属性作默认支持。

## 验证与失败

DAG 构造只通过 [`Operator`](../../contracts/include/opfusion/graph/Operator.h) 的 `validate(inputs)`、input-only `infer(inputs)` 和 `indexSemantics(inputs)` 交接：比较声明输出的 shape/dtype，核对逻辑轴、索引访问、tensor 引用、唯一生产者、数值约定和无环性。轴信息在发布前按公共访问关系单调收敛，推导不能读取声明输出作为答案。通用读取器无逐算子分支，重复 UID 在导入时直接失败。

公共 contractVersion=6.0.0，operatorApiVersion=5.0.0。算子身份只含命名空间和名称，删除语义版本；JSON 的来源 profile 和原始字段保持不变。来源 compute_data_type 只声明计算类型，未声明累加精度保持未知。CLI inspection 输出升级为 /3，新增已验证属性、完整数值要求和实际广播，不改变八图结构。

factory 的 `Error` 保留 code，并附来源节点 index/name。DAG 错误保留其验证 code 和 node ID/语义上下文；对于以 `node <id> (` 开头的外层节点错误，frontend 补回相应来源 index/name，不将内部提及的其他节点误当成错误来源。所有导入错误附 `sourceName`。图级环等错误可以只有图级来源，不伪造某个单独节点是根因。

内部 JSON 适配保留 `benchmark_*` 来源错误码，包括 JSON 语法、重复对象键、整数/浮点范围、嵌套深度、schema 类型/缺项/未知字段、profile 不匹配、重复 UID、factory 空返回或错误节点 ID 和随机图 ID 生成失败。统一入口的未冻结 registry、未支持格式、文件打开和文件读取分别使用 `model_registry_unfrozen`、`model_format_unsupported`、`model_file_open`、`model_file_read`。流读取抛出的 `std::ios_base::failure` 同样转换为含来源路径的 `model_file_read`。未知绑定、非法属性、缺失引用、虚假声明输出、多生产者和环等契约/扩展错误不被吞掉，也不转换为成功诊断。

## 推断边界不等于执行边界

来源没有 initializer、显式有序模型输入和完整模型输出，因此本适配保持 `DAG.modelInputs() == nullopt` 及 `DAG.modelOutputs() == nullopt`。DAG 公共模型具备这些字段，不代表此来源已经提供它们。真实 producer 的值标为 OperatorResult，其余来源未声明的角色仍为 Unknown；可推导逻辑轴以 Derived/Mixed 标记，不伪装成来源声明。

- `externalInputs()` 是根据无内部生产者推断的依赖，不是已确认的模型运行时输入清单。不能把外部 Add 操作数、bias 或其他外部值猜成常量。
- `terminalOutputs()` 是根据生产/消费关系推断的终端值，不升级为显式模型输出；也不意味着未使用 tensor 可以丢弃。
- 无权重内容、平台和部署 Boundary，输入驻留、输出交付目标、容量、计时口径均未建立。
- 成功诊断明确标记 `external_inputs_inferred`、`terminal_outputs_inferred`、`weight_contents_unavailable`、`physical_hints_only`、`execution_conditions_unestablished`、`candidates_not_searched`，另以 `source` 保存来源标签。
- 以上限制仅说明执行条件未建立、候选未搜索，不构成对模型、算子、设备或全部候选的全局不可行性证明。

## 验证职责

[BenchmarkTest.cpp](../tests/BenchmarkTest.cpp) 已使用真实 ModelImporter 和独立 `Registration` / `SourceBinding` 装配；中立 Projection 实现完整 Operator 协议，`infer` 与恒等访问 `indexSemantics` 均只读输入，不用声明输出作 oracle。原有 16 个测试分组及全部反例保留，另增加非法 ModelFormat 的文件/文本入口反例。

- 中立来源回归保留完整递归字段、null 端口、重复操作数/多输出、UID 与来源提示、随机图身份与 locale、整数/浮点边界、raw/escaped NUL、重复键/深度/schema、工厂与 DAG 失败上下文、虚假声明输出、环、多生产者、文件错误及冻结目录要求。
- `BuiltinSourceBindings` 分组接收旧 standard 测试的完整来源解析用例；先显式登记十个语义，再安装九个来源 selector，核对 MatMul 无绑定。九个具体工厂的角色次序、类型化逻辑轴、FP32 声明、重复 UID、null/空占位与非法 inactive 字段、插值/池化/卷积政策、坏类型/端口、index 溢出及未知来源反例仍在。纯数学语义测试不因此依赖 frontend。
- 测试 fixture、目标名和 CMake 单份登记保持不变；不删除/禁用目标或放宽期望来掩盖接口迁移。现有逗号小数 locale 用例仅在环境没有所需 locale 时报告 SKIP，行为未变，不是完整本地化覆盖证明。

本轮基线默认配置成功，但旧 CoreTest/ModelTest 编译失败，未运行基线测试。迁移后已在另一全新目录完成默认选项配置/完整构建及 **6/6 CTest 目标**；本文件描述的中立来源和九工厂回归通过，但逗号小数 C locale 子用例因缺少环境明确 SKIP。contracts/frontend 共 30 个公开头通过独立 C++17 语法检查；新 CLI 八图核验 264 节点、415 张量、301 条完整有序端口边，执行/候选/成本仍为未建立/未搜索/未知。这不是设备或数值 kernel 验收，clang-format/tidy 未安装、未运行。完整结果见 [整改状态](../../contracts/docs/CodeQuality.md)。
