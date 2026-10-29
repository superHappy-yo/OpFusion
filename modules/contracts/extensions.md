# 算子、设备与实现扩展契约

状态：设计规格与当前静态子集。算子静态发现/登记及 manifest 2.0.0 已实现，operator API 5.0.0、contract 6.1.0；设备/编译器/实现插件、动态加载和执行仍是未来规格。扩展是代码所有权与能力交付单元，不是新的研究模块或独立服务。

## 目标与适用边界

在既有契约能够表达新增能力时，新增一个算子、设备描述或后端实现，只修改对应扩展包及其测试和说明，核心 frontend、fusion、cost_model、planner、codegen、runtime 不增加按名称分支，也不维护逐项注册清单。扩展无法用现有语义表达时，先提出有版本的契约变更；“可扩展”不意味着任意新语义都能零改动接入。

采用组合、公开 C++ 端口接口和显式工厂。面向对象可以组织行为实现，但不要求所有算子继承一个庞大基类，也不把所有设备排成 CPU/GPU/NPU 继承树。描述记录不可变；有资源或状态的执行对象通过端口提供，并遵守 runtime 的生命周期。

## 扩展包与 Manifest

约定发现路径为以下各类别下的直接子目录，每个扩展包以本目录 `plugin.toml` 声明能力。当前不添加占位 manifest 或可执行实现。

```text
extensions/operators/<plugin_id>/plugin.toml
extensions/devices/<plugin_id>/plugin.toml
extensions/compilers/<plugin_id>/plugin.toml
extensions/implementations/<plugin_id>/plugin.toml
```

目录类别表示主要责任，不限制一个算子包携带其来源绑定、分解规则及相关目标实现贡献。扩展依赖通过稳定 ID 和版本约束声明，不能通过导入兄弟扩展源码复用内部实现。

plugin_id 使用单一路径段（小写字母开头，后续为小写字母、数字、点、下划线或短横线），禁止斜线、父目录段和通配符；全仓唯一，不因位于不同类别而允许重名。带不同作用的贡献 ID 另行命名。

| Manifest 字段 | 必需语义 |
|---|---|
| manifest_schema_version | manifest 格式版本；当前为 2.0.0，与包/API/契约版本分别管理 |
| plugin_id、plugin_version | 全局命名空间内唯一的扩展标识、发布版本；目录名与 plugin_id 一致 |
| plugin_api_range | 兼容的 OpFusion 扩展 API 版本范围；无匹配时拒绝装配 |
| kind | operators、devices、compilers 或 implementations，与所在类别一致 |
| contributions | 带贡献类型、贡献 ID、静态描述引用、可选 factory_ref，以及贡献级 requires/external_dependencies/环境条件的列表 |
| requires | 所有贡献共同必需的契约/提供者 ID、版本范围与能力；目标专用依赖放到对应贡献，不使用私有头文件路径或内部链接符号 |
| external_dependencies | 全包共同必需的第三方包/库、版本与环境条件；目标专用依赖放到对应贡献；发现器不自动安装依赖 |
| artifact_refs | 描述、规则、测试向量或程序的包内相对引用；需要随包计算内容摘要 |

`factory_ref` 指向本包内实现公开端口的 C++ 工厂。其序列化形式、二进制入口及 ABI 版本方案尚未实现，必须由独立集成任务明确，不能把任意类名或符号地址当作稳定 ABI。解析 manifest 不加载工厂库、不执行静态初始化；装配获选提供者时才显式加载。纯描述包可以没有工厂，复用通过 requires 引用的已有提供者。第三方依赖不因写入 manifest 就变成已安装或已验证。

按启用贡献及其依赖闭包检查环境。同包中的语义/CPU 参考贡献可独立有效；缺少某 GPU SDK 只使依赖它的贡献 provider_unavailable，不能连带禁掉无关语义。显式请求该 GPU 贡献时必须明确失败，不能把必需依赖称作可选而略过。包的初始化入口不得提前加载全部可选依赖。循环检查覆盖实际贡献依赖闭包；纯分解循环另按语义规则拒绝。

扩展包可以依赖 contracts 的公开接口及其声明的第三方依赖，不得导入 apps、其他业务模块具体实现或兄弟扩展内部代码。共享行为需要由端口注入；必要的通用实现归其已有所有者维护，不能在多个包中复制公共类型。

## 发现、解析与注册快照

apps 拥有启动和装配流程，按约定目录枚举 manifest；新增扩展不需要修改 apps 中的逐插件头文件/链接清单、switch、注册数组或仓库级白名单。部署任务可以显式限制启用集合，但这是运行配置，不能改变扩展之间的语义或写入权限。

处理顺序为：枚举 manifest → 校验结构与包内引用 → 解析贡献 ID 和依赖 → 检查冲突及版本 → 显式构造所需提供者 → 冻结 RegistrySnapshot → 向各模块注入公开查询端口。路径排序只用于可复现诊断，不作为解决冲突或选择更快实现的规则。

| RegistrySnapshot 字段 | 含义 |
|---|---|
| registry_snapshot_id、schema_version | 不可变注册快照身份与契约版本 |
| plugin_records | 参与装配的扩展 ID、版本、内容摘要、依赖解析结果 |
| operator_definitions、frontend_bindings | 已通过结构校验的语义定义及来源映射 |
| device_descriptors、provider_descriptors | 设备描述和编译/执行/实现提供者的公开描述 |
| resolved_dependencies | 精确解析后的提供者 ID 和版本，不保留运行时随机选择 |
| diagnostics | 被拒绝或未启用项及原因，不能静默忽略必需依赖失败 |

注册表为一次应用/任务装配的实例，不是进程全局可变 singleton。库加载与静态初始化时不注册能力、不探测设备、不编译或执行。新增包、更新版本或改变启用集合产生新的 registry_snapshot_id；GraphSpec、HardwareProfile 和 CandidateSnapshot 绑定此身份，现有图、能力和候选快照继续引用其原注册环境，不能被热修改。注册快照冻结的是描述符、内容摘要和依赖解析结果；实际工厂构造的设备 session、内存句柄及执行事件由运行时管理，不能误写成不可变公共记录。

内容身份覆盖 manifest、被使用的包内实现/规则/数据及解析后的依赖，不能只散列版本字符串；缓存/产物等生成物不参与源内容身份。引用必须解析到该包允许的来源，不能通过路径跳转或符号链接隐含依赖兄弟插件。部署复现需保留确切包内容和依赖，读取旧快照时缺失则拒绝，不能仅以相同 plugin_id 重绑定新文件。

同一贡献 ID 被多个包定义必须报冲突，禁止后加载覆盖先加载。多个不同实现提供者可以合法支持同一算子/区域，它们作为独立候选保留；选择由显式请求或所属优化模块完成，发现器不按名称、目录顺序或默认优先级替用户选实现。依赖循环、未满足范围及不兼容 API 应有明确诊断。

## 算子扩展

`OperatorKey = (namespace, name)` 表示规范化算子身份，不含 semantic_version。不同广播或数值规则由已验证属性与数值要求表达；互不兼容的语义需不同命名空间/名称或契约/API 主版本迁移。包、来源 opset、后端及系统契约版本保留。注册快照仍不可变，不原地改变旧图所引用的定义。当前 operator 贡献包含 semantic_version 会明确失败。

| OperatorDefinition 字段/能力 | 责任与约束 |
|---|---|
| op_key | 规范化语义身份；与 plugin_version、ONNX opset 分开；OperatorSpec 用同名字段引用 |
| signature_schema | 输入输出的有序端口、可选/可变参数及属性约束；保留 Add(U,U) 的两次使用 |
| semantic_spec_ref | 明确数学/索引/数值语义及允许变换，不以一句名称替代 |
| validate | 对属性、端口、dtype、轴和已知 shape 进行纯校验；不执行设备探测或调优 |
| infer | 在输入签名与约束下推导输出签名，返回已知结果、带条件结果或 unknown；首版不能解析为静态形状时报告不支持当前执行范围 |
| traits | 有证据的逐元素、广播、归约、纯性、重计算及索引关系等描述；必须声明适用条件，未知值不补成 true |
| reference | 正确性参考的声明及实现引用，或可验证分解；它不自动是目标设备的高性能实现 |
| frontend_bindings | 外部格式算子到该语义的有界映射，见下表 |
| decomposition | 可选：分解为已注册算子的规则、条件与数值约束；分解依赖必须有界且终止 |
| implementation_contributions | 可选：本包提供的 lowering、库调用或模板引用及所需提供者 ID；缺失时不伪造可执行候选 |

语义/shape/traits 能力通过 `OperatorSemanticPort` 查询，核心模块使用公共语义而不是 `if op == "NewOp"`。参考执行也必须经声明的参考提供者调用，不能让纯校验函数隐式运行硬件。分解后的节点保留原算子来源、规范化说明和契约身份，不能把一次图改写的收益混入融合方法的独立收益。

| FrontendBinding 字段 | 含义 |
|---|---|
| source_format、source_domain、source_op | 外部格式、域与名称；例如 ONNX 的 domain/op_type |
| source_version_range | 明确覆盖的源算子版本范围；不能无依据声称匹配所有未来 opset |
| target_op_key / decomposition_ref | 映射到一个规范化算子或声明的规范化子图 |
| attribute_mapping、port_mapping | 属性默认值、端口、轴和广播的转换规则；无损保留必要来源信息 |
| applicability、validation_ref | dtype/shape/属性等匹配条件及其测试证据 |

同一源节点必须能得到唯一语义映射；两个绑定的版本/条件范围有歧义时拒绝导入，而非随机选一个。提供者通过契约定义的来源节点描述接收数据，外部框架对象不进入全仓公共语义图。新增源格式本身可能需要 frontend 适配，不能与“既有格式内新增算子绑定”混为一谈。

“识别算子”意味着能够解析其身份与语义。没有目标 lowering/库实现时，系统可以返回结构化不支持诊断，或使用该包声明且已验证的分解；不能静默回退到其他设备。未知融合 traits 时保留边界，未知成本时返回 unknown 或采用明确来源的保守策略，不能填零或直接视为有融合收益。

## 设备与后端扩展

设备型号、物理实例、编译器和运行适配器分开建模。新增设备包可以仅提供某个已有架构的新型号描述；全新架构通常还需要编译、执行和测量适配。二者都可以在相关扩展包内交付，但后者不能仅增加 metadata 就获得执行能力。

| 对象 | 核心字段与边界 |
|---|---|
| DeviceDescriptor | device_model_id、厂商/型号、arch_family、描述版本、静态资源字段及来源；不包含虚构实例 ID 或运行时可用容量 |
| DeviceMatchRule | 真实探测属性与型号描述的匹配条件、适用驱动/软件范围；模糊匹配失败需报告 |
| ProbeProvider | 提供者 ID/版本及工厂；经 apps 显式调用后返回实际实例、资源、软件条件和探测来源 |
| CompilerProvider | 提供者 ID/版本、可消费程序表示及版本、目标架构、支持查询与 CompilePort 工厂；接收 BackendCompileRequest |
| ExecutorProvider | 提供者 ID/版本、产物格式/ABI、目标架构/驱动条件、ExecutePort 及设备同步/复制能力 |
| ImplementationProvider | 区域或算子语义、接口和目标约束 → 默认程序、lowering 或库调用方案；复用 ImplementationCatalogPort 的候选语义 |

编译器和设备是多对多关系。例如同一编译器可支持多个设备家族，同一 GPU 可接受多个工具链的产物。通过稳定的提供者 ID、版本、目标条件和 ABI 联结，不要求一个 Device 子类同时实现所有编译器。

独立的 `implementations/<plugin_id>/` 可以为已有 op_key 或区域模式贡献新的 lowering/实现，引用既有编译器和设备条件，无需修改原算子包、原设备包或核心模块。匹配条件至少包括算子身份/区域模式、编译器提供者及版本、目标架构、shape/dtype/布局/数值政策和接口限制；贡献自身具有唯一 ID。多个满足条件的实现作为有独立身份的候选集合返回，不能通过最后注册者覆盖或无依据优先级隐藏其他实现。请求明确指定提供者时只使用该提供者；未指定时由 codegen/fusion 在显式预算内处理候选，planner 从已验证候选中做最终选择。

首版实现生成只承诺两条路径：分解为已支持的规范算子交给现有生成器；或由实现贡献为其声明匹配的**完整区域**返回 ProgramSpec/可登记库程序。compiler 按 representation_kind/version 消费可序列化方案，不通过跨插件传递私有 TVM/MLIR builder 组合程序。单算子实现不自动支持与任意邻居融合；细粒度片段组合需要另立公开、版本化协议。预算内枚举与结构选择由 codegen 控制，插件不得在一次生成调用中隐藏无预算搜索。

静态描述用于解释能力，实际 `HardwareProfile` 绑定 device_id、设备实例、探测时的软件版本和已解析提供者。标称峰值不能作为实测性能，配置中的物理容量不能覆盖 DeploymentContext 的动态可用容量。没有真实设备的测试只产生模拟证据。

`CapabilityPort` 按区域/接口/设备/编译器/执行器组合返回 supported、unsupported 或 unknown 及原因；不能只依据 type="GPU" 或设备名称推断支持。候选与制品须关联 registry_snapshot_id 及可解析的精确提供者版本/摘要，编译缓存身份还应纳入实际依赖解析结果、ProgramSpec、目标和编译选项，避免更新插件后旧计划执行了不同实现。ProgramSpec、BackendCompileRequest 及入口请求的定义见 [程序与任务输入契约](program-inputs.md)。

## 阶段与失败语义

以下是不同对象的检查结果，不应压缩为一个全局 ready 布尔值，也不是所有插件必经的单一路径。

| 结果 | 已证明的内容 | 尚未证明的内容 |
|---|---|---|
| discovered | 找到可读取的 manifest | 格式正确、依赖可满足、可执行 |
| validated | manifest/契约/引用与依赖通过结构检查 | 支持某个算子、设备或模型输入 |
| supported | 给定区域/接口/设备组合满足已声明支持规则 | 编译成功、数值正确、性能收益 |
| compiled | 得到可校验的目标产物 | 声明用例内正确或资源可行 |
| verified | 对应 CandidateSpec 在声明环境和验证范围内有执行证据 | 任意输入正确、任何设备可用或已经获得加速 |

预编译库候选不必伪造 compiled 阶段；纯设备描述包也不会获得候选执行状态。能力、制品、候选验证与成本来源分别保存，沿用 [候选契约](candidates-costs.md) 和 [后端端口契约](backend-ports.md)。

至少返回：manifest_invalid、duplicate_contribution、api_incompatible、dependency_missing、dependency_cycle、binding_ambiguous、operator_unknown、shape_unresolved、device_match_failed、provider_unavailable、unsupported、compile_failed、validation_failed，以及预算耗尽等原因。原因应关联具体包/贡献/请求；一个实现失败不意味着整个算子或设备永久不支持。

## 必须满足的未来验收

1. 只增加一个算子扩展目录，重新装配后即可发现来源绑定并构造规范化节点；不修改核心逐插件头文件/链接清单、switch 或中央注册表。
2. 已有执行架构的新设备型号只增加描述与匹配规则；真实探测与已有提供者匹配后形成新 HardwareProfile，无需改 planner 或 cost_model 的设备名称分支。
3. 全新架构的相关扩展包同时交付所需适配与测试；缺少执行器、工具链或依赖时明确拒绝，不能仍登记 verified。
4. 测试重复贡献 ID、来源映射歧义、依赖循环、不兼容 API、缺少第三方依赖及不支持的 shape/dtype。
5. 测试未知 trait 不导致非法融合、未知成本不变为零、没有可执行候选不触发隐藏设备回退。
6. 相同 manifest、版本及启用集合得到相同注册内容；新快照不修改旧计划所引用的提供者和产物身份。
7. 分别验证“可识别”和“可执行”；新增 lowering 在固定接口、数值和工作区条件下完成真实测试后才能登记执行证据。
8. 同包 GPU 贡献依赖缺失时语义/CPU 贡献仍可使用；显式请求不可用贡献明确失败。完整区域与额外中间输出接口必须精确匹配，不从单算子支持推导任意融合支持。

以上验收均为实现要求，当前尚未执行插件发现、编译或设备验证测试。
