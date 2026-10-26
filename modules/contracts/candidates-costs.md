# 区域候选、接口与成本契约

状态：0.2.0 草案，未实现。定义研究一输出和研究三反馈共享的记录；预测器、候选搜索和具体程序生成不在 contracts 实现。

## 区域与完整划分

`RegionSpec` 包含 `region_id`、`graph_id`、`member_node_ids`、`input_tensor_ids`、`output_tensor_ids`。成员非空，边界从语义图派生：输入由区域外产生并在区域内使用；输出在区域内产生且被外部使用，或属于模型输出。

`PartitionSpec` 包含 `partition_id`、`graph_id` 和区域集合。各区域互斥并覆盖全部算子；收缩后的图必须无环。不能让 planner 任意拼接重叠区域。首版不跨区域复制算子；原始单算子划分是参考候选，不无条件保证部署环境中可行。

Conv 示例中，R123={v1,v2,v3} 的输入为 X、W、b，输出为 Y。如果外部 v4 使用 T1，R123 的输出必须增加 T1。

## InterfaceSpec

| 字段 | 含义 |
|---|---|
| interface_id、region_id | 不可变接口与所属区域 |
| input_bindings、output_bindings | 按稳定边界端口绑定 tensor_id 和 RepresentationSpec |
| numeric_policy | 边界 dtype、内部允许变换及正确性要求 |
| visibility_policy | 首版固定为 `region_complete`：全部边界输出在区域完成后可见 |
| ownership_policy | 输入只读、输出由调用方预分配，无跨边界未声明 alias |

`RepresentationSpec` 描述物理排列、`strides_bytes`、`padding_elements_by_axis`、`alignment_bytes`、storage_format 和 `physical_size_bytes`。stride、对齐与物理大小按字节；padding 按指定逻辑轴的元素数，不能混用。特殊打包若不能由这些字段表达，须使用明确、可验证的格式描述，不能猜测裸 stride 的单位。转换不能隐含改变数值语义；量化或精度变化需要相应语义与验证依据。工作区上限是请求约束，不属于逻辑值身份。

## ImplementationSpec、CandidateSpec 与验证状态

`ImplementationSpec` 包含 `implementation_id`、`backend_id`、`registry_snapshot_id`、精确提供者 ID/版本/内容摘要、后端/编译器版本、ProgramSpec 引用（生成程序适用）、算法及内核结构摘要、编译选项、制品引用和已知实现特征。分块、物化和线程结构可以未知，但不能编造。制品缺失的探索实现不能交付运行。

`CandidateSpec` 通过 `candidate_id` 唯一关联 `region_id`、`interface_id`、`device_id`、`profile_id`、`implementation_id`、执行配置和证据。选择一个候选即选择完整记录，不允许从不同候选拼出设备、时延和工作区。

| 字段/状态 | 含义 |
|---|---|
| execution_status=`pending` | 尚未完成所需编译/接口/正确性/资源验证 |
| execution_status=`verified` | 声明环境与测试范围内具备可执行性证据 |
| execution_status=`failed` | 当前具体实现验证失败，附原因 |
| support_result | 单独记录后端能力：supported、unsupported 或 unknown |
| search_result | 单独记录 found、budget_exhausted 或 not_searched |

一次编译失败、未测时延或搜索耗尽不能推出该设备永久不支持区域。`verified` 不代表时延一定已经实测；可执行性和成本来源是两个独立维度。

`EvidenceRecord` 至少绑定对象、设备/后端版本、验证方法、输入范围或用例引用、数值政策、结果及原始记录引用。不能只用一个无来源的布尔值替代证据。

## MeasurementRecord：原始观测的唯一记录

此对象由 runtime 的受控测量服务结合后端原始观测产生，cost_model 只读取；experiments 构造评价政策并分析记录，不回写原始样本。生产/调优政策由 apps 提供，不依赖 experiments；MeasurementProtocol 的唯一定义见 program-inputs.md。

| 字段 | 含义 |
|---|---|
| measurement_id、schema_version | 不可变观测标识和版本 |
| candidate_id、profile_id | 被测实现及实际设备/软件条件 |
| run_id、attempt_id、task_id | 对应执行身份；独立测量也分配明确执行身份，不与生产运行混淆 |
| protocol_id、measurement_context | 协议、计时起止、同步、驻留、热身、重复次数、线程/设备配置及缓存处理 |
| samples_us、raw_observation_refs | 实际重复样本及原始观测引用；失败样本单独标记，不填零 |
| resource_observations、correctness_evidence_refs | 资源观测及使用的数值/接口测试证据 |
| observation_kind、data_role | real_device 或 simulation；以及 calibration、validation、test 等用途，防止标签泄漏 |
| budget_usage | 编译、测量、搜索的真实消耗；未发生的阶段不伪填 |
| outcome、failure_reason | 成功、失败或未知及关联原因；部分记录不冒充完整成功 |

原始记录创建后不覆写；修正产生新记录并引用被修正记录。其存在不自动证明任意输入正确，也不自动授权用于训练。只有适用域、协议及用途匹配的观测才能被派生为相应成本。

## CostRecord

| 字段 | 含义 |
|---|---|
| cost_id、candidate_id | 成本记录与完整候选关联 |
| source | analytical、predicted 或 measured |
| latency_us | 输入已驻留且就绪到区域输出满足接口可见条件的服务时延；未取得则 unknown |
| latency_statistic | latency_us 的统计口径，如 mean、median 或分位数及参数；预测值也声明对应目标统计量，比较时保持一致 |
| samples_us | 实测重复值或其制品引用；预测记录不得伪填 |
| measurement_refs | measured 记录必须引用对应 MeasurementRecord；派生统计要可追溯，不能成为另一份可独立改写的样本真值 |
| workspace_bytes_by_pool | 私有工作区峰值；排除边界输入输出及持久参数 |
| scoped_resources | 每线程、每块或每设备的寄存器、显式片上存储和执行资源需求 |
| traffic_bytes_by_level | 各存储层访问量及来源，未知项保持 unknown |
| ready_offset_us_by_tensor | 输出完整就绪偏移；首版与 latency_us 相同来源、同一口径 |
| measurement_context | 计时边界、热身、线程、驻留、同步、设备配置与缓存处理等 |
| uncertainty | 误差/适用域描述及依据；没有校准则不声称置信保证 |

区域内调用、转换、物化、同步计入本记录，区域外传输和排队不计入。私有内部完整临时张量计入工作区；同一物理分配不能又在边界缓冲中重复计费。测量前提不匹配时不能静默复用记录。

## CandidateSnapshot

`CandidateSnapshot` 包含 `schema_version`、`snapshot_id`、`registry_snapshot_id`、`graph_id`、`profile_id`、完整划分列表、区域/接口/候选/成本/证据引用、适配器目录及兼容关系引用。所有引用都应能解析，且属于同一冻结条件。

候选可执行性、局部资源通过，不代表完整划分已全局可行。planner 还需联合选择候选及适配器，并检查部署环境。pending 探索候选与可运行候选分开列出。

发布快照后不原地改写。新测量或新实现产生新记录和新 snapshot_id；现有计划始终引用旧快照的确切内容。不可兼容字段或含义变化须明确升级 schema，并提供迁移/拒绝策略。

## TuningRequest 与 TuningResult

请求包含 `request_id`、`snapshot_id`、`registry_snapshot_id`、`region_id`、`device_id`、`profile_id`、`interface_id`、必需 `oracle_ref`/`validation_policy_id`、可选 `seed_candidate_id`、`workspace_budget_bytes_by_pool`、搜索/编译/测量预算和执行条件。oracle 的身份和限制按 program-inputs.md 定义；不能用尚未验证的新实现自证正确。

snapshot_id 所指快照可包含仅有区域/接口的探索项，不要求该区域已有 verified 候选。已有实现优化和无种子区域探索使用同一固定接口契约；apps 可以从 fusion 保留的探索区域发起请求，不必先被 planner 选中。若单算子均有可执行参考，允许构造逐算子组合参考，但必须验证边界与资源，不能称为已经单内核融合。缺少可用 oracle 或生成规则时返回明确未满足条件，不伪造种子。

codegen 只能在固定语义、设备和接口内优化程序结构。成功结果返回新实现/候选、制品、成本和证据，以及实际消耗预算；失败结果返回原因和预算，允许状态为未找到。接口改变必须形成新接口候选，不能冒充同接口替换。

结果通过 apps 进入新快照并重新规划/验证；codegen 不直接修改正在运行的计划，也不导入 fusion 或 planner 形成反馈调用环。
