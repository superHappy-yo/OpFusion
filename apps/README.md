# apps — 装配和建模 CLI

apps 是唯一装配层：显式登记 contracts 语义、frontend 来源绑定及已链接可选插件，冻结实例级 Registry，再调用公开 ModelImporter。没有设备初始化、硬件执行、优化或成本预测。

```bash
./compile.sh
./run.sh model.onnx
./run.sh --context scenario.json model.onnx
./run.sh models/
./run.sh benchmark/yolo/*.json
build/apps/opfusion operators --json
build/apps/opfusion inspect model.onnx --json
```

脚本的路径、JSONL、失败继续与缓存规则见 [根 README](../README.md)。CLI inspection schema 为 `/4`、operators schema 保持 `/2`，节点以 type/namespace 标识，不再输出冗余版本化 semantic。算子属性由公共只读接口提供，CLI 不按名称或 dynamic_cast 获取属性。数值要求缺失保持 null，广播能力与本实例行为分别展示，索引未知显式报告。

## 部署 context

`--context` 可省略。格式为 `opfusion-context/1`，张量通过 ONNX 名称引用，由 apps 装配 Platform、Device、Representation、Boundary、TimingPolicy 并调用公共校验。字段严格校验；非法引用/容量/计时冲突直接失败，必要条件未知保留 issues，不填默认设备能力或成本。

示例对应输入 `images`、输出 `out` 均为 FP32 `[1,3,4]` 的图：

```json
{
  "schema": "opfusion-context/1",
  "id": "example",
  "platform": {
    "id": "platform",
    "devices": [{"id": "cpu", "node_id": "host", "kind": "cpu"}],
    "pools": [{"id": "ram", "physical_identity": "host-ram", "physical_capacity_bytes": 4096, "accessible_devices": ["cpu"]}],
    "resources": [],
    "links": []
  },
  "representations": [
    {"id": "input", "tensor": "images", "pool": "ram", "dtype": "float32", "layout": "row_major", "packing": "dense", "actual_bytes": 48},
    {"id": "output", "tensor": "out", "pool": "ram", "dtype": "float32", "layout": "row_major", "packing": "dense", "actual_bytes": 48}
  ],
  "inputs": [{"id": "source", "tensor": "images", "representation": "input", "data_ref": "caller-input", "ready_us": 0}],
  "outputs": [{"id": "target", "tensor": "out", "representation": "output"}],
  "available_bytes_by_pool": {"ram": 2048},
  "available_resources": {},
  "timing": {"start": "inputs_available", "finish": "outputs_delivered", "includes_input_transfers": true, "includes_output_transfers": true}
}
```

该示例保留 CPU 条件、后端、stride、padding、对齐等未知。输入驻留必须覆盖运行时输入，输出目标覆盖全部模型输出；参数可另外提供驻留，声明 data_ref 不等于读取输入数据。

可选设备字段：architecture、instance、configuration、backends、cpu_allocation（quota_cores/affinity/numa_binding/numa_node）、environments（backend/version/abi/runtime_version/requires_driver/driver_version）。设备类别 cpu/gpu/accelerator/other/unknown 只描述类别。

Resource 使用 id/kind/scope/unit/capacity/device/pool/shared_devices/sharing_ref；Link 使用 id/source_pool/target_pool/resources（resource/amount）/bandwidth_bytes_per_us/startup_us。缺省动态可用容量保持未知，已知上限不能大于物理库存。

Representation 可提供 strides_bytes、padding_elements_by_axis、alignment_bytes、backend_context（backend/version/abi/context）、quantization（kind/scales/zero_points/axis）。只支持声明 dense packing；布局支持 row_major/column_major/nchw/nhwc/oihw/hwio/strided。缺失信息不默认连续或无量化。

TimingPolicy 仅声明 start（inputs_available/first_region_start/before_initialization）、finish（region_outputs_ready/outputs_delivered）和三个 includes 标志。禁止用户提供 start_event/finish_event。结果为 `timing_event_binding=pending_plan`；未来 Plan 绑定具体事件，初始化费用尚未建模。

## 可选静态插件

manifest schema 为 `2.0.0`，operator API 兼容范围须包含 `5.0.0`，contracts 为 `6.1.0`。贡献 ID 为 `namespace:name`；semantic_version 字段已删除，保留时明确拒绝。插件包自身 plugin_version、后端版本和依赖版本范围保留。贡献依赖所查版本属于提供者（内置贡献使用 operator API，外部贡献使用包版本），不是数学算子版本。

启动只发现约定 operators 目录中的 manifest，核对 schema/API、冲突、引用、依赖及编译内容，再显式调用已链接工厂；未链接贡献必须重新编译，没有动态共享库加载或隐式回退。目录发现可选插件为空不移除内置语义。

## 验证

既有 CLI 八图和插件负例保留；新增 ONNX、context、单文件/目录/通配符/空格路径/失败继续回归，以及两个完整模型的独立签名与来源验收。参考运行时仅为测试，不是本系统设备执行证据。

inspection/4 的 index_semantics.pieces 展示 domain、inputs/outputs、shape_reads、归约类型和缩放；表达式包含可选 modulus 以及带显式值域条件的 lookup。modeling.index_relations 汇总关系知识，不表示设备支持结果。来源未声明的精度保持 null 并说明原因。

提供 timing 政策而尚无 Plan 时，诊断为 timing_events_pending_plan，不再误报政策缺失；仍不能把未绑定的计划事件当作计时已验证。

inspection/4 的 logical_axes 包含各轴范围、可选 semantic 和 definitions。definitions 引用生产者的 relation piece，保留坐标表达式、输入端口/维度及 shape_reads；axis_model 表明来源是签名、算子关系或标量无轴。modeling.axis_models 是结构完整性，named_axis_annotations 是可选命名覆盖，两者分开。旧 axes_origin 字段删除；命名缺失使用 null，不猜测 Head、Token 或物理布局。
