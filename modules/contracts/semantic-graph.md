# 语义图契约

状态：0.2.0 草案，未实现。语义图描述需要计算什么，不包含设备放置、内核启动、物理缓冲或传输指令。

## GraphSpec

| 字段 | 含义 |
|---|---|
| schema_version、graph_id | 契约版本及不可变语义图标识 |
| registry_snapshot_id、source_map_ref | 解析算子语义所用的扩展快照；源模型与规范化节点/端口的对应记录 |
| nodes | 算子节点表，每个节点只代表一次逻辑运算 |
| tensors | 逻辑张量表；一个值只登记一次 |
| model_input_ids | 模型运行时输入；不包含固定参数 |
| model_output_ids | 完整推理必须交付的结果，可以直接引用模型输入 |
| numeric_policy | 全图精度、允许变换、特殊浮点值及输出验证要求 |

首版为静态 shape、无控制流的 DAG，不支持跨区域未声明的原地覆盖。输入输出张量关系采用不可变值语义；物理存储在执行计划中单独定义。

## OperatorSpec

| 字段 | 含义 |
|---|---|
| node_id、op_key | 唯一节点标识及 OperatorKey=(namespace,name)，定义见 extensions.md |
| attributes | 影响语义的参数，如卷积步长、padding、groups、归约轴 |
| input_tensor_ids | 按输入端口排列的有序列表，允许重复引用同一张量 |
| output_tensor_ids | 按输出端口排列的有序列表 |
| index_semantics | 迭代、归约、广播、输入输出索引关系的规范化描述 |
| numeric_override | 必需的局部数值约定；不得无依据放宽全图数值政策 |

输入输出 shape、dtype 从张量表读取。运算量、扇入扇出等可以缓存为派生特征，但必须能追溯到上述语义，不成为另一份事实来源。实际寄存器、流量、工作区和时延不写成算子的固有常数。

算子身份、属性校验、shape 推导和有条件 traits 通过 [扩展契约](extensions.md) 查询，核心不维护新增算子名的 switch。图中实例化的 index_semantics 必须引用相应语义版本并符合实际属性/形状；未知项不推断成逐元素或任意可融合。ONNX domain/opset 属于来源映射，不能直接替代规范化 OperatorKey 的语义版本。

## TensorSpec 与端口边

| 字段 | 含义 |
|---|---|
| tensor_id | 不可变逻辑值标识 |
| shape、axes、dtype | 逻辑维度、轴含义和元素类型 |
| source_kind | `input`、`parameter` 或 `operator_result` |
| parameter_ref | 参数内容的制品引用；仅参数需要，不内嵌大数组 |
| logical_size_bytes | 稠密张量元素数乘以每元素字节数；需与 shape/dtype 一致 |

模型输出身份由 `model_output_ids` 统一决定。物理 stride、padding、打包、设备位置和 alias 不属于 TensorSpec。稀疏或压缩格式的物理占用属于候选接口中的表示。

内部边为：`(producer_id, output_port, consumer_id, input_port, tensor_id)`。端口从 0 开始，必须满足生产者相应输出与消费者相应输入引用同一 tensor_id。

有序输入输出列表是生产/使用关系的唯一规范来源；边表、producer、uses 和 consumers 都是可重建视图。若序列化缓存边表，验证器必须检查它与输入输出列表完全一致。`Add(U,U)` 应产生输入端口 0、1 两条使用边，但只登记一个 U。

模型输入和参数通过输入端口绑定，不为其强制增加可调度节点。前端必须区分常量与运行时运算，不能为了减少节点把计算结果标成参数。

## Conv–BiasAdd–ReLU 规范例子

本例 Conv 不含 bias。所有张量为 FP32，Conv 采用 FP32 累加。Conv 参数：spatial_dimensions=2、kernel 3×3、stride 1、四边 padding 1、dilation 1、groups 1。

| 值 | shape / 逻辑轴 | 来源或使用 |
|---|---|---|
| X | [1,16,32,32] / NCHW | 模型输入；v1 输入 0 |
| W | [32,16,3,3] / OIHW | 参数；v1 输入 1 |
| b | [32] / 输出通道 | 参数；v2 输入 1 |
| T1 | [1,32,32,32] / NCHW | v1 输出 0；v2 输入 0 |
| T2 | [1,32,32,32] / NCHW | v2 输出 0；v3 输入 0 |
| Y | [1,32,32,32] / NCHW | v3 输出 0；模型输出 |

```text
v1: Conv(X,W) -> T1
v2: BiasAdd(T1,b,channel_axis=1) -> T2
v3: ReLU(T2) -> Y

v1 --T1--> v2 --T2--> v3
```

BiasAdd 按输出通道广播，明确 `T2[n,o,h,w]=T1[n,o,h,w]+b[o]`，不能默认把 [32] 对齐最后一维。若源框架 Conv 已带 bias，导入时应明确语义规范化方式，不能额外再加一次偏置。

T1、T2、Y 各 131072 bytes；X 为 65536 bytes；W 为 18432 bytes；b 为 128 bytes。上述是逻辑大小，不是运行时访问量或分配数量。

## 必须验证的性质

每个算子结果有且只有一个生产端口；外部值没有内部生产者；每个输入引用存在；操作数个数、shape、dtype 和轴约束正确；图无环。数值测试记录只能说明声明测试范围内通过，不能变成对任意输入的等价性证明。
