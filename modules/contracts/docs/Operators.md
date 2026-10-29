# 基础算子语义

基础算子的选定生产实现属于 contracts：公共头在 [operators](../include/opfusion/operators/)，同名源在 [src/operators](../src/operators/)，由 `OpFusionContracts` 提供。不再作为独立基础模块，也不要求内置语义包装成可选插件。旧独立模块及 standard 包的生产文件已删除，见 [清理记录](LegacyCleanup.md)；旧测试源保留，不提供第二套生产定义或兼容壳。

## 公共行为与领域边界

[Operator](../include/opfusion/graph/Operator.h) 保存身份、有序端口和数值要求，仅定义共享行为：

| 方法 | 责任 |
|---|---|
| `validate(inputs)` | 非虚入口：签名数量、静态维度、逻辑轴和数值要求，再调用具体属性钩子 |
| `validateAttributes()` | 具体类端口、属性和支持模式检查；不在父类构造中调用 |
| `infer(inputs)` | 仅据输入签名和自身属性推导输出，不读取声明输出作答案 |
| `attributes(inputs)` | 只读投影自身属性与可由签名确定的属性，不暴露解析库 |
| `broadcasting(inputs)` | 实例广播端口和轴，与能力分开 |
| `evaluateIntegers(inputs,values)` | 静态整数传播，用于形状计算，未知保持 nullopt |
| `traits()` | 独立描述纯性、逐元素、归约和广播；未知保持未知 |
| `indexSemantics(inputs)` | 迭代域、端口访问、归约与填充值，不是 kernel 或融合合法性证明 |
| `semanticIdentity()` | 具体类的静态数学语义身份；不是虚静态方法，也不包含来源工厂 |

DAG 只调用公共协议，不包含具体算子头，不使用名称 switch、dynamic_cast 或中央算子枚举。具体类为 `final : public Operator`，持有类型化只读属性，不依赖 Registry、SourceNode、SourceFields 或任何模型格式。输入须具备已知静态 shape/dtype；未知与不支持明确失败，不默认支持。

## 当前基础类及支持边界

| 类 / 同名头与源 | 数学语义及限制 |
|---|---|
| Conv | 浮点一至三维 NC.../OI... 分组 cross-correlation，无 bias；正 stride/dilation，非负 padding，越界输入填零 |
| Add | 同 dtype、尾轴对齐多向广播；0 与 1 广播得 0 |
| ReLU | 保持 shape/dtype 的逐元素 max(x,0) |
| Sigmoid | 浮点逐元素 1/(1+exp(-x)) |
| Mul | 与 Add 相同的广播，逐元素乘法 |
| Slice | 完整 rank、正 stride、右开 [start,limit)；不归一化负索引或隐式截断 |
| Concat | 同 rank/dtype，非连接维相等，允许合法负 axis |
| Resize | 仅 NCHW nearest/floor/asymmetric 空间 2 倍；其他模式组合明确拒绝 |
| MaxPool | 浮点一至三维，floor/ceil 窗口、dilation、负无穷 padding，不返回 indices |
| MatMul | 浮点 [...,M,K] @ [...,K,N] -> [...,M,N]，前导批维广播；至少二维，K=0 为零和输出 |

数学身份为 `opfusion:<Name>`，不含语义版本。逻辑轴顺序是 AxisOrder，候选物理排列是独立 Layout；不会根据 tensor 名称或来源 stride 转置 shape。Resize/Pool 的固定政策同样为 enum class。

索引表达式为 `offset + sum(coefficient * (floor(iterator[axis] / divisor) % modulus)) + lookups`（modulus 可省略），域半开、迭代器非负、除数为正。Concat 分段互斥，Resize 表达 floor-div，Conv/Pool 显式表达窗口与归约。Sum 初值为零、Maximum 初值为浮点负无穷，输出域不因空归约消失。Unknown/Unsupported 须附原因，不能带伪造的部分关系。

## 增加领域类：不修改中央名单

仅在任务已授权相应路径、公共契约能表达新语义的前提下：

1. 在 `include/opfusion/operators/Foo.h` 与 `src/operators/Foo.cpp` 定义 `opfusion::operators::Foo`。
2. 继承 Operator，实现公共行为及自身 `static SemanticIdentity semanticIdentity()`。
3. 具体属性由自身类持有；构造完成成员初始化后检查属性，推导仅使用输入。
4. 重新配置/构建时，CMake 按文件约定生成 include、派生关系、非抽象、身份类型检查和 `registry.add` 调用；无需手改 apps 或中央 include/名称列表。

[Operators.cmake](../../../cmake/Operators.cmake) 定义发现约定，[BuiltinOperators.h](../include/opfusion/registry/BuiltinOperators.h) 提供通用装配入口。没有静态初始化自注册、全局可变 Registry 或 Python 生产生成器。

该路径只增加语义建模能力。来源格式、ONNX 规范化、kernel、lowering 和设备候选不会因继承自动出现；新格式映射另提供 frontend 绑定，可选扩展仍遵守 extensions 的独立任务边界。

## 来源绑定独立于领域类

[frontend/JsonGraphBindings](../../frontend/src/JsonGraphBindings/) 保存九个现有 `benchmark-json/1` 映射，MatMul 没有该映射：

- ConvolutionFwdAttributes → Conv：来源 dilation=1，stride=1/2，pad=0/1。
- PointwiseAttributes/add、relu_fwd、sigmoid_fwd、mul → 各自具体类。
- SliceAttributes → Slice；ConcatenateAttributes → Concat：来源 axis=1、2–4 输入、无原地覆盖。
- ResizeAttributes → 受限 Resize；ResampleFwdAttributes/maxpool → MaxPool。

每个绑定文件以 `make<Name>Binding()` 自由函数返回 SourceBinding，不新增静态工厂继承层。严格字段检查保留在 frontend，UID 与端口顺序不丢失；来源 `compute_data_type=float` 显式要求 Float32 计算；累加精度未声明，保持未知。详见 [来源格式](../../frontend/docs/BenchmarkJsonV1.md)。

## 新增完整模型语义

Constant、Reshape、Transpose、Split、Softmax、Shape（C++ 类 ShapeOf）、Gather、Sub、Div、GlobalAveragePool、Flatten、Gemm 各有类型化属性与独立同名头源。它们覆盖锁定完整模型的静态形状与数值类型建模；访问关系已补齐，仍不冒充 kernel 或可融合实现。ONNX 绑定见 frontend/OnnxBindings，通用读取器不维护名称 switch。

`attributes(inputs)` 返回具体类验证属性的展示副本；Conv 从只读权重签名展示 kernel_shape，不把内部属性换成任意键值袋。`broadcasting(inputs)` 描述本次发生广播的端口/输出轴，与 traits.broadcasting 能力分开。`evaluateIntegers` 只传播已证明的静态整数内容，缺失返回 nullopt，溢出和非法索引报错。

## 版本与验证限制

contractVersion=6.1.0、operatorApiVersion=5.0.0。删除算子语义版本、旧二维类型与冗余 CLI semantic 字段；不提供旧类型别名。ONNX opset、插件包、后端及契约仍各自版本化。CLI inspection /4、operators /2、manifest 2.0.0，全部仓库调用方迁移并重新编译。

保留原有领域和兼容 JSON 回归；新增 SpatialModeling 独立数值索引、ONNX/context/脚本及完整模型验收。完整命令和已知限制见根 README；没有设备执行证据。逗号小数 locale 子用例仍受本机安装条件限制。

## 完整索引关系（contract 6 / operator API 5）

- Reshape、Flatten 用逻辑行主序 flat 域与商/余数映射到各轴；不推断物理连续性、别名或零拷贝。
- Transpose 反置 permutation 建立输入读取，Split 按每个输出域加输入偏移，空输出域仍有明确关系。
- Constant 无输入元素读取；Shape 通过 shapeReads 读取签名的维度元数据，不扫描输入元素。
- Gather 用 IndexLookup 显式表示整数输入决定的地址。约束为 `-extent <= index < extent`，负索引加 extent；未知运行时内容不妨碍关系已知，但执行者必须保留检查。
- Softmax 使用 NormalizedExponential，区分全行归约读取与归约后分子读取，定义稳定的减最大值指数归一化；不能用普通 Sum 伪装。
- GlobalAveragePool 使用 Mean，非空归约；Gemm 用 Sum、alpha 缩放和归约后 beta*C，偏置不随 K 重复累加。归约为空时仍读取/写出有效的归约后部分。

验证器检查取模/lookup 边界、整数源类型、溢出、归约阶段和范围。evaluateIndex 是纯整数表达式求值器，不是 tensor kernel。CompleteIndexModeling 用独立坐标/数值参考及非法关系反例验证协议；完整模型测试要求每个节点有实际 pieces，不能只把状态改为 Known。


结构轴公共查询见 [AxisModel.h](../include/opfusion/graph/AxisModel.h)。`buildAxisModels(DAG)` 从静态签名和已验证索引关系构造坐标范围及逐 piece 来源，记录直接、拆分、合并、重索引、间接读取、元数据和生成关系；不按算子或张量名称分派。`TensorAxisModel::validate` 核对图/registry、端口、维度、表达式与结构来源，拒绝篡改。未知提供者的索引关系仍保持 Unknown，不以签名齐全冒充关系已知。标量无轴，命名轴为独立可选注释，既有 Tensor 命名约束不变。新增接口令 contract 升至 6.1.0，operator API 保持 5.0.0；CLI inspection/4 改变轴输出格式，调用方需要迁移。
