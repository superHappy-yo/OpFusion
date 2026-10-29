# frontend — 静态模型来源与规范化

生产实现为 C++17，公共入口 [ModelImporter](include/opfusion/frontend/ModelImporter.h)。只依赖 contracts 公共接口，解析库为私有依赖；不装配设备、不搜索融合、不执行硬件或 Python。

`importFile(path, ModelFormat::Auto)` 按 `.onnx/.json` 选择入口；也可显式指定格式。`importText` 的默认格式仍为兼容 JSON，二进制 ONNX 字节须显式传 `ModelFormat::Onnx`。文件、格式、协议或图校验失败抛出 Error，不交付部分图。

## 正式 ONNX profile

`onnx-static/1` 接受标准空 domain、opset 17、IR 7–13、静态张量形状。保留真实 graph inputs/outputs、initializer、dtype、属性、IR/opset、生产工具及 SHA-256。不要求列出中间 value_info；每个节点按输入签名和类型化算子规则推导，已有声明反向用来核对，不当作推导答案。

支持当前完整 ResNet50/YOLO11n 所需的 22 个语义：Conv、MaxPool、Add、Sub、Mul、Div、ReLU（ONNX Relu）、Sigmoid、Concat、Slice、Resize、MatMul、Constant、Reshape、Transpose、Split、Softmax、Shape、Gather、GlobalAveragePool、Flatten、Gemm。支持边界由各绑定和具体类校验；不是这些算子的所有 ONNX 属性组合均可用。

- Conv/MaxPool 支持一至三维、显式 pads/strides/dilations；Conv 支持 groups。auto_pad 仅 NOTSET，MaxPool 不返回 indices，拒绝 storage_order 非零。
- Conv 可带一维 channel bias，规范化为 Conv + Reshape + Add，保证仅加一次。原始节点位置/名称和端口名称对每个规范节点保留。
- MatMul 支持至少二维矩阵和前导批广播；向量 MatMul 明确拒绝。
- Resize 首版仍为 NCHW nearest/asymmetric/floor、空间 2x；其他组合明确拒绝。
- Slice 仅正 step，依 ONNX 规则归一化/截断静态 starts/ends/axes；负 step 拒绝。
- Reshape 的目标、Split 的 sizes 需可静态求值。Shape、Gather、整数 Add/Sub/Mul/Div、Constant、Reshape、Transpose、Split、Concat、Slice、Flatten 在 C++ 中传播静态形状值；溢出、除零和索引越界报错。节点保留在图中，静态求值/属性折叠有来源说明，不计融合收益。
- 当前 22 类算子的访问关系已实现，包括取模重排、间接 Gather、形状元数据及复合归约；Known 不代表可融合或设备执行支持，运行时 Gather 值域仍须检查。

动态维、控制流、自定义域、函数/训练图、稀疏参数、不支持的 opset/属性、可覆写 initializer 输入均拒绝。

## 参数内容与来源

initializer 按声明区分 Parameter 与运行时 Input，不根据名字猜权重。校验 raw/typed 字段与 dtype/元素数，摘要基于规范小端字节。支持 embedded 与 external-data 的 location/offset/length，拒绝越界、缺文件、路径逃逸、符号链接逃逸、重复键和混合内容。未支持的 external-data 附加字段（包括 checksum）明确拒绝；系统自行记录 SHA-256。参数字节不进入图对象或仓库，仅发布 ParameterRef 与来源范围。

ImportResult 发布只读 DAG、逐规范节点来源、一对多规范化说明、参数摘要、工具及 IR/opset。ONNX 图接口完整；兼容 JSON 的外部输入/终端输出只是依赖推断，modelInputs/modelOutputs 保持未知。物理来源提示不升级为 Buffer、驻留或对齐证明。

## 绑定与契约

公共契约 6.1.0，operator API 5.0.0。算子只有 namespace/name 身份；ONNX opset、插件包、后端和系统契约版本各自保留。

`JsonGraphBindings` 和 `OnnxBindings` 提供同名 `*Binding.h/.cpp`，CMake 自动发现生成显式登记函数。通用读取器通过冻结 Registry 查询，不按具体算子名称分派。SourceBinding 可提供 SourceNormalizer 返回多个真实算子及新张量；额外语义必须已登记。具体算子拥有类型化属性，来源字段树只存在于来源边界。没有万能算子类或全局可变 registry。

## 验证

兼容 JSON 字段见 [BenchmarkJsonV1.md](docs/BenchmarkJsonV1.md)。现有来源回归保留，完整模型准备与签名核对见 [根 README](../../README.md)。FullModelAcceptance 缺模型直接失败，测试仅使用构建缓存，不提交模型或权重。原有 locale 用例仍受本机 locale 安装情况限制，不能把跳过当作覆盖。
