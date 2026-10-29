# 完整静态索引建模

本轮补齐此前 YOLO11n 的 140 个 unknown 访问关系，以及 ResNet50 的 Flatten、GlobalAveragePool、Gemm。目标是关系可表示、可校验、可在小规模参考用例中解释；不等同于设备 kernel、融合可执行性或成本测量。

## 协议与实现

contract 6.0.0、operator API 5.0.0。索引记录增加取模、整数间接读取、shape 元数据读取、归约后阶段和缩放。C++ 数据布局改变，静态调用方须重新编译，旧 API 插件明确拒绝。算子身份仍不版本化。CLI inspection/3 输出完整关系；operators/2 和 manifest 2.0.0 的格式保持不变。

所有 22 类现有算子提供访问关系。Reshape/Flatten 采用逻辑行主序 flat 坐标，Transpose 采用逆置换，Split 逐输出偏移。Shape 只读取签名元数据，Constant 不读取图输入元素。Gather 的 lookup 必须验证负索引归一化前的值域，关系 Known 不代表未知运行时内容已合法。

Softmax 的 NormalizedExponential 显式描述稳定指数归一化，不能冒充简单求和。GlobalAveragePool 使用 Mean。Gemm 偏置在归约后读取，alpha/beta 不混入 K 次重复加法，空 K 仍有有效输出。新表达式含溢出、除数/模数、端口、dtype、越界与阶段检查。

Transpose、Split、Concat、Slice、Flatten 的静态整数传播也已补齐，能接续 Shape/Reshape 形状链。CLI 展示 resolved permutation、split sizes 和 Shape 已知值。部署 timing 已提供政策时，诊断具体说明事件待 Plan 绑定。

## 验证入口

```bash
./compile.sh
ctest --test-dir build --output-on-failure
./run.sh benchmark/yolo/yolo11n.onnx
./run.sh benchmark/resnet/resnet50.onnx
```

CompleteIndexModeling 用独立的坐标/数值参考验证重排、多输出、标量、空张量、负 Gather、稳定 Softmax、大数、Gemm 缩放与空归约，并拒绝非法模数、lookup 源、越界、归约范围和溢出。FullModelAcceptance 要求两份完整模型所有节点不仅 status=known，还必须提供实际 pieces；继续核对独立 ONNX Runtime 的全部原始输出签名和参数摘要。ONNX 黑盒另覆盖动态索引关系、静态形状重排链、精度保留未知和新计时诊断。

## 仍需外部信息

设备未提供时保持 not_provided；设备能力、实现候选、服务时间、工作区和传输成本需要具体后端或测量，不能由索引已知推断。来源未声明计算/累加精度及误差界时仍为 null，不能用存储 dtype 自动补造。逻辑重排不保证零复制，逻辑字节数不等于峰值分配。

## 本轮实测结果

GCC 11.4.0 / C++17：compile.sh 成功，10/10 CTest 目标通过，42 个 contracts/frontend 公开头逐个独立语法检查通过。现有 locale 子用例仍受本机 locale 缺失限制，不能视为覆盖；完整模型未跳过。改动 C++ 通过 clang-format 18.1.8 检查，未运行 clang-tidy。

| 模型 | 修改前 unknown 关系 | 修改后 known / unknown | 节点 / 张量 / 边 |
|---|---:|---:|---|
| YOLO11n | 140 | 527 / 0 | 527 / 715 / 657 |
| ResNet50 | 56 | 228 / 0 | 228 / 337 / 243 |

实际执行两份 benchmark ONNX 单文件入口及 benchmark 混合目录，10 个文件全部导入成功。原八个 JSON 结构不变，SPPF 仍为 12/17/16。完整模型逐原始输出签名、参数摘要与来源映射继续匹配独立参考；本轮并未新增设备执行器或性能证据。

本轮开始前保存了源文件摘要及工作区状态，完成时核对工作区、暂存区、未跟踪文件和保护文件摘要；治理提示词与原始 benchmark 内容保持不变。


## 命名轴元数据的后续修正

索引关系全部 Known 不表示所有 axes_origin 都已确定。复查发现 YOLO11n 原有 148 个未指定命名轴张量中，87 个偏置向量可从一维展开到唯一非单例轴的等价取模坐标推导通道轴。现在只对该明确保留轴的重排传播身份，不对拆分、合并或单例轴猜测名称；声明冲突仍拒绝。

修正后 YOLO11n 的索引关系仍为 527 Known / 0 Unknown，命名轴元数据为 654 specified / 61 unspecified。后者不表示张量形状、内容摘要或访问关系未知。CLI 增加独立统计，避免把不同含义的 unknown 混在一起。新增一至三维偏置轴断言、轴冲突拒绝和拆轴不猜测反例。
