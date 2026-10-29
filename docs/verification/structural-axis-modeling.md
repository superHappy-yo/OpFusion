# 61 个缺失轴项：结构化建模验证

本记录对应 YOLO11n 检查输出中的 61 个 `axes_origin=unknown`。这些张量的静态 shape 与算子索引关系已有定义，但原命名轴模型不能表达通道拆分、空间合并、常量、形状向量等结构。仅补 N/C/H/W 名称会产生错误约束，尤其不能由张量名称猜测注意力或检测业务含义。

## 改动与契约

contracts 新增 [AxisModel.h](../../modules/contracts/include/opfusion/graph/AxisModel.h) 及其纯 C++17 实现：从已验证签名和生产者索引关系构造结构轴视图，逐轴记录半开范围、坐标表达式、读取端口/输入维度及所属关系分段。结构包含直接映射、拆分、合并、一般重索引、间接读取、形状元数据及生成关系。标量显式没有轴。生产者关系未知时仍返回 Unknown 及原因。

来源视图可通过 `TensorAxisModel::validate` 核验，篡改范围、坐标、来源张量、结构分类、分段或 registry 绑定均被拒绝。查询不按算子或张量名称分派，不修改既有 Tensor 命名轴约束，不推断物理布局。

公共 contract 由 6.0.0 升至 6.1.0，新增独立查询与记录，不改变 Operator 虚接口，operator API 保持 5.0.0。CLI inspection 由 /3 升至 /4：`logical_axes` 改为结构坐标记录，旧 `axes_origin` 删除；`axis_model` 展示生产者和来源分段，`semantic` 与 `semantic_annotation_origin` 只承载可选命名。消费旧格式的调用方须迁移并重新编译。manifest 保持 2.0.0、operators 输出保持 /2，版本相互独立。

修改范围为 contracts 结构轴接口、实现、版本、测试和说明，apps 的 CLI 序列化及回归，frontend 的完整模型验收与版本说明，以及根 README 和本验证记录。治理提示词、所有权规则、构建依赖、模型和原始 benchmark JSON 没有因本次修复改变。

## 实际验证

执行 `./compile.sh` 成功；执行 `ctest --test-dir build -C Release --output-on-failure`，10/10 成功，两个完整模型没有跳过。41 个 contracts 公共头、2 个 frontend 公共头分别通过 C++17 独立编译；本次 C++ 文件通过 clang-format 18.1.8 检查。

执行 `./run.sh benchmark/yolo/yolo11n.onnx`，以及 `./run.sh benchmark`，目录内 10/10 文件成功。归一化图规模保持：

| 模型 | 节点 | 张量 | 端口边 | 已知结构轴模型 | 未知结构轴模型 |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 527 | 715 | 657 | 715 | 0 |
| ResNet50 | 228 | 337 | 243 | 337 | 0 |
| 八个兼容 JSON 合计 | 264 | 415 | 301 | 415 | 0 |
| SPPF（属于八图） | 12 | 17 | 16 | 17 | 0 |

YOLO11n 527 个节点的索引关系均 Known。注意力 `[1,256,20,20] → [1,2,128,400]` 的两维来源明确为输入通道轴拆分，最后一维为输入高、宽两轴合并；回归独立枚举扁平坐标边界验证表达式，没有为拆分维编造 Head 等名称。一般非整除重排 `[6,10] → [4,15]` 保持 Reindexed。新增 ONNX 黑盒用例覆盖生成常量、标量、Shape 元数据、运行时 Gather 间接来源，以及 Reshape 的真实双输入端口。

完整模型验收继续与独立 ONNX Runtime 对照全部原节点输出签名、参数摘要与 Conv bias 一对多来源映射，并核对每个轴来源的端口、坐标及生产者索引分段。测试提供的未知关系不会因为 shape 齐全而升级 Known，伪造 Known 记录校验失败。

## 输出解释与限制

`modeling.axis_models` 为结构完整性；`modeling.named_axis_annotations` 为可选命名覆盖。YOLO11n 仍有 61 个张量未指定命名注释，此事实没有被隐藏，但这些张量均具有完整坐标模型；其 `semantic` 为 null。全文件的字符串值 `unknown` 为零，并不意味着来源没有声明的所有信息都已获得。

来源未声明的 compute/accumulation 类型和误差要求继续为 null；未配置部署、没有实现及测量证据时成本保持未知。本次验证没有执行 OpFusion 硬件后端，不证明融合合法性、设备执行或性能。

可复查输出：

```bash
./compile.sh
./run.sh benchmark/yolo/yolo11n.onnx > build/yolo11n-inspection.jsonl
```
