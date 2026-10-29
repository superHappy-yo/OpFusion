# OpFusion

一个 C++17 研究系统，按职责分目录。当前实现逻辑值图、算子语义、静态模型导入、部署声明校验和建模 CLI；尚无设备执行器、性能代价模型、融合搜索或计划生成器。

正式模型入口为 **静态 ONNX，标准 domain、opset 17**；八个 benchmark JSON 保留为兼容测试入口。原图只含逻辑算子、张量值和生产/消费端口；设备、实现、物理表示与运行时缓冲分开。

公共契约 **6.1.0**、operator API **5.0.0**。算子身份只有命名空间和名称，不再含数学语义版本。`Conv`、`MaxPool` 用 `spatial_dimensions=1/2/3` 表达维数，不保留旧二维类型别名。CLI inspection schema 为 `/4`、operators schema 保持 `/2`，插件 manifest schema 为 `2.0.0`；旧插件必须迁移并重新编译。

## 编译与运行

需要 CMake 3.20+、C++17 编译器；开启测试目标时还需 Python 3。直接执行：

```bash
./compile.sh
./run.sh model.onnx
./run.sh --context scenario.json model.onnx
./run.sh models/
./run.sh benchmark/yolo/*.json
```

`compile.sh` 在 `build/` 配置 Release 并编译，产物为 `build/apps/opfusion`，不自动运行测试。首次构建从官方来源准备锁版 Protobuf 31.1 和 Abseil 20250127.0，校验 SHA-256，并从 ONNX 1.20.0 官方 schema 生成 C++ 解析代码；下载失败返回非零，不安装系统包。下载、生成物、模型和权重仅放构建缓存。后续复用缓存，依赖及许可证见 [third_party](third_party/README.md)。

`run.sh` 接受单文件、多个文件、shell 展开的通配符或目录；目录递归扫描 `.onnx` 和兼容图 `.json`，路径排序，排除本次 `--context` 指定文件。路径含空格时加引号。逐文件 JSONL 写标准输出，进度、错误和汇总写标准错误；单文件失败后继续，任一失败则最终非零退出。

```bash
./run.sh benchmark > build/benchmark-results.jsonl
ctest --test-dir build -C Release --output-on-failure
```

脚本可从其他工作目录以绝对路径调用。输入相对调用者工作目录；`OPFUSION_BUILD_DIR` 指定构建目录，`CMAKE_BUILD_PARALLEL_LEVEL` 指定编译并行数（默认 4）。

## 输出含义

每份结果包含显式模型接口（ONNX）、逻辑张量/类型/轴、节点、端口边、已验证属性、数值要求及来源映射。`supports_broadcasting` 是算子能力，`broadcasting.applied/inputs` 是本实例的实际广播端口和输出轴。同形状 Mul 的 `applied` 为 false。

ONNX 来源保留 IR/opset、生产工具和模型摘要；initializer 依据声明标为 Parameter，记录内容摘要和 external-data 范围，运行时输入标为 Input。带 bias 的 Conv 规范化为无 bias Conv、Reshape、Add，每个规范节点保留原节点位置及输入/输出名称。静态形状值由 C++ 计算，保留规范化说明，不记为融合收益。

`logical_axes` 为每个维度提供 `[lower, upper)` 坐标范围、生产者访问表达式及输入来源。`axis_model` 绑定图、生产节点和索引分段；`modeling.axis_models` 汇总结构模型状态。Reshape 可记录轴拆分、合并或一般重索引，Shape 记录元数据读取，常量记录生成关系，标量显式没有轴。YOLO11n 的 715 个张量均已完成结构轴建模。

轴的 `semantic` 是可选命名注释；ONNX 没有声明且无法安全推导时为 null。例如 `[1,256,20,20] → [1,2,128,400]` 可确定通道拆成 `2×128`、空间合并成 `400`，不能据此擅自命名 AttentionHead。`modeling.named_axis_annotations` 单独统计这些可选注释（YOLO11n 有 61 个张量未指定命名）。旧的 `axes_origin` 字段由上述明确分层替代。

`compute_type`、累加类型和误差要求只展示来源声明，未知为 null。`index_semantics` 展示域、地址表达式、读取阶段及归约类型；`modeling.index_relations` 汇总已知/未知数量。当前 22 类内置算子均提供完整的访问关系；Gather 的间接读取附带必须保留的值域检查。形状校验通过不代表设备可执行或融合已合法。context 只提供部署声明，具体计时事件待未来 Plan 绑定；成本保持 null，不虚构 CPU 回退、运行时间或硬件证据。

## 完整模型验收

固定：ResNet50 FP32 `[1,3,224,224]` → `[1,1000]`；YOLO11n FP32 `[1,3,640,640]`、80 类、无 NMS → `[1,84,8400]`，均为静态 opset 17。Python 仅准备测试模型与独立参考，生产导入不调用 Python。

准备独立测试环境，再导出缓存模型：

```bash
python3 -m venv --system-site-packages build/onnx-test-env
build/onnx-test-env/bin/python -m pip install -r modules/frontend/tests/requirements.txt
build/onnx-test-env/bin/python modules/frontend/tests/PrepareModels.py --cache build/model-cache
./compile.sh
ctest --test-dir build --output-on-failure
```

锁版工具为 PyTorch 2.5.1、torchvision 0.20.1、Ultralytics 8.3.0、ONNX 1.20.0；无系统图形库的测试环境使用 OpenCV headless。`PrepareModels.py` 记录导出选项、模型和权重 SHA-256。`FullModelAcceptance` 必须找到两个模型及来源清单，否则失败，不以跳过代替验收。可用 CMake `opfusionModelTestPython` 指向已有测试环境。

完整模型测试对照独立 ONNX Runtime 的所有原始节点输出签名、initializer 及规范化映射；不把参考运行时的运行当作本系统设备执行。其余回归覆盖 ND Conv/MaxPool 的形状、分组、非对称填充、步长、dilation、ceil、数值索引及非法输入，ONNX 外部参数/来源/接口、context 和脚本。八图结构仍合计 264 节点、415 张量、301 条边，SPPF 为 12/17/16。

当前结构轴修复见 [结构轴建模验证](docs/verification/structural-axis-modeling.md)；索引完善记录见 [完整索引建模](docs/verification/complete-index-modeling.md)；此前入口迁移记录见 [ONNX 建模迁移验证](docs/verification/onnx-modeling.md)。

## 代码入口

| 路径 | 职责 |
|---|---|
| [contracts](modules/contracts/README.md) | 纯领域图、算子、表示、设备、代价记录和计划不变量 |
| [frontend](modules/frontend/README.md) | ONNX/兼容 JSON 来源、静态推导和规范化 |
| [apps](apps/README.md) | 唯一装配层、CLI、部署 context 与可选插件 |
| cost_model / fusion / planner / codegen | 后续算法职责；尚无相应生产实现 |
| backends / runtime | 后续设备执行职责；尚无相应生产实现 |
| experiments | 实验职责，生产目录不得反向依赖 |

开始修改前阅读 [AGENTS.md](AGENTS.md)、[C++ 规范](docs/cpp-coding-standard.md) 及目标目录规则。历史规格和旧验证记录不代表当前实现能力。仓库治理提示词、所有权策略及原始 benchmark 不在本次修改范围。
