# ONNX 与算子建模迁移验证

日期：2026-10-29。范围：contracts、frontend、apps、构建脚本、原生解析依赖、相关规格与测试。基线已有 C++ 建模 CLI 和六个 CTest 目标；不是从“无源码”状态开始。保留用户已有工作区，不修改治理提示词、所有权策略和原始八个 benchmark。

## 变更与兼容

contract 5.0.0、operator API 4.0.0、CLI inspection/operators /2、manifest 2.0.0。算子身份由 namespace/name 决定，删除 semantic_version 与二维类型别名。旧调用方和插件须迁移重编译；旧 manifest、API 和 semantic_version 字段分别有拒绝用例。

Conv/MaxPool 支持 1–3 维静态形状及空间访问；MatMul 支持前导批广播。CLI 通用属性接口展示 kernel/stride/dilation/pads/groups 等，计算类型、累加类型及误差未声明时为 null，同形状 Mul 不再展示为实际广播。新增完整模型算子的部分索引 lowering 明确为 Unknown；不能以导入成功推断融合或设备支持。

ONNX 使用原生 C++ Protobuf 解析，正式子集为标准 domain、opset 17、静态形状。initializer、完整 IO、内嵌/外部参数内容与摘要、静态整数形状计算、Conv bias 一对多规范化均有回归。重复节点名不覆盖值；外部参数内容改变会改变图内容身份；已知 Gather 索引越界即使数据未知也拒绝。

context 使用张量名称装配公共 Platform/Representation/Boundary/TimingPolicy。物理/动态容量、引用和计时冲突拒绝；计时仅声明口径，事件待未来 Plan 绑定。声明设备不升级为执行能力，成本保持未知。

## 实际验证

工具链：Linux x86-64、GCC 11.4.0、C++17、CMake/Ninja。执行过：

```bash
./compile.sh
ctest --test-dir build --output-on-failure
./run.sh ./benchmark/yolo/yolo11-sppf.json
./run.sh build/model-cache/resnet50.onnx build/model-cache/yolo11n.onnx
./run.sh benchmark
```

九个 CTest 目标全部通过：SpatialModeling、ContractsCore、ConditionalModelAndPlan、DeviceResourceContracts、OpFusionStandardTests、FrontendBenchmarkJson、BenchmarkIntegration、OnnxIntegration、FullModelAcceptance。原有逗号小数 locale 子用例因本机缺少该 locale 仍报告 SKIP，不作为 locale 覆盖；完整模型没有跳过。

另在全新临时目录只预置锁定依赖归档、将所有 HTTP/HTTPS 代理指向不可用地址，实际执行 compile.sh 完整构建；最终增量离线重编译和该目录的全部九项测试也通过。模型参考使用共享的构建缓存，生产依赖没有借用主目录编译产物。

42 个 contracts/frontend 公开头逐个通过 C++17 独立语法检查。改动 C++ 使用 clang-format 18.1.8 校验；不声称运行过 clang-tidy。脚本语法、工作区/暂存区 diff、保护文件摘要和未跟踪文件范围均检查。另以空模型缓存直接运行完整模型验收，确认返回失败。

## 完整模型

使用测试准备脚本，版本为 PyTorch 2.5.1、torchvision 0.20.1、Ultralytics 8.3.0、ONNX 1.20.0，静态 FP32、opset 17、batch 1；YOLO 80 类，无 NMS。每份模型与权重摘要、导出选项记录在构建缓存 `.provenance.json`，不提交模型、权重或原始参考输出。

| 模型 | 输入 | 输出 | 规范化节点/张量/边 |
|---|---|---|---|
| ResNet50 | [1,3,224,224] | [1,1000] | 228 / 337 / 243 |
| YOLO11n | [1,3,640,640] | [1,84,8400] | 527 / 715 / 657 |

FullModelAcceptance 对所有原始节点输出的形状/类型逐项对照独立 ONNX Runtime，另核对全部 initializer 的角色、形状、dtype、内容摘要以及每个 Conv bias 的来源与规范化输出。参考运行时仅供测试，不是 OpFusion 执行器。

八个 JSON 图仍为 264 节点、415 张量、301 条边；SPPF 仍为 12 节点、17 张量、16 条边。单文件、混合目录、通配符、空格路径、context 排除及失败后继续有黑盒回归。

## 保留边界

没有硬件执行、设备能力验证、融合合法性/收益证明、代价预测或计划生成。external-data 的附加 checksum 字段尚不支持，显式拒绝并自行计算 SHA-256；其他来源限制见 frontend README。完整模型验收只证明锁定模型在当前子集内完整导入、规范化和静态建模一致。
