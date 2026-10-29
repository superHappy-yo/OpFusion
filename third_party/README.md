# 原生解析依赖

contracts 仅依赖 C++ 标准库；格式库只在 frontend/apps 内使用。系统保持 C++17，不安装系统包，不在生产路径调用 Python。

| 依赖 | 锁定版本 | 内容 SHA-256 | 许可证 |
|---|---|---|---|
| nlohmann/json | 3.11.3 | `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6` | [MIT](nlohmann/LICENSE.MIT) |
| toml++ | 3.4.0 | `6b5172ad4dd6519aec67b919181fa7a38a2234131e5b2afa232dfe444819783e` | [MIT](tomlplusplus/LICENSE) |
| ONNX schema | 1.20.0 | `470e64dfc5338477d3adc1853f5875618f70cc698306d9bed1232305680b121f` | [Apache 2.0](onnx/LICENSE) |
| PicoSHA2 header | 1.0.1 | `b13c180161ffac8d0adc81e033e493c409457c4d1258ab9781ac80579ba3bdd8` | [MIT](picosha2/LICENSE) |
| Protobuf source archive | 31.1 | `12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82` | 源码包内 LICENSE |
| Abseil source archive | 20250127.0 | `16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8` | 源码包内 LICENSE |

前四项未改上游内容，随仓库保留许可证。Protobuf/Abseil 通过 [OnnxDependencies.cmake](../cmake/OnnxDependencies.cmake) 从官方源下载，校验摘要并缓存于构建目录 `dependency-cache/`；UPB、utf8_range 使用该 Protobuf 发行包自带内容，没有额外浮动下载。构建本地 protoc 并从官方 schema 生成解析代码，生成物仅写构建目录；不使用系统 protobuf 或 Python onnx 作为生产解析器。

首次网络或摘要校验失败，CMake 明确失败；以后复用缓存。保留整个构建目录可离线重编译；新构建目录也可预置相同 `dependency-cache` 归档。删除缓存后重新准备需要网络。

上游来源：[JSON](https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp)、[TOML](https://raw.githubusercontent.com/marzer/tomlplusplus/v3.4.0/toml.hpp)、[ONNX schema](https://raw.githubusercontent.com/onnx/onnx/v1.20.0/onnx/onnx.proto3)、[PicoSHA2](https://github.com/okdshin/PicoSHA2/tree/v1.0.1)、[Protobuf](https://github.com/protocolbuffers/protobuf/releases/tag/v31.1)、[Abseil](https://github.com/abseil/abseil-cpp/releases/tag/20250127.0)。

Python 的锁版模型导出/独立参考仅供测试，见 [requirements.txt](../modules/frontend/tests/requirements.txt)。模型、权重、原始参考输出均不随源码提交。
