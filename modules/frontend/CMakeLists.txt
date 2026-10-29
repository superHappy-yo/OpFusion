# frontend 只编译统一入口、内部 JSON 读取器和来源绑定，不装配设备或外部插件。
# 同名 *Binding.h/.cpp 由下方生成器发现并校验；SourceFields.cpp 只是共享字段校验。
file(GLOB bindingSources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/JsonGraphBindings/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/OnnxBindings/*.cpp")
add_library(OpFusionFrontend
  src/ModelImporter.cpp
  src/JsonGraphImporter.cpp
  src/OnnxImporter.cpp
  ${bindingSources})
target_compile_features(OpFusionFrontend PUBLIC cxx_std_17)
# 公共头只暴露 contracts；内部读取器/绑定头不成为 apps 的 include 接口。
target_include_directories(OpFusionFrontend
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# JSON 库只供私有解析使用，公共接口不泄漏其类型或为 contracts 添加解析依赖。
target_link_libraries(OpFusionFrontend
  PUBLIC OpFusionContracts
  PRIVATE OpFusionJson OpFusionOnnxProto OpFusionSha256)
target_compile_options(OpFusionFrontend PRIVATE -Wall -Wextra -Wpedantic)
# 生成装配函数而非静态初始化注册；apps 必须先登记语义，再显式调用绑定入口。
# 产物只写构建目录，模块的 JsonGraphBindings 结构与逐算子源文件保持不动。
set(bindingSource "${CMAKE_CURRENT_BINARY_DIR}/BuiltinSourceBindings.cpp")
opfusionBuiltinSourceBindings(OpFusionFrontend "${bindingSource}")
target_sources(OpFusionFrontend PRIVATE "${bindingSource}")

if(BUILD_TESTING)
  # 中立夹具及真实来源绑定回归只验证逻辑导入/失败边界，不宣称数值或设备验收。
  # 固定使用可分享的中立 fixture；来源绑定用例在同一目标中独立装配公共语义和绑定。
  add_executable(OpFusionFrontendTests tests/BenchmarkTest.cpp)
  target_link_libraries(OpFusionFrontendTests
    PRIVATE OpFusionFrontend OpFusionJson)
  add_test(NAME FrontendBenchmarkJson COMMAND OpFusionFrontendTests
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/Projection.json)
endif()
