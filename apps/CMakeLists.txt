# 应用装配目标：生成当前构建已发现的静态工厂表，不维护逐插件名称或具体算子名单。
# LinkedPlugins.cpp 仅固化清单和函数地址；运行时校验通过后才显式调用工厂。
opfusionStaticPlugins("${CMAKE_CURRENT_BINARY_DIR}/LinkedPlugins.cpp" pluginTargets)
add_executable(OpFusion src/Main.cpp src/Plugins.cpp src/Context.cpp "${CMAKE_CURRENT_BINARY_DIR}/LinkedPlugins.cpp")
set_target_properties(OpFusion PROPERTIES OUTPUT_NAME opfusion)
target_include_directories(OpFusion PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# 只链接公开模块入口与私有解析依赖，不向 contracts 注入 JSON/TOML 或 Python。
target_link_libraries(OpFusion PRIVATE
  OpFusion::System OpFusionJson OpFusionToml ${pluginTargets})
target_compile_options(OpFusion PRIVATE -Wall -Wextra -Wpedantic)
if(BUILD_TESTING)
  # Python 仅驱动 C++ CLI 黑盒测试；-I 隔离导入路径，生产目标不依赖解释器。
  # benchmark 缺失由具体用例/subTest 跳过，不让整个 CLI 测试以 77 提前退出。
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME BenchmarkIntegration COMMAND "${Python3_EXECUTABLE}" -I
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/CliTest.py" --binary "$<TARGET_FILE:OpFusion>"
    --benchmark-root "${PROJECT_SOURCE_DIR}/benchmark")
  set(opfusionModelTestPython "${PROJECT_BINARY_DIR}/onnx-test-env/bin/python" CACHE FILEPATH
    "Python environment for ONNX fixtures and full-model reference tests")
  if(NOT EXISTS "${opfusionModelTestPython}")
    set(opfusionModelTestPython "${Python3_EXECUTABLE}")
  endif()
  add_test(NAME OnnxIntegration COMMAND "${opfusionModelTestPython}" -I
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/OnnxTest.py" --binary "$<TARGET_FILE:OpFusion>"
    --root "${PROJECT_SOURCE_DIR}")
  add_test(NAME FullModelAcceptance COMMAND "${opfusionModelTestPython}" -I
    "${PROJECT_SOURCE_DIR}/modules/frontend/tests/FullModels.py"
    --binary "$<TARGET_FILE:OpFusion>" --cache "${PROJECT_BINARY_DIR}/model-cache")
endif()
