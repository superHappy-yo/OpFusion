# C++17 生产目标的本地装配入口；只使用仓库已有解析头，不下载或探测设备。
cmake_minimum_required(VERSION 3.20)
project(OpFusion VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)

# 解析库仅作为 frontend/apps 的边界依赖；contracts 不链接解析库。
add_library(OpFusionJson INTERFACE)
target_include_directories(OpFusionJson SYSTEM INTERFACE
  "${PROJECT_SOURCE_DIR}/third_party" "${PROJECT_SOURCE_DIR}/third_party/nlohmann")
add_library(OpFusionToml INTERFACE)
target_include_directories(OpFusionToml SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/tomlplusplus")

include(cmake/OperatorPlugins.cmake)
include(cmake/Operators.cmake)
include(cmake/SourceBindings.cmake)
include(cmake/OnnxDependencies.cmake)
add_subdirectory(modules/contracts)
add_subdirectory(modules/frontend)

# 单一应用系统的链接入口；内部功能目录仍按各自职责编译，不创建空算法目标。
# 只有 apps/集成调用方使用此聚合入口，领域和算法目录不反向依赖系统装配。
add_library(OpFusionSystem INTERFACE)
add_library(OpFusion::System ALIAS OpFusionSystem)
target_link_libraries(OpFusionSystem INTERFACE OpFusionContracts OpFusionFrontend)

# 可选静态包按 manifest 目录通用发现，默认启用，不按旧包名称做例外。
option(opfusionEnableOperatorPlugins "Build optional static operator plugin packages" ON)
if(opfusionEnableOperatorPlugins)
  file(GLOB operatorManifests CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/extensions/operators/*/plugin.toml")
  list(SORT operatorManifests)
  foreach(manifest IN LISTS operatorManifests)
    get_filename_component(pluginDirectory "${manifest}" DIRECTORY)
    get_filename_component(pluginId "${pluginDirectory}" NAME)
    if(NOT pluginId MATCHES "^[a-z][a-z0-9_.-]*$")
      message(FATAL_ERROR "Invalid plugin directory: ${pluginId}")
    endif()
    if(NOT EXISTS "${pluginDirectory}/CMakeLists.txt")
      message(FATAL_ERROR "Operator plugin has no C++ build entry: ${pluginId}")
    endif()
    add_subdirectory("${pluginDirectory}" "extensions/operators/${pluginId}")
  endforeach()
endif()
add_subdirectory(apps)
