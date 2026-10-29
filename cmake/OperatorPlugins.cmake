# 内容身份覆盖路径、源码、公共依赖与构建条件；移动或补注释也会改变摘要。
# 只读取包内文件及声明的依赖，不把旧二进制身份用于新源码。
function(opfusionContentIdentity target directory output)
  file(GLOB_RECURSE content LIST_DIRECTORIES false CONFIGURE_DEPENDS "${directory}/*")
  foreach(path IN LISTS content)
    file(REAL_PATH "${path}" realPath)
    cmake_path(IS_PREFIX directory "${realPath}" NORMALIZE inside)
    if(NOT inside)
      message(FATAL_ERROR "Operator content escapes its package: ${path}")
    endif()
  endforeach()
  file(GLOB_RECURSE modelDependencies LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/modules/contracts/include/*"
    "${PROJECT_SOURCE_DIR}/modules/contracts/src/*"
    "${PROJECT_SOURCE_DIR}/modules/frontend/include/*"
    "${PROJECT_SOURCE_DIR}/modules/frontend/src/*"
    "${PROJECT_SOURCE_DIR}/apps/src/*")
  list(APPEND content ${modelDependencies}
    "${PROJECT_SOURCE_DIR}/modules/contracts/CMakeLists.txt"
    "${PROJECT_SOURCE_DIR}/modules/frontend/CMakeLists.txt"
    "${PROJECT_SOURCE_DIR}/apps/CMakeLists.txt"
    "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
    "${PROJECT_SOURCE_DIR}/cmake/Operators.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/SourceBindings.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/OnnxDependencies.cmake"
    "${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto"
    "${PROJECT_SOURCE_DIR}/third_party/picosha2/picosha2.h"
    "${PROJECT_SOURCE_DIR}/cmake/OperatorPlugins.cmake"
    "${PROJECT_SOURCE_DIR}/third_party/nlohmann/json.hpp"
    "${PROJECT_SOURCE_DIR}/third_party/tomlplusplus/toml.hpp")
  list(REMOVE_DUPLICATES content)
  list(SORT content)
  set(identity "cpp-operators/4\n")
  foreach(path IN LISTS content)
    file(SHA256 "${path}" digest)
    file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${path}")
    string(APPEND identity "${relative}:${digest}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
  endforeach()
  get_target_property(definitions "${target}" COMPILE_DEFINITIONS)
  get_target_property(options "${target}" COMPILE_OPTIONS)
  string(APPEND identity "${CMAKE_CXX_COMPILER}:${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}:${CMAKE_CXX_STANDARD}:${CMAKE_BUILD_TYPE}:${CMAKE_CXX_FLAGS}:${CMAKE_CXX_FLAGS_DEBUG}:${CMAKE_CXX_FLAGS_RELEASE}:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}:${CMAKE_CXX_FLAGS_MINSIZEREL}:${definitions}:${options}")
  string(SHA256 digest "${identity}")
  set("${output}" "${digest}" PARENT_SCOPE)
endfunction()

# 包只声明自己的静态工厂及摘要；实际版本/贡献冲突由装配阶段统一校验。
function(opfusionOperatorPlugin target factory)
  if(NOT TARGET "${target}" OR NOT factory MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "Invalid static operator plugin declaration")
  endif()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin.toml")
    message(FATAL_ERROR "Operator plugin requires plugin.toml")
  endif()
  opfusionContentIdentity("${target}" "${CMAKE_CURRENT_SOURCE_DIR}" digest)
  set_property(GLOBAL APPEND PROPERTY OpFusionPluginRecords
    "${target}|${factory}|${CMAKE_CURRENT_SOURCE_DIR}|${digest}")
endfunction()

# 转义 manifest 和文件清单后嵌入 C++ 字面量，不能把包文本当作源代码执行。
function(opfusionCppString out text)
  string(REPLACE "\\" "\\\\" escaped "${text}")
  string(REPLACE "\"" "\\\"" escaped "${escaped}")
  string(REPLACE "\n" "\\n" escaped "${escaped}")
  string(REPLACE "\r" "\\r" escaped "${escaped}")
  string(REPLACE "\t" "\\t" escaped "${escaped}")
  set("${out}" "\"${escaped}\"" PARENT_SCOPE)
endfunction()

# 生成已链接包的记录与工厂引用；运行期仅核对这些包，不是动态 SO 加载器。
function(opfusionStaticPlugins output targetsOut)
  get_property(records GLOBAL PROPERTY OpFusionPluginRecords)
  set(declarations "")
  set(entries "")
  set(targets "")
  foreach(record IN LISTS records)
    string(REPLACE "|" ";" fields "${record}")
    list(GET fields 0 target)
    list(GET fields 1 factory)
    list(GET fields 2 directory)
    list(GET fields 3 digest)
    file(READ "${directory}/plugin.toml" manifest)
    get_filename_component(id "${directory}" NAME)
    opfusionCppString(quoted "${manifest}")
    file(GLOB_RECURSE files RELATIVE "${directory}" CONFIGURE_DEPENDS "${directory}/*")
    string(JOIN "\n" inventory ${files})
    opfusionCppString(quotedFiles "${inventory}")
    string(APPEND declarations "extern void ${factory}(opfusion::Registry&);\n")
    string(APPEND entries "  {\"${id}\", ${quoted}, \"${digest}\", \"${factory}\", ${quotedFiles}, &${factory}},\n")
    list(APPEND targets "${target}")
  endforeach()
  file(WRITE "${output}" "#include \"LinkedPlugins.h\"\n${declarations}\nnamespace opfusion::app {\nstd::vector<LinkedPlugin> linkedPlugins() { return {\n${entries}}; }\n}\n")
  set("${targetsOut}" "${targets}" PARENT_SCOPE)
endfunction()
