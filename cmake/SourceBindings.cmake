# 仅发现 Binding 贡献的同名头源；SourceFields 等辅助文件不自动注册。
# 生成代码留在构建目录，来源工厂仍由 apps 显式装配。
function(opfusionBuiltinSourceBindings target output)
  foreach(group IN ITEMS JsonGraphBindings OnnxBindings)
  if(group STREQUAL "JsonGraphBindings")
    set(ns jsonGraph)
  else()
    set(ns onnxGraph)
  endif()
  file(GLOB headers CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/${group}/*Binding.h")
  list(SORT headers)
  if(NOT DEFINED includes)
  set(includes
    "#include \"opfusion/frontend/BuiltinSourceBindings.h\"\n#include <type_traits>\n")
  set(checks "")
  set(calls "")
  endif()
  foreach(header IN LISTS headers)
    get_filename_component(name "${header}" NAME_WE)
    if(NOT name MATCHES "^[A-Z][A-Za-z0-9]*Binding$" OR
        NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${group}/${name}.cpp")
      message(FATAL_ERROR
        "Each source binding needs matching PascalCase header/source files: ${name}")
    endif()
    string(APPEND includes "#include \"${group}/${name}.h\"\n")
    string(APPEND checks
      "static_assert(std::is_same_v<decltype(${ns}::make${name}()), SourceBinding>);\n")
    string(APPEND calls "  registry.bind(${ns}::make${name}());\n")
  endforeach()
  endforeach()
  opfusionContentIdentity("${target}" "${CMAKE_CURRENT_SOURCE_DIR}" digest)
  file(WRITE "${output}"
    "${includes}\nnamespace opfusion::frontend {\n${checks}\n"
    "void registerBuiltinSourceBindings(Registry& registry) {\n${calls}}\n"
    "const char* builtinSourceBindingsDigest() noexcept { return \"${digest}\"; }\n}\n")
endfunction()
