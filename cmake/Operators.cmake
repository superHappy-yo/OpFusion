# 从具体类的同名头源生成语义登记，新增算子不需要中央名单。
function(opfusionBuiltinOperators target output)
  file(GLOB headers CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opfusion/operators/*.h")
  list(SORT headers)
  if(NOT headers)
    message(FATAL_ERROR "The builtin operator library has no concrete classes")
  endif()
  set(includes "#include \"opfusion/registry/BuiltinOperators.h\"\n#include <type_traits>\n")
  set(checks "")
  set(calls "")
  foreach(header IN LISTS headers)
    get_filename_component(name "${header}" NAME_WE)
    if(NOT name MATCHES "^[A-Z][A-Za-z0-9]*$" OR
        NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/${name}.cpp")
      message(FATAL_ERROR
        "Each concrete operator needs matching PascalCase header/source files: ${name}")
    endif()
    string(APPEND includes "#include \"opfusion/operators/${name}.h\"\n")
    string(APPEND checks
      "static_assert(std::is_base_of_v<Operator, operators::${name}>);\n"
      "static_assert(!std::is_abstract_v<operators::${name}>);\n"
      "static_assert(std::is_same_v<decltype(operators::${name}::semanticIdentity()), SemanticIdentity>);\n")
    string(APPEND calls "  registry.add({operators::${name}::semanticIdentity()});\n")
  endforeach()
  # 编译期协议检查和显式登记共同生成，摘要不是运行或数值验证证据。
  opfusionContentIdentity("${target}" "${CMAKE_CURRENT_SOURCE_DIR}" digest)
  file(WRITE "${output}"
    "${includes}\nnamespace opfusion {\n${checks}\n"
    "void registerBuiltinOperators(Registry& registry) {\n${calls}}\n"
    "const char* builtinOperatorsDigest() noexcept { return \"${digest}\"; }\n}\n")
endfunction()
