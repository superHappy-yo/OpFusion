# 锁版原生 ONNX 解析依赖；下载和生成物仅写构建目录，不安装系统包。
include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Use the host C++ standard" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "No upstream tests" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "No system installation" FORCE)
FetchContent_Declare(absl
  URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20250127.0.tar.gz
  URL_HASH SHA256=16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8
  DOWNLOAD_DIR "${PROJECT_BINARY_DIR}/dependency-cache"
  DOWNLOAD_NAME abseil-20250127.0.tar.gz)
FetchContent_MakeAvailable(absl)
set_property(DIRECTORY "${absl_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "No upstream tests" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "No system installation" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Static native parser" FORCE)
set(protobuf_BUILD_LIBUPB ON CACHE BOOL "Required by the official protoc build" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "No extra compression dependency" FORCE)
FetchContent_Declare(protobuf
  URL https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protobuf-31.1.tar.gz
  URL_HASH SHA256=12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82
  DOWNLOAD_DIR "${PROJECT_BINARY_DIR}/dependency-cache"
  DOWNLOAD_NAME protobuf-31.1.tar.gz)
FetchContent_MakeAvailable(protobuf)
set_property(DIRECTORY "${protobuf_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

set(onnxGenerated "${PROJECT_BINARY_DIR}/generated/onnx")
file(MAKE_DIRECTORY "${onnxGenerated}")
add_custom_command(OUTPUT "${onnxGenerated}/onnx.pb.cc" "${onnxGenerated}/onnx.pb.h"
  COMMAND $<TARGET_FILE:protobuf::protoc>
    --cpp_out=${onnxGenerated} --proto_path=${PROJECT_SOURCE_DIR}/third_party/onnx
    "${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto"
  DEPENDS protobuf::protoc "${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto"
  VERBATIM)
add_library(OpFusionOnnxProto STATIC "${onnxGenerated}/onnx.pb.cc")
target_include_directories(OpFusionOnnxProto SYSTEM PUBLIC "${onnxGenerated}")
target_link_libraries(OpFusionOnnxProto PUBLIC protobuf::libprotobuf)
add_library(OpFusionSha256 INTERFACE)
target_include_directories(OpFusionSha256 SYSTEM INTERFACE
  "${PROJECT_SOURCE_DIR}/third_party/picosha2")
