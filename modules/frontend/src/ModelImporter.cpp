#include "opfusion/frontend/ModelImporter.h"

#include "JsonGraphImporter.h"
#include "OnnxImporter.h"
#include "opfusion/common/Error.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <utility>

namespace opfusion::frontend {

// 入口只持有装配层交付的目录快照，不补注册、不访问全局 Registry 或探测设备。
ModelImporter::ModelImporter(Registry registry) : mRegistry(std::move(registry)) {
  require(mRegistry.frozen(), "model_registry_unfrozen", "model import requires a frozen registry");
}

ImportResult ModelImporter::importFile(const std::string &path, ModelFormat format) const {
  require(format == ModelFormat::Auto || format == ModelFormat::JsonGraph ||
              format == ModelFormat::Onnx,
          "model_format_unsupported", path + ": unsupported model format");
  std::ifstream input(path, std::ios::binary);
  require(input.is_open(), "model_file_open", path + ": cannot open model file");
  std::string text;
  try {
    text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  } catch (const std::ios_base::failure &error) {
    // 读取目录等路径时，流迭代器可能直接抛异常；与 badbit 路径统一为领域读取错误。
    throw Error("model_file_read", path + ": cannot read model file: " + error.what());
  }
  require(!input.bad(), "model_file_read", path + ": cannot read model file");
  if (format == ModelFormat::Auto) {
    auto extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    require(extension == ".onnx" || extension == ".json", "model_format_unsupported",
            "model suffix must be .onnx or .json");
    format = extension == ".onnx" ? ModelFormat::Onnx : ModelFormat::JsonGraph;
  }
  return importText(text, format, path);
}

ImportResult ModelImporter::importText(const std::string &text, ModelFormat format,
                                       const std::string &sourceName) const {
  switch (format) {
  case ModelFormat::Onnx:
    return detail::importOnnx(text, mRegistry, sourceName);
  case ModelFormat::Auto:
    throw Error("model_format_unsupported", "in-memory input requires an explicit format");
  case ModelFormat::JsonGraph:
    return detail::importJsonGraphText(text, mRegistry, sourceName);
  }
  throw Error("model_format_unsupported", sourceName + ": unsupported model format");
}

} // namespace opfusion::frontend
