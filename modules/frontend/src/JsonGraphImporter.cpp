#include "JsonGraphImporter.h"

#include "opfusion/common/Error.h"
#include "opfusion/graph/DAG.h"
#include "opfusion/graph/Operator.h"
#include "opfusion/graph/Tensor.h"
#include "opfusion/registry/Registry.h"
#include "opfusion/registry/Source.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace opfusion::frontend::detail {
namespace {

using Json = nlohmann::json;

std::string memberPath(const std::string &path, const std::string &key) {
  std::string escaped;
  for (const char c : key) {
    if (c == '~')
      escaped += "~0";
    else if (c == '/')
      escaped += "~1";
    else
      escaped += c;
  }
  return path + "/" + escaped;
}

/// 通过公开 SAX 事件建立 DOM，拒绝普通 DOM 解析会覆盖的重复键及溢出整数降级。
/// 保存路径和容器栈以附精确来源上下文；下划线覆写名/继承别名由第三方协议固定。
class JsonReader final : public nlohmann::json_sax<Json> {
public:
  bool null() override { return append(nullptr); }
  bool boolean(bool value) override { return append(value); }
  bool number_integer(number_integer_t value) override { return append(value); }
  bool number_unsigned(number_unsigned_t value) override { return append(value); }
  bool number_float(number_float_t value, const string_t &token) override {
    // 库可能本地化 token 的小数点；仅“可选负号加 ASCII 数字”表示整数溢出后的浮点回退。
    // 不能将这种 token 当有限 double 接受，否则整数边界会被舍入后悄悄通过。
    const auto firstDigit = !token.empty() && token.front() == '-' ? 1U : 0U;
    const bool integerToken =
        firstDigit < token.size() &&
        token.find_first_not_of("0123456789", firstDigit) == std::string::npos;
    require(!integerToken, "benchmark_integer_range", nextPath() + ": integer is out of range");
    require(std::isfinite(value), "benchmark_number_range", nextPath() + ": non-finite number");
    return append(value);
  }
  bool string(string_t &value) override { return append(value); }
  bool binary(binary_t &) override {
    throw Error("benchmark_json_type", nextPath() + ": binary JSON is not supported");
  }
  bool start_object(std::size_t) override { return start(Json::object()); }
  bool start_array(std::size_t) override { return start(Json::array()); }
  bool key(string_t &key) override {
    auto &frame = mStack.back();
    require(!frame.value.contains(key), "benchmark_duplicate_key",
            memberPath(frame.path, key) + ": duplicate object key");
    frame.key = key;
    return true;
  }
  bool end_object() override { return finish(); }
  bool end_array() override { return finish(); }
  bool parse_error(std::size_t position, const std::string &,
                   const nlohmann::detail::exception &error) override {
    throw Error(error.id == 406 ? "benchmark_number_range" : "benchmark_json_syntax",
                "JSON byte " + std::to_string(position) + ": " + error.what());
  }
  Json take() { return std::move(mRoot); }

private:
  struct Frame {
    Json value;
    std::optional<std::string> key;
    std::string path;
  };

  std::string nextPath() const {
    if (mStack.empty())
      return "$";
    const auto &frame = mStack.back();
    return memberPath(frame.path, frame.value.is_array() ? std::to_string(frame.value.size())
                                                         : frame.key.value_or(""));
  }
  bool start(Json value) {
    require(mStack.size() < 256, "benchmark_json_depth",
            nextPath() + ": nesting exceeds the profile limit of 256");
    auto path = nextPath();
    mStack.push_back({std::move(value), std::nullopt, std::move(path)});
    return true;
  }
  bool finish() {
    auto value = std::move(mStack.back().value);
    mStack.pop_back();
    return append(std::move(value));
  }
  bool append(Json value) {
    if (mStack.empty()) {
      mRoot = std::move(value);
    } else {
      auto &frame = mStack.back();
      if (frame.value.is_array()) {
        frame.value.push_back(std::move(value));
      } else {
        frame.value.emplace(*frame.key, std::move(value));
        frame.key.reset();
      }
    }
    return true;
  }

  Json mRoot;
  std::vector<Frame> mStack;
};

void object(const Json &value, const std::string &path) {
  require(value.is_object(), "benchmark_schema_type", path + ": expected object");
}

void array(const Json &value, const std::string &path) {
  require(value.is_array(), "benchmark_schema_type", path + ": expected array");
}

const Json &field(const Json &value, const std::string &key, const std::string &path) {
  object(value, path);
  const auto found = value.find(key);
  require(found != value.end(), "benchmark_schema_missing",
          memberPath(path, key) + ": required field is missing");
  return *found;
}

const std::string &string(const Json &value, const std::string &path) {
  require(value.is_string(), "benchmark_schema_type", path + ": expected string");
  return value.get_ref<const std::string &>();
}

bool boolean(const Json &value, const std::string &path) {
  require(value.is_boolean(), "benchmark_schema_type", path + ": expected boolean");
  return value.get<bool>();
}

std::uint64_t unsignedInteger(const Json &value, const std::string &path) {
  require(value.is_number_integer(), "benchmark_schema_type",
          path + ": expected nonnegative integer");
  if (value.is_number_unsigned())
    return value.get<std::uint64_t>();
  const auto result = value.get<std::int64_t>();
  require(result >= 0, "benchmark_integer_range", path + ": negative integer");
  return static_cast<std::uint64_t>(result);
}

std::int64_t signedInteger(const Json &value, const std::string &path) {
  require(value.is_number_integer(), "benchmark_schema_type",
          path + ": expected signed 64-bit integer");
  if (value.is_number_unsigned()) {
    const auto result = value.get<std::uint64_t>();
    require(result <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            "benchmark_integer_range", path + ": integer exceeds INT64_MAX");
    return static_cast<std::int64_t>(result);
  }
  return value.get<std::int64_t>();
}

std::vector<std::int64_t> integers(const Json &value, const std::string &path) {
  array(value, path);
  std::vector<std::int64_t> result;
  result.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i)
    result.push_back(signedInteger(value[i], memberPath(path, std::to_string(i))));
  return result;
}

void onlyFields(const Json &value, const std::set<std::string> &allowed, const std::string &path) {
  object(value, path);
  for (const auto &item : value.items())
    require(allowed.count(item.key()) != 0, "benchmark_schema_field",
            memberPath(path, item.key()) + ": field is outside benchmark-json/1");
}

void validateProfile(const Json &root) {
  onlyFields(root,
             {"name", "nodes", "tensors", "compute_data_type", "intermediate_data_type",
              "io_data_type", "is_override_shape_enabled", "min_required_engine_api_version"},
             "$");
  for (const auto &key : {"compute_data_type", "intermediate_data_type", "io_data_type"})
    require(string(field(root, key, "$"), memberPath("$", key)) == "float", "benchmark_profile",
            memberPath("$", key) + ": benchmark-json/1 requires float");
  require(!boolean(field(root, "is_override_shape_enabled", "$"), "$/is_override_shape_enabled"),
          "benchmark_profile", "$/is_override_shape_enabled: shape override is not supported");
  const auto &version = field(root, "min_required_engine_api_version", "$");
  const std::string path = "$/min_required_engine_api_version";
  onlyFields(version, {"major", "minor", "patch"}, path);
  require(unsignedInteger(field(version, "major", path), memberPath(path, "major")) == 1 &&
              unsignedInteger(field(version, "minor", path), memberPath(path, "minor")) == 0 &&
              unsignedInteger(field(version, "patch", path), memberPath(path, "patch")) == 0,
          "benchmark_profile", path + ": benchmark-json/1 requires version 1.0.0");
}

// 复制完整递归来源，不借用 JSON DOM；整数/布尔/浮点保持各自类型，null 仍交给工厂解释。
SourceValue sourceValue(const Json &value, const std::string &path) {
  if (value.is_null())
    return SourceValue{};
  if (value.is_boolean())
    return SourceValue(value.get<bool>());
  if (value.is_number_integer())
    return SourceValue(signedInteger(value, path));
  if (value.is_number_float()) {
    const double result = value.get<double>();
    require(std::isfinite(result), "benchmark_number_range", path + ": non-finite number");
    return SourceValue(result);
  }
  if (value.is_string())
    return SourceValue(value.get<std::string>());
  if (value.is_array()) {
    SourceValue::Array result;
    result.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i)
      result.push_back(sourceValue(value[i], memberPath(path, std::to_string(i))));
    return SourceValue(std::move(result));
  }
  if (value.is_object()) {
    SourceValue::Object result;
    for (const auto &item : value.items())
      result.emplace(item.key(), sourceValue(item.value(), memberPath(path, item.key())));
    return SourceValue(std::move(result));
  }
  throw Error("benchmark_json_type", path + ": unsupported source value type");
}

// 每次导入创建随机图身份，不用文件名/内容作缓存键；随机源失败时不伪造稳定回退值。
std::string graphToken() {
  try {
    std::random_device entropy;
    std::uniform_int_distribution<std::uint32_t> distribution;
    std::ostringstream token;
    token.imbue(std::locale::classic());
    token << std::hex << std::setfill('0');
    for (unsigned i = 0; i < 4; ++i)
      token << std::setw(8) << distribution(entropy);
    return token.str();
  } catch (const std::exception &error) {
    throw Error("benchmark_graph_id",
                std::string("random graph token generation failed: ") + error.what());
  }
}

std::string nodeContext(const NodeOrigin &origin) {
  return "source node[" + std::to_string(origin.index) + "] (" + origin.name + ")";
}

std::vector<Tensor> readTensors(const Json &values, ImportResult &result) {
  array(values, "$/tensors");
  std::vector<Tensor> tensors;
  tensors.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto path = memberPath("$/tensors", std::to_string(i));
    const auto &value = values[i];
    onlyFields(value,
               {"uid", "name", "dims", "strides", "data_type", "alignment", "virtual",
                "is_runtime_pass_by_value"},
               path);
    const auto id = unsignedInteger(field(value, "uid", path), memberPath(path, "uid"));
    require(result.tensorOrigins.count(id) == 0, "benchmark_duplicate_uid",
            memberPath(path, "uid") + ": duplicate tensor UID " + std::to_string(id));
    const auto name = string(field(value, "name", path), memberPath(path, "name"));
    const auto shape = integers(field(value, "dims", path), memberPath(path, "dims"));
    const auto &dtype = string(field(value, "data_type", path), memberPath(path, "data_type"));
    require(dtype == "float", "benchmark_profile",
            memberPath(path, "data_type") + ": benchmark-json/1 requires float");
    TensorOrigin origin;
    if (const auto found = value.find("strides"); found != value.end())
      origin.strides = integers(*found, memberPath(path, "strides"));
    if (const auto found = value.find("alignment"); found != value.end())
      origin.alignment = unsignedInteger(*found, memberPath(path, "alignment"));
    if (const auto found = value.find("virtual"); found != value.end())
      origin.virtualTensor = boolean(*found, memberPath(path, "virtual"));
    if (const auto found = value.find("is_runtime_pass_by_value"); found != value.end())
      origin.runtimePassByValue = boolean(*found, memberPath(path, "is_runtime_pass_by_value"));
    try {
      // shape 原序保留；存储提示只进入来源映射，角色/轴不据名字或 stride 猜成权重/布局。
      tensors.emplace_back(id, name, shape, DataType::Float32);
    } catch (const Error &error) {
      throw Error(error.code(), path + ": " + error.what());
    }
    result.tensorOrigins.emplace(id, std::move(origin));
  }
  return tensors;
}

std::vector<std::unique_ptr<Operator>> readNodes(const Json &values, const Registry &registry,
                                                 ImportResult &result) {
  array(values, "$/nodes");
  std::vector<std::unique_ptr<Operator>> nodes;
  nodes.reserve(values.size());
  result.nodeOrigins.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto path = memberPath("$/nodes", std::to_string(i));
    const auto &value = values[i];
    object(value, path);
    const auto name = string(field(value, "name", path), memberPath(path, "name"));
    require(i < std::numeric_limits<OperatorId>::max(), "benchmark_integer_range",
            path + ": source index cannot be represented as an operator ID");
    NodeOrigin origin{static_cast<OperatorId>(i) + 1, i, name, {}, {}, {}};
    try {
      SourceNode source;
      source.name = name;
      source.index = i;
      source.type = string(field(value, "type", path), memberPath(path, "type"));
      source.format = result.profile;
      if (const auto inputs = value.find("inputs"); inputs != value.end() && inputs->is_object())
        if (const auto operation = inputs->find("operation"); operation != inputs->end())
          source.operation =
              string(*operation, memberPath(memberPath(path, "inputs"), "operation"));
      if (!source.operation.has_value())
        if (const auto mode = value.find("resample_mode"); mode != value.end())
          source.operation = string(*mode, memberPath(path, "resample_mode"));
      // 包括 null、未知属性及扁平端口在内的完整节点交给绑定；此处不按算子名称解释字段。
      source.fields = sourceValue(value, path);
      auto node = registry.create(source);
      require(node != nullptr, "benchmark_factory_null", "factory returned a null operator");
      require(node->id() == origin.id, "benchmark_factory_id",
              "factory must use source.index + 1 as operator ID");
      nodes.push_back(std::move(node));
    } catch (const Error &error) {
      throw Error(error.code(), nodeContext(origin) + ": " + error.what());
    }
    result.nodeOrigins.push_back(std::move(origin));
  }
  return nodes;
}

std::shared_ptr<const DAG> makeGraph(const std::string &name, std::vector<Tensor> tensors,
                                     std::vector<std::unique_ptr<Operator>> nodes,
                                     const Registry &registry,
                                     const std::vector<NodeOrigin> &origins) {
  try {
    // DAG 负责 input-only 推导、声明输出比较、引用/唯一生产者、端口边及环检测。
    // 来源没有显式完整模型接口，不将依赖推断结果伪装为 modelInputs/modelOutputs。
    return std::make_shared<const DAG>(graphToken(), name, std::move(tensors), std::move(nodes),
                                       std::nullopt, registry.id());
  } catch (const Error &error) {
    const std::string message = error.what();
    // 仅匹配外层节点错误前缀；正文偶然提到其他 node ID 时不能误贴来源标签。
    for (const auto &origin : origins)
      if (message.rfind("node " + std::to_string(origin.id) + " (", 0) == 0)
        throw Error(error.code(), nodeContext(origin) + ": " + message);
    throw;
  }
}

// 区分“来源未声明”“执行条件未建立”和“未搜索”；不能把这些缺失升级为全局不可行。
void addDiagnostics(ImportResult &result, const std::string &sourceName) {
  result.diagnostics = {
      {"source", sourceName},
      {"external_inputs_inferred",
       "externalInputs() is inferred from graph dependencies, not declared model inputs; "
       "external operands are not assumed to be constants or weights."},
      {"terminal_outputs_inferred",
       "terminalOutputs() is inferred from graph dependencies, not declared model outputs; "
       "modelOutputs() remains nullopt."},
      {"weight_contents_unavailable", "No initializer or weight contents are established."},
      {"physical_hints_only",
       "Source strides/alignment/virtual/pass-by-value are provenance hints only; "
       "their units and representation/buffer semantics are not established."},
      {"execution_conditions_unestablished",
       "No platform or deployment Boundary is established, including input residency, "
       "output destinations, available capacity and timing boundary."},
      {"candidates_not_searched",
       "No execution candidates were searched or evaluated; this is not a global "
       "infeasibility conclusion."},
  };
}

} // namespace

ImportResult importJsonGraphText(const std::string &text, const Registry &registry,
                                 const std::string &sourceName) {
  try {
    // 某些解析路径把 raw NUL 当输入终止，必须先拒绝，避免尾随内容被截断；转义 NUL 仍合法。
    require(text.find('\0') == std::string::npos, "benchmark_json_syntax",
            "raw NUL bytes are not valid JSON; use an escaped Unicode string value");
    JsonReader reader;
    require(Json::sax_parse(text, &reader), "benchmark_json_syntax", "invalid JSON document");
    const auto root = reader.take();
    validateProfile(root);
    const auto name = string(field(root, "name", "$"), "$/name");
    ImportResult result;
    result.profile = "benchmark-json/1";
    auto tensors = readTensors(field(root, "tensors", "$"), result);
    auto nodes = readNodes(field(root, "nodes", "$"), registry, result);
    result.graph =
        makeGraph(name, std::move(tensors), std::move(nodes), registry, result.nodeOrigins);
    addDiagnostics(result, sourceName);
    return result;
  } catch (const Error &error) {
    throw Error(error.code(), sourceName + ": " + error.what());
  } catch (const Json::exception &error) {
    throw Error("benchmark_json_syntax", sourceName + ": " + error.what());
  }
}

} // namespace opfusion::frontend::detail
