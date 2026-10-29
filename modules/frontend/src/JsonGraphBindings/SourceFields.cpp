#include "SourceFields.h"

#include <algorithm>
#include <limits>
#include <variant>

namespace opfusion::frontend::jsonGraph {

// 共享的是来源字段读取规则而非算子数学语义；不依赖 contracts 的私有辅助头。
bool empty(const SourceValue &value) {
  if (value.isNull()) {
    return true;
  }
  if (const auto *array = std::get_if<SourceValue::Array>(&value.value)) {
    return array->empty();
  }
  if (const auto *object = std::get_if<SourceValue::Object>(&value.value)) {
    return object->empty();
  }
  return false;
}

void known(const SourceValue &value, std::initializer_list<const char *> names) {
  for (const auto &item : value.object()) {
    const bool allowed = std::any_of(names.begin(), names.end(),
                                     [&](const char *name) { return item.first == name; });
    require(allowed || empty(item.second), "unsupported_attribute",
            "unrecognized nonempty source field: " + item.first);
  }
}

void inactive(const SourceValue &value, const char *name) {
  if (const auto *item = value.find(name)) {
    require(item->isNull(), "unsupported_attribute",
            std::string("source field must be null: ") + name);
  }
}

const SourceValue &field(const SourceValue &value, const char *name) {
  const auto *result = value.find(name);
  require(result != nullptr, "invalid_source", std::string("missing source field: ") + name);
  return *result;
}

TensorId uid(const SourceValue &value) {
  require(!value.isNull(), "invalid_ports", "required tensor UID is null");
  const auto id = value.integer();
  require(id > 0, "invalid_ports", "tensor UID must be a positive int64 integer");
  return static_cast<TensorId>(id);
}

TensorId uid(const SourceValue &value, const char *name) {
  return uid(field(value, name));
}

Shape integers(const SourceValue &value) {
  Shape result;
  for (const auto &item : value.array()) {
    result.push_back(item.integer());
  }
  return result;
}

std::vector<double> numbers(const SourceValue &value) {
  std::vector<double> result;
  for (const auto &item : value.array()) {
    result.push_back(item.number());
  }
  return result;
}

std::array<std::int64_t, 2> pair(const SourceValue &value, const char *name) {
  const auto values = integers(field(value, name));
  require(values.size() == 2, "invalid_attribute", std::string(name) + " must have two entries");
  return {values[0], values[1]};
}

Shape pads(const SourceValue &value) {
  const auto before = pair(value, "pre_padding");
  const auto after = pair(value, "post_padding");
  return {before[0], before[1], after[0], after[1]};
}

OperatorId nodeId(const SourceNode &source) {
  require(source.index < std::numeric_limits<OperatorId>::max(), "invalid_source",
          "node index+1 overflows OperatorId");
  return static_cast<OperatorId>(source.index) + 1;
}

// 只消费节点自己的来源声明；本函数不接收、也不补传顶层全图数值政策。
NumericalRequirements common(const SourceNode &source, const char *expectedType,
                             bool topLevelPool) {
  require(source.format == "benchmark-json/1" && source.type == expectedType, "unsupported",
          "source factory called for a different format/type");
  if (topLevelPool) {
    known(source.fields,
          {"type", "name", "compute_data_type", "generate_index", "padding_mode", "post_padding",
           "pre_padding", "resample_mode", "stride", "window", "x_tensor_uid", "y_tensor_uid"});
  } else {
    known(source.fields, {"type", "name", "compute_data_type", "inputs", "outputs", "parameters"});
  }
  if (const auto *type = source.fields.find("type")) {
    require(type->string() == source.type, "invalid_source", "source type metadata mismatch");
  }
  if (const auto *name = source.fields.find("name")) {
    (void)name->string();
  }
  NumericalRequirements result;
  if (const auto *dtype = source.fields.find("compute_data_type")) {
    require(dtype->string() == "float", "unsupported_dtype",
            "source compute_data_type must be float");
    result.computeType = DataType::Float32;
  }
  return result;
}

std::vector<TensorId> singleInput(const SourceNode &source) {
  const auto &inputs = field(source.fields, "inputs");
  known(inputs, {"x_tensor_uid"});
  return {uid(inputs, "x_tensor_uid")};
}

std::vector<TensorId> singleOutput(const SourceNode &source) {
  const auto &outputs = field(source.fields, "outputs");
  known(outputs, {"y_tensor_uid"});
  return {uid(outputs, "y_tensor_uid")};
}

SourceBinding sourceBinding(SemanticIdentity semantic, const char *type,
                            std::optional<std::string> operation, OperatorFactory factory) {
  return {std::move(semantic),
          {{"benchmark-json/1", type, std::move(operation)}},
          std::move(factory),
          {}};
}

} // namespace opfusion::frontend::jsonGraph
