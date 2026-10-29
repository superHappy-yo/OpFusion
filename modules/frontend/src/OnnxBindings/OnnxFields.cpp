#include "OnnxFields.h"

#include <algorithm>
namespace opfusion::frontend::onnxGraph {
void known(const SourceNode &s, std::initializer_list<const char *> fields) {
  require(s.format == "onnx/17", "source_invalid", "ONNX factory requires opset 17");
  for (const auto &item : s.fields.object())
    require(std::any_of(fields.begin(), fields.end(),
                        [&](const char *name) { return item.first == name; }),
            "unsupported_attribute", "unsupported ONNX attribute: " + item.first);
}
std::int64_t integer(const SourceNode &s, const char *key, std::int64_t fallback) {
  const auto *p = s.fields.find(key);
  return p ? p->integer() : fallback;
}
double number(const SourceNode &s, const char *key, double fallback) {
  const auto *p = s.fields.find(key);
  return p ? p->number() : fallback;
}
std::string text(const SourceNode &s, const char *key, const char *fallback) {
  const auto *p = s.fields.find(key);
  return p ? p->string() : fallback;
}
Shape integers(const SourceNode &s, const char *key, Shape fallback) {
  const auto *p = s.fields.find(key);
  if (!p)
    return fallback;
  Shape result;
  for (const auto &v : p->array())
    result.push_back(v.integer());
  return result;
}
const Shape &constant(const SourceNode &s, std::size_t port) {
  require(port < s.integerInputs.size() && s.integerInputs[port].has_value(),
          "shape_value_unresolved", "ONNX shape input is not a static integer tensor");
  return *s.integerInputs[port];
}
OperatorId id(const SourceNode &s) {
  return static_cast<OperatorId>(s.index) + 1;
}
SourceBinding binding(SemanticIdentity semantic, const char *name, OperatorFactory factory,
                      SourceNormalizer normalizer) {
  return {std::move(semantic),
          {{"onnx/17", name, std::nullopt}},
          std::move(factory),
          std::move(normalizer)};
}
} // namespace opfusion::frontend::onnxGraph
