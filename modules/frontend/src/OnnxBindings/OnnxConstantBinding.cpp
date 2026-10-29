// Constant 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxConstantBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Constant.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"value"});
  const auto &v = s.fields.at("value");
  SourceNode data;
  data.fields = v;
  operators::ConstantAttrs a;
  a.contentDigest = v.at("content_sha256").string();
  a.signature.shape = integers(data, "shape");
  const auto type = v.at("dtype").integer();
  require(type >= 0 && type <= static_cast<std::int64_t>(DataType::Bool), "unsupported_dtype",
          "Constant dtype invalid");
  a.signature.dtype = static_cast<DataType>(type);
  if (v.find("integers"))
    a.integers = integers(data, "integers");
  if (const auto *f = v.find("floating")) {
    a.floating = std::vector<double>{};
    for (const auto &x : f->array())
      a.floating->push_back(x.number());
  }
  return std::make_unique<operators::Constant>(id(s), s.inputs, s.outputs, a);
}

} // namespace
SourceBinding makeOnnxConstantBinding() {
  return binding(operators::Constant::semanticIdentity(), "Constant", create);
}
} // namespace opfusion::frontend::onnxGraph
