// Concat 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxConcatBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Concat.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"axis"});
  require(s.fields.find("axis") != nullptr, "field_missing", "Concat axis missing");
  return std::make_unique<operators::Concat>(id(s), s.inputs, s.outputs,
                                             operators::ConcatAttrs{integer(s, "axis", 0)});
}

} // namespace
SourceBinding makeOnnxConcatBinding() {
  return binding(operators::Concat::semanticIdentity(), "Concat", create);
}
} // namespace opfusion::frontend::onnxGraph
