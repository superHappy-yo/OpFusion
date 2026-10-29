// Split 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxSplitBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Split.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"axis"});
  operators::SplitAttrs a;
  a.axis = integer(s, "axis", 0);
  if (s.inputs.size() == 2)
    a.sizes = constant(s, 1);
  return std::make_unique<operators::Split>(id(s), s.inputs, s.outputs, a);
}

} // namespace
SourceBinding makeOnnxSplitBinding() {
  return binding(operators::Split::semanticIdentity(), "Split", create);
}
} // namespace opfusion::frontend::onnxGraph
