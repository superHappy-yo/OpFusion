// Softmax 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxSoftmaxBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"axis"});
  return std::make_unique<operators::Softmax>(id(s), s.inputs, s.outputs,
                                              operators::SoftmaxAttrs{integer(s, "axis", -1)});
}

} // namespace
SourceBinding makeOnnxSoftmaxBinding() {
  return binding(operators::Softmax::semanticIdentity(), "Softmax", create);
}
} // namespace opfusion::frontend::onnxGraph
