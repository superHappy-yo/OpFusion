// Mul 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxMulBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Mul.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {});
  return std::make_unique<operators::Mul>(id(s), s.inputs, s.outputs);
}

} // namespace
SourceBinding makeOnnxMulBinding() {
  return binding(operators::Mul::semanticIdentity(), "Mul", create);
}
} // namespace opfusion::frontend::onnxGraph
