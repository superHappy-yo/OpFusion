// Transpose 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxTransposeBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Transpose.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"perm"});
  return std::make_unique<operators::Transpose>(id(s), s.inputs, s.outputs,
                                                operators::TransposeAttrs{integers(s, "perm")});
}

} // namespace
SourceBinding makeOnnxTransposeBinding() {
  return binding(operators::Transpose::semanticIdentity(), "Transpose", create);
}
} // namespace opfusion::frontend::onnxGraph
