// Reshape 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxReshapeBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Reshape.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"allowzero"});
  require(s.inputs.size() == 2, "invalid_ports", "ONNX Reshape requires shape input");
  const auto zero = integer(s, "allowzero", 0);
  require(zero == 0 || zero == 1, "invalid_attribute", "invalid allowzero");
  return std::make_unique<operators::Reshape>(id(s), s.inputs, s.outputs,
                                              operators::ReshapeAttrs{constant(s, 1), zero == 1});
}

} // namespace
SourceBinding makeOnnxReshapeBinding() {
  return binding(operators::Reshape::semanticIdentity(), "Reshape", create);
}
} // namespace opfusion::frontend::onnxGraph
