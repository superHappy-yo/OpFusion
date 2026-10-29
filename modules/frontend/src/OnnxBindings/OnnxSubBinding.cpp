// Sub 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxSubBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Sub.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {});
  return std::make_unique<operators::Sub>(id(s), s.inputs, s.outputs);
}

} // namespace
SourceBinding makeOnnxSubBinding() {
  return binding(operators::Sub::semanticIdentity(), "Sub", create);
}
} // namespace opfusion::frontend::onnxGraph
