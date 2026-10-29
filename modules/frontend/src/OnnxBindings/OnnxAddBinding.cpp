// Add 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxAddBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Add.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {});
  return std::make_unique<operators::Add>(id(s), s.inputs, s.outputs);
}

} // namespace
SourceBinding makeOnnxAddBinding() {
  return binding(operators::Add::semanticIdentity(), "Add", create);
}
} // namespace opfusion::frontend::onnxGraph
