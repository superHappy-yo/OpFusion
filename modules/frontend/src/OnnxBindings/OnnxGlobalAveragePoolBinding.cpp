// GlobalAveragePool 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxGlobalAveragePoolBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/GlobalAveragePool.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {});
  return std::make_unique<operators::GlobalAveragePool>(id(s), s.inputs, s.outputs);
}

} // namespace
SourceBinding makeOnnxGlobalAveragePoolBinding() {
  return binding(operators::GlobalAveragePool::semanticIdentity(), "GlobalAveragePool", create);
}
} // namespace opfusion::frontend::onnxGraph
