// Flatten 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxFlattenBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Flatten.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"axis"});
  return std::make_unique<operators::Flatten>(id(s), s.inputs, s.outputs,
                                              operators::FlattenAttrs{integer(s, "axis", 1)});
}

} // namespace
SourceBinding makeOnnxFlattenBinding() {
  return binding(operators::Flatten::semanticIdentity(), "Flatten", create);
}
} // namespace opfusion::frontend::onnxGraph
