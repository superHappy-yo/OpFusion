// Gather 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxGatherBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Gather.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"axis"});
  return std::make_unique<operators::Gather>(id(s), s.inputs, s.outputs,
                                             operators::GatherAttrs{integer(s, "axis", 0)});
}

} // namespace
SourceBinding makeOnnxGatherBinding() {
  return binding(operators::Gather::semanticIdentity(), "Gather", create);
}
} // namespace opfusion::frontend::onnxGraph
