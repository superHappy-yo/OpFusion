// Shape 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxShapeOfBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/ShapeOf.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"start", "end"});
  operators::ShapeOfAttrs a;
  a.start = integer(s, "start", 0);
  if (s.fields.find("end"))
    a.end = integer(s, "end", 0);
  return std::make_unique<operators::ShapeOf>(id(s), s.inputs, s.outputs, a);
}

} // namespace
SourceBinding makeOnnxShapeOfBinding() {
  return binding(operators::ShapeOf::semanticIdentity(), "Shape", create);
}
} // namespace opfusion::frontend::onnxGraph
