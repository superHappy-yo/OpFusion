// Slice 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxSliceBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Slice.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {});
  require(s.inputs.size() >= 3 && s.inputs.size() <= 5, "invalid_ports", "Slice ports invalid");
  const auto &shape = s.inputSignatures[0].shape;
  operators::SliceAttrs a;
  a.starts = Shape(shape.size(), 0);
  a.limits = shape;
  a.strides = Shape(shape.size(), 1);
  const auto starts = constant(s, 1), ends = constant(s, 2);
  Shape axes;
  if (s.inputs.size() > 3 && s.inputs[3])
    axes = constant(s, 3);
  else
    for (std::size_t i = 0; i < starts.size(); ++i)
      axes.push_back(static_cast<std::int64_t>(i));
  Shape steps(starts.size(), 1);
  if (s.inputs.size() > 4 && s.inputs[4])
    steps = constant(s, 4);
  require(starts.size() == ends.size() && starts.size() == axes.size() &&
              steps.size() == axes.size(),
          "invalid_shape", "Slice argument lengths mismatch");
  std::vector<bool> seen(shape.size());
  for (std::size_t i = 0; i < axes.size(); ++i) {
    auto d = axes[i];
    if (d < 0)
      d += static_cast<std::int64_t>(shape.size());
    require(d >= 0 && static_cast<std::size_t>(d) < shape.size() && !seen[d], "invalid_axis",
            "Slice axis invalid or duplicate");
    seen[d] = true;
    require(steps[i] > 0, "unsupported_attribute", "negative Slice step not implemented");
    auto start = starts[i], end = ends[i];
    if (start < 0)
      start += shape[d];
    if (end < 0)
      end += shape[d];
    start = std::clamp(start, std::int64_t{0}, shape[d]);
    end = std::clamp(end, std::int64_t{0}, shape[d]);
    a.starts[d] = start;
    a.limits[d] = std::max(start, end);
    a.strides[d] = steps[i];
  }
  return std::make_unique<operators::Slice>(id(s), std::vector<TensorId>{s.inputs[0]}, s.outputs,
                                            a);
}

} // namespace
SourceBinding makeOnnxSliceBinding() {
  return binding(operators::Slice::semanticIdentity(), "Slice", create);
}
} // namespace opfusion::frontend::onnxGraph
