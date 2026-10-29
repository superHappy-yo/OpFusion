// Resize 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxResizeBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Resize.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"mode", "coordinate_transformation_mode", "nearest_mode", "cubic_coeff_a",
            "exclude_outside", "extrapolation_value"});
  require(s.inputs.size() >= 3 && s.inputs.size() <= 4 && s.inputs[0] != 0, "invalid_ports",
          "Resize ports invalid");
  require(text(s, "mode", "nearest") == "nearest" &&
              text(s, "coordinate_transformation_mode", "half_pixel") == "asymmetric" &&
              text(s, "nearest_mode", "round_prefer_floor") == "floor",
          "unsupported_attribute", "Resize supports nearest/asymmetric/floor only");
  operators::ResizeAttrs a;
  a.cubicCoefficient = number(s, "cubic_coeff_a", -0.75);
  a.extrapolationValue = number(s, "extrapolation_value", 0);
  a.excludeOutside = integer(s, "exclude_outside", 0) != 0;
  require(s.floatingInputs.size() > 2 && s.floatingInputs[2].has_value(), "shape_value_unresolved",
          "Resize scales must be static");
  a.scales = *s.floatingInputs[2];
  require(s.inputs.size() < 4 || s.inputs[3] == 0 || elements(s.inputSignatures[3].shape) == 0,
          "unsupported_attribute", "Resize sizes not supported");
  require(s.inputs[1] == 0 || elements(s.inputSignatures[1].shape) == 0, "unsupported_attribute",
          "Resize ROI not supported");
  return std::make_unique<operators::Resize>(id(s), std::vector<TensorId>{s.inputs[0]}, s.outputs,
                                             a);
}

} // namespace
SourceBinding makeOnnxResizeBinding() {
  return binding(operators::Resize::semanticIdentity(), "Resize", create);
}
} // namespace opfusion::frontend::onnxGraph
