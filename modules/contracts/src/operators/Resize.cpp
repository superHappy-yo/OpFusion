#include "opfusion/operators/Resize.h"

#include "SemanticChecks.h"

#include <utility>

namespace opfusion::operators {
using namespace detail;

Resize::Resize(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
               ResizeAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}

SemanticIdentity Resize::semanticIdentity() {
  return identity("Resize");
}

void Resize::validateAttributes() const {
  ports(*this, 1);
  require(mAttrs.dataAxes == AxisOrder::NCHW, "unsupported_layout", "Resize requires NCHW data");
  // 按完整政策组合拒绝未支持选项，不能只采用 scales 而忽略其余语义字段。
  const bool supportedPolicy = mAttrs.mode == ResizeMode::Nearest &&
                               mAttrs.coordinateTransform == CoordinateTransform::Asymmetric &&
                               mAttrs.nearestMode == NearestMode::Floor &&
                               mAttrs.aspectPolicy == AspectPolicy::Stretch;
  const bool supportedShape = mAttrs.scales == std::vector<double>({1.0, 1.0, 2.0, 2.0}) &&
                              mAttrs.sizes.empty() && mAttrs.axes.empty() && mAttrs.roi.empty();
  const bool supportedFilter = !mAttrs.antialias && !mAttrs.excludeOutside &&
                               mAttrs.cubicCoefficient == -0.75 && mAttrs.extrapolationValue == 0.0;
  require(supportedPolicy && supportedShape && supportedFilter, "unsupported",
          "Resize supports nearest/floor/asymmetric NCHW 2x spatial scaling only");
}

std::vector<TensorSignature> Resize::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, 1);
  rank(inputs[0], 4);
  require(compatibleAxes(inputs[0].axes, nchw()), "axis_mismatch",
          "Resize requires logical NCHW axes");
  Shape output = inputs[0].shape;
  output[2] = multiply(output[2], 2);
  output[3] = multiply(output[3], 2);
  return {{std::move(output), inputs[0].dtype, nchw()}};
}

Traits Resize::traits() const {
  return {true, false, false, false};
}

IndexSemantics Resize::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  // 固定 asymmetric/floor 2x 政策的源坐标正是 floor(i/2)，不泛化到其他插值模式。
  piece.inputs = {
      {0, {index(0), index(1), index(2, 1, 2), index(3, 1, 2)}, FillValue::None, nchw(), true}};
  piece.outputs = {{0, identityCoordinates(4)}};
  return known(std::move(piece));
}

std::vector<Attribute> Resize::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"spatial_dimensions", std::int64_t{2}},
          {"mode", std::string("nearest")},
          {"coordinate_transform", std::string("asymmetric")},
          {"nearest_mode", std::string("floor")},
          {"scales", mAttrs.scales},
          {"sizes", mAttrs.sizes},
          {"axes", mAttrs.axes},
          {"roi", mAttrs.roi},
          {"aspect_policy", std::string("stretch")},
          {"antialias", mAttrs.antialias},
          {"exclude_outside", mAttrs.excludeOutside},
          {"cubic_coefficient", mAttrs.cubicCoefficient},
          {"extrapolation_value", mAttrs.extrapolationValue}};
}

} // namespace opfusion::operators
