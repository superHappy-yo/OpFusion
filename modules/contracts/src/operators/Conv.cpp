// 一至三维、可分组的无偏置互相关；所有空间坐标均是逻辑元素坐标。
#include "opfusion/operators/Conv.h"

#include "SemanticChecks.h"

#include <utility>

namespace opfusion::operators {
using namespace detail;
Conv::Conv(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
           ConvAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Conv::semanticIdentity() {
  return identity("Conv");
}
void Conv::validateAttributes() const {
  ports(*this, 2);
  const auto n = mAttrs.spatialDimensions;
  require(mAttrs.dataAxes == spatialDataOrder(n) && mAttrs.weightAxes == spatialWeightOrder(n),
          "unsupported_layout", "Conv logical axes disagree with spatial dimensions");
  require(mAttrs.strides.size() == n && mAttrs.dilations.size() == n && mAttrs.pads.size() == 2 * n,
          "invalid_attribute", "Conv spatial attribute length mismatch");
  require(mAttrs.kernelShape.empty() || mAttrs.kernelShape.size() == n, "invalid_attribute",
          "Conv kernel dimension mismatch");
  if (!mAttrs.kernelShape.empty())
    positive(mAttrs.kernelShape);
  positive(mAttrs.strides);
  positive(mAttrs.dilations);
  padding(mAttrs.pads);
  require(mAttrs.groups > 0, "invalid_attribute", "groups must be positive");
}
std::vector<TensorSignature> Conv::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, 2, true);
  const auto n = mAttrs.spatialDimensions;
  rank(inputs[0], n + 2);
  rank(inputs[1], n + 2);
  const auto dataAxes = logicalAxes(mAttrs.dataAxes), weightAxes = logicalAxes(mAttrs.weightAxes);
  require(compatibleAxes(inputs[0].axes, dataAxes) && compatibleAxes(inputs[1].axes, weightAxes),
          "axis_mismatch", "Conv logical axes mismatch");
  const auto &x = inputs[0].shape;
  const auto &w = inputs[1].shape;
  require(x[1] > 0 && w[0] > 0 && w[1] > 0 && x[1] == multiply(w[1], mAttrs.groups) &&
              w[0] % mAttrs.groups == 0,
          "invalid_shape", "grouped convolution channel mismatch");
  require(mAttrs.kernelShape.empty() || mAttrs.kernelShape == Shape(w.begin() + 2, w.end()),
          "invalid_shape", "Conv kernel shape mismatch");
  Shape output{x[0], w[0]};
  for (std::size_t d = 0; d < n; ++d) {
    output.push_back(spatial(x[d + 2], w[d + 2], mAttrs.dilations[d], mAttrs.pads[d],
                             mAttrs.pads[d + n], mAttrs.strides[d]));
  }
  return {{std::move(output), inputs[0].dtype, dataAxes}};
}
Traits Conv::traits() const {
  return {true, false, true, false};
}
std::vector<Attribute> Conv::attributes(const std::vector<TensorSignature> &inputs) const {
  infer(inputs);
  return {{"spatial_dimensions", static_cast<std::int64_t>(mAttrs.spatialDimensions)},
          {"kernel_shape", Shape(inputs.at(1).shape.begin() + 2, inputs.at(1).shape.end())},
          {"strides", mAttrs.strides},
          {"dilations", mAttrs.dilations},
          {"pads", mAttrs.pads},
          {"groups", mAttrs.groups},
          {"mode", std::string("cross_correlation")}};
}
IndexSemantics Conv::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto n = mAttrs.spatialDimensions, r = n + 2;
  const auto &w = inputs[1].shape;
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.domain.push_back({"ic", 0, w[1], true});
  for (std::size_t d = 0; d < n; ++d) {
    piece.domain.push_back({"k" + std::to_string(d), 0, w[d + 2], true});
  }
  piece.reduction = ReductionKind::Sum;
  std::vector<IndexExpression> x{index(0), {{{1, w[1], w[0] / mAttrs.groups}, {r, 1, 1}}, 0}};
  std::vector<IndexExpression> weights{index(1), index(r)};
  for (std::size_t d = 0; d < n; ++d) {
    x.push_back(
        {{{d + 2, mAttrs.strides[d], 1}, {r + 1 + d, mAttrs.dilations[d], 1}}, -mAttrs.pads[d]});
    weights.push_back(index(r + 1 + d));
  }
  piece.inputs = {{0, std::move(x), FillValue::Zero, logicalAxes(mAttrs.dataAxes), false},
                  {1, std::move(weights), FillValue::None, logicalAxes(mAttrs.weightAxes), false}};
  piece.outputs = {{0, identityCoordinates(r)}};
  return known(std::move(piece));
}
} // namespace opfusion::operators
