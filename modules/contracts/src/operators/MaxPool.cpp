// 一至三维最大池化；负无穷填充，输出为完整值，不返回最大值索引。
#include "opfusion/operators/MaxPool.h"

#include "SemanticChecks.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace opfusion::operators {
using namespace detail;
MaxPool::MaxPool(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                 MaxPoolAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity MaxPool::semanticIdentity() {
  return identity("MaxPool");
}
void MaxPool::validateAttributes() const {
  ports(*this, 1);
  const auto n = mAttrs.spatialDimensions;
  require(mAttrs.dataAxes == spatialDataOrder(n), "unsupported_layout",
          "MaxPool logical axes mismatch");
  require(mAttrs.window.size() == n && mAttrs.strides.size() == n && mAttrs.dilations.size() == n &&
              mAttrs.pads.size() == 2 * n,
          "invalid_attribute", "MaxPool spatial attribute length mismatch");
  positive(mAttrs.window);
  positive(mAttrs.strides);
  positive(mAttrs.dilations);
  padding(mAttrs.pads);
  require(mAttrs.padding == PoolPadding::NegativeInfinity && !mAttrs.returnIndices, "unsupported",
          "MaxPool requires negative-infinity padding and no indices");
}
std::vector<TensorSignature> MaxPool::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, 1, true);
  const auto n = mAttrs.spatialDimensions;
  rank(inputs[0], n + 2);
  const auto axes = logicalAxes(mAttrs.dataAxes);
  require(compatibleAxes(inputs[0].axes, axes), "axis_mismatch", "MaxPool logical axes mismatch");
  const auto &x = inputs[0].shape;
  require(x[1] > 0, "invalid_shape", "MaxPool channels must be positive");
  Shape output{x[0], x[1]};
  for (std::size_t d = 0; d < n; ++d) {
    const auto effective = add(multiply(mAttrs.window[d] - 1, mAttrs.dilations[d]), 1);
    const auto padded = add(add(x[d + 2], mAttrs.pads[d]), mAttrs.pads[d + n]);
    require(x[d + 2] > 0, "invalid_shape", "empty pooling input");
    const auto difference = padded - effective;
    auto quotient = difference / mAttrs.strides[d];
    if (difference % mAttrs.strides[d]) {
      if (mAttrs.ceilMode && difference > 0)
        ++quotient;
      if (!mAttrs.ceilMode && difference < 0)
        --quotient;
    }
    require(quotient < std::numeric_limits<std::int64_t>::max(), "shape_overflow",
            "pool output overflow");
    auto size = quotient + 1;
    // ceil 模式排除所有起点位于右侧 padding 的窗口。
    if (mAttrs.ceilMode)
      size = std::min(size, (add(x[d + 2], mAttrs.pads[d]) - 1) / mAttrs.strides[d] + 1);
    require(size > 0, "invalid_shape", "empty pooling output");
    output.push_back(size);
  }
  return {{std::move(output), inputs[0].dtype, axes}};
}
Traits MaxPool::traits() const {
  return {true, false, true, false};
}
std::vector<Attribute> MaxPool::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"spatial_dimensions", static_cast<std::int64_t>(mAttrs.spatialDimensions)},
          {"kernel_shape", mAttrs.window},
          {"strides", mAttrs.strides},
          {"dilations", mAttrs.dilations},
          {"pads", mAttrs.pads},
          {"ceil_mode", mAttrs.ceilMode},
          {"padding", std::string("negative_infinity")},
          {"return_indices", mAttrs.returnIndices}};
}
IndexSemantics MaxPool::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto n = mAttrs.spatialDimensions, r = n + 2;
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.reduction = ReductionKind::Maximum;
  std::vector<IndexExpression> coordinates{index(0), index(1)};
  for (std::size_t d = 0; d < n; ++d) {
    piece.domain.push_back({"k" + std::to_string(d), 0, mAttrs.window[d], true});
    coordinates.push_back(
        {{{d + 2, mAttrs.strides[d], 1}, {r + d, mAttrs.dilations[d], 1}}, -mAttrs.pads[d]});
  }
  piece.inputs = {
      {0, std::move(coordinates), FillValue::NegativeInfinity, logicalAxes(mAttrs.dataAxes), true}};
  piece.outputs = {{0, identityCoordinates(r)}};
  return known(std::move(piece));
}
} // namespace opfusion::operators
