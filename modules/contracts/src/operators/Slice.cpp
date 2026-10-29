#include "opfusion/operators/Slice.h"

#include "IndexHelpers.h"
#include "SemanticChecks.h"

#include <cstddef>
#include <utility>

namespace opfusion::operators {
using namespace detail;

Slice::Slice(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
             SliceAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}

SemanticIdentity Slice::semanticIdentity() {
  return identity("Slice");
}

void Slice::validateAttributes() const {
  ports(*this, 1);
  require(mAttrs.starts.size() == mAttrs.limits.size() &&
              mAttrs.starts.size() == mAttrs.strides.size(),
          "invalid_attribute", "Slice start/limit/stride lengths must match");
  for (std::size_t i = 0; i < mAttrs.starts.size(); ++i) {
    require(mAttrs.starts[i] >= 0 && mAttrs.limits[i] >= mAttrs.starts[i] && mAttrs.strides[i] > 0,
            "invalid_attribute", "Slice needs nonnegative ordered bounds and positive strides");
  }
}

std::vector<TensorSignature> Slice::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, 1);
  require(mAttrs.starts.size() == inputs[0].shape.size(), "invalid_shape",
          "Slice bounds must cover full rank");
  Shape output;
  for (std::size_t i = 0; i < mAttrs.starts.size(); ++i) {
    require(mAttrs.limits[i] <= inputs[0].shape[i], "invalid_shape",
            "Slice limit exceeds input; no clipping");
    const auto length = mAttrs.limits[i] - mAttrs.starts[i];
    // 商与余数表达右开区间的 ceil，避免 length + stride - 1 的额外溢出风险。
    output.push_back(length / mAttrs.strides[i] + (length % mAttrs.strides[i] != 0));
  }
  return {{std::move(output), inputs[0].dtype, inputs[0].axes}};
}

Traits Slice::traits() const {
  return {true, false, false, false};
}

IndexSemantics Slice::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  InputAccess access;
  access.axes = inputs[0].axes;
  access.preservesAxes = true;
  for (std::size_t i = 0; i < output.shape.size(); ++i) {
    access.coordinates.push_back(index(i, mAttrs.strides[i], 1, mAttrs.starts[i]));
  }
  piece.inputs.push_back(std::move(access));
  piece.outputs = {{0, identityCoordinates(output.shape.size())}};
  return known(std::move(piece));
}

std::vector<Attribute> Slice::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"starts", mAttrs.starts}, {"limits", mAttrs.limits}, {"strides", mAttrs.strides}};
}

std::vector<std::optional<Shape>>
Slice::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                        const std::vector<std::optional<Shape>> &values) const {
  return foldCopyIntegers(*this, inputs, values);
}
} // namespace opfusion::operators
