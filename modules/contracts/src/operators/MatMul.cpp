// 至少二维矩阵乘法及前导批维广播；归约坐标与批广播分别表达。
#include "opfusion/operators/MatMul.h"

#include "SemanticChecks.h"

#include <algorithm>
#include <utility>
namespace opfusion::operators {
using namespace detail;
MatMul::MatMul(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
               MatMulAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(attrs) {
  validateAttributes();
}
SemanticIdentity MatMul::semanticIdentity() {
  return identity("MatMul");
}
void MatMul::validateAttributes() const {
  ports(*this, 2);
}
std::vector<TensorSignature> MatMul::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, 2, true);
  const auto &a = inputs[0].shape;
  const auto &b = inputs[1].shape;
  require(a.size() >= 2 && b.size() >= 2, "unsupported",
          "MatMul vector inputs are not implemented");
  require(a.back() == b[b.size() - 2], "invalid_shape", "MatMul contracting dimensions must match");
  const auto r = std::max(a.size(), b.size());
  Shape output(r, 1);
  for (std::size_t i = 0; i < r - 2; ++i) {
    const auto x = i + a.size() >= r ? a[i + a.size() - r] : 1;
    const auto y = i + b.size() >= r ? b[i + b.size() - r] : 1;
    require(x == y || x == 1 || y == 1, "invalid_shape",
            "MatMul batch dimensions cannot broadcast");
    output[i] = x == 1 ? y : x;
  }
  output[r - 2] = a[a.size() - 2];
  output[r - 1] = b.back();
  std::optional<LogicalAxes> axes;
  if (r == 2) {
    axes = matrixAxes();
    require(compatibleAxes(inputs[0].axes, axes) && compatibleAxes(inputs[1].axes, axes),
            "axis_mismatch", "MatMul requires matrix axes");
  }
  return {{std::move(output), inputs[0].dtype, axes}};
}
Traits MatMul::traits() const {
  return {true, false, true, true};
}
BroadcastInfo MatMul::broadcasting(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  BroadcastInfo info;
  for (std::size_t p = 0; p < inputs.size(); ++p) {
    PortBroadcast entry{p, {}};
    const auto offset = output.shape.size() - inputs[p].shape.size();
    for (std::size_t d = 0; d + 2 < output.shape.size(); ++d) {
      if (d < offset || inputs[p].shape[d - offset] != output.shape[d]) {
        entry.axes.push_back(d);
      }
    }
    info.applied = info.applied || !entry.axes.empty();
    info.inputs.push_back(std::move(entry));
  }
  return info;
}
IndexSemantics MatMul::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto r = output.shape.size();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.domain.push_back({"k", 0, inputs[0].shape.back(), true});
  piece.reduction = ReductionKind::Sum;
  for (std::size_t p = 0; p < 2; ++p) {
    std::vector<IndexExpression> coordinates;
    const auto &shape = inputs[p].shape;
    const auto offset = r - shape.size();
    for (std::size_t d = 0; d + 2 < shape.size(); ++d) {
      coordinates.push_back(shape[d] == 1 ? IndexExpression{} : index(d + offset));
    }
    coordinates.push_back(index(p == 0 ? r - 2 : r));
    coordinates.push_back(index(p == 0 ? r : r - 1));
    piece.inputs.push_back({p, std::move(coordinates), FillValue::None,
                            r == 2 ? std::optional<LogicalAxes>(matrixAxes()) : std::nullopt,
                            false});
  }
  piece.outputs = {{0, identityCoordinates(r)}};
  return known(std::move(piece));
}
} // namespace opfusion::operators
