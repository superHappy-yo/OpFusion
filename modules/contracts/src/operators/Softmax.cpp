// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Softmax.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Softmax::Softmax(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                 SoftmaxAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Softmax::semanticIdentity() {
  return identity("Softmax");
}
void Softmax::validateAttributes() const {
  ports(*this, 1);
}
std::vector<TensorSignature> Softmax::infer(const std::vector<TensorSignature> &inputs) const {
  auto result = unary(*this, inputs, true);
  (void)axisIndex(mAttrs.axis, inputs[0].shape.size());
  return result;
}
Traits Softmax::traits() const {
  return {true, false, true, false};
}
std::vector<Attribute> Softmax::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"axis", mAttrs.axis}};
}
IndexSemantics Softmax::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto axis = axisIndex(mAttrs.axis, output.shape.size());
  const auto rank = output.shape.size();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.domain.push_back({"reduction", 0, output.shape[axis], true});
  piece.reduction = ReductionKind::NormalizedExponential;
  auto row = identityCoordinates(rank);
  row[axis] = index(rank);
  piece.inputs = {{0, row, FillValue::None, inputs[0].axes, false},
                  {0, identityCoordinates(rank), FillValue::None, inputs[0].axes, true,
                   AccessPhase::AfterReduction}};
  piece.outputs = {{0, identityCoordinates(rank)}};
  return known(std::move(piece));
}
} // namespace opfusion::operators
