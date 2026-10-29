// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/GlobalAveragePool.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
GlobalAveragePool::GlobalAveragePool(OperatorId id, std::vector<TensorId> inputs,
                                     std::vector<TensorId> outputs, GlobalAveragePoolAttrs attrs,
                                     NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity GlobalAveragePool::semanticIdentity() {
  return identity("GlobalAveragePool");
}
void GlobalAveragePool::validateAttributes() const {
  ports(*this, 1);
}
std::vector<TensorSignature>
GlobalAveragePool::infer(const std::vector<TensorSignature> &inputs) const {
  auto result = unary(*this, inputs, true);
  require(inputs[0].shape.size() >= 3, "invalid_shape", "GlobalAveragePool rank must be >=3");
  for (std::size_t d = 2; d < result[0].shape.size(); ++d) {
    require(result[0].shape[d] > 0, "invalid_shape", "empty spatial average");
    result[0].shape[d] = 1;
  }
  return result;
}
Traits GlobalAveragePool::traits() const {
  return {true, false, true, false};
}
std::vector<Attribute>
GlobalAveragePool::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {};
}
IndexSemantics GlobalAveragePool::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto rank = output.shape.size();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.reduction = ReductionKind::Mean;
  std::vector<IndexExpression> read{index(0), index(1)};
  for (std::size_t d = 2; d < rank; ++d) {
    read.push_back(index(piece.domain.size()));
    piece.domain.push_back({"spatial" + std::to_string(d), 0, inputs[0].shape[d], true});
  }
  piece.inputs = {{0, read, FillValue::None, inputs[0].axes, false}};
  piece.outputs = {{0, identityCoordinates(rank)}};
  return known(std::move(piece));
}
} // namespace opfusion::operators
