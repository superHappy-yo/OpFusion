#include "opfusion/operators/Sigmoid.h"

#include "SemanticChecks.h"

#include <utility>

namespace opfusion::operators {
using namespace detail;

Sigmoid::Sigmoid(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                 SigmoidAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(attrs) {
  validateAttributes();
}

SemanticIdentity Sigmoid::semanticIdentity() { return identity("Sigmoid"); }

void Sigmoid::validateAttributes() const { ports(*this, 1); }

std::vector<TensorSignature> Sigmoid::infer(const std::vector<TensorSignature>& inputs) const {
  // 保持输入签名，但不能把整数或未知类型按浮点近似规则默认为支持。
  return unary(*this, inputs, true);
}

Traits Sigmoid::traits() const { return {true, true, false, false}; }

IndexSemantics Sigmoid::indexSemantics(const std::vector<TensorSignature>& inputs) const {
  return pointwiseIndex(inputs, infer(inputs).front());
}

}  // namespace opfusion::operators
