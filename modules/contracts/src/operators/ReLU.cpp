#include "opfusion/operators/ReLU.h"

#include "SemanticChecks.h"

#include <utility>

namespace opfusion::operators {
using namespace detail;

ReLU::ReLU(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
           ReLUAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(attrs) {
  validateAttributes();
}

SemanticIdentity ReLU::semanticIdentity() { return identity("ReLU"); }

void ReLU::validateAttributes() const { ports(*this, 1); }

std::vector<TensorSignature> ReLU::infer(const std::vector<TensorSignature>& inputs) const {
  // ReLU 不改逻辑签名；复制输入记录而非读取声明输出，也不计算元素值。
  return unary(*this, inputs);
}

Traits ReLU::traits() const { return {true, true, false, false}; }

IndexSemantics ReLU::indexSemantics(const std::vector<TensorSignature>& inputs) const {
  return pointwiseIndex(inputs, infer(inputs).front());
}

}  // namespace opfusion::operators
