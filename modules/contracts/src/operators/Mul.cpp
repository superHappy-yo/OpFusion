#include "opfusion/operators/Mul.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <utility>

namespace opfusion::operators {
using namespace detail;

Mul::Mul(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs, MulAttrs attrs,
         NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(attrs) {
  validateAttributes();
}

SemanticIdentity Mul::semanticIdentity() {
  return identity("Mul");
}

void Mul::validateAttributes() const {
  ports(*this, 2);
}

std::vector<TensorSignature> Mul::infer(const std::vector<TensorSignature> &inputs) const {
  // 与 Add 共用逻辑签名推导；数值乘法不是此纯领域实现的职责。
  return binary(*this, inputs);
}

Traits Mul::traits() const {
  return {true, true, false, true};
}

IndexSemantics Mul::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  return pointwiseIndex(inputs, infer(inputs).front());
}

std::vector<std::optional<Shape>>
Mul::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                      const std::vector<std::optional<Shape>> &values) const {
  const auto output = infer(inputs).front();
  if (!values[0] || !values[1])
    return {std::nullopt};
  require(inputs[0].shape == inputs[1].shape || elements(inputs[0].shape) == 1 ||
              elements(inputs[1].shape) == 1,
          "unsupported", "static integer broadcasting requires equal shapes or scalar");
  Shape result;
  for (Bytes i = 0; i < elements(output.shape); ++i)
    result.push_back(integerValue(values[0]->at(values[0]->size() == 1 ? 0 : i),
                                  values[1]->at(values[1]->size() == 1 ? 0 : i), '*'));
  return {result};
}
} // namespace opfusion::operators
