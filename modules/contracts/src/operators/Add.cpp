#include "opfusion/operators/Add.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <utility>

namespace opfusion::operators {
using namespace detail;

Add::Add(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs, AddAttrs attrs,
         NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(attrs) {
  validateAttributes();
}

SemanticIdentity Add::semanticIdentity() {
  return identity("Add");
}

void Add::validateAttributes() const {
  ports(*this, 2);
}

std::vector<TensorSignature> Add::infer(const std::vector<TensorSignature> &inputs) const {
  // 加法与乘法共享广播规则，避免各自维护一套轴对齐和空维语义。
  return binary(*this, inputs);
}

Traits Add::traits() const {
  return {true, true, false, true};
}

IndexSemantics Add::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  return pointwiseIndex(inputs, infer(inputs).front());
}

std::vector<std::optional<Shape>>
Add::evaluateIntegers(const std::vector<TensorSignature> &inputs,
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
                                  values[1]->at(values[1]->size() == 1 ? 0 : i), '+'));
  return {result};
}
} // namespace opfusion::operators
