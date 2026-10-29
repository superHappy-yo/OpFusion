// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Div.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Div::Div(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs, DivAttrs attrs,
         NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Div::semanticIdentity() {
  return identity("Div");
}
void Div::validateAttributes() const {
  ports(*this, 2);
}
std::vector<TensorSignature> Div::infer(const std::vector<TensorSignature> &inputs) const {
  return binary(*this, inputs);
}
Traits Div::traits() const {
  return {true, true, false, true};
}
std::vector<Attribute> Div::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {};
}
IndexSemantics Div::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  return pointwiseIndex(inputs, infer(inputs).front());
}
std::vector<std::optional<Shape>>
Div::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                      const std::vector<std::optional<Shape>> &values) const {
  const auto out = infer(inputs).front();
  if (!values[0] || !values[1])
    return {std::nullopt};
  require(inputs[0].shape == inputs[1].shape || elements(inputs[0].shape) == 1 ||
              elements(inputs[1].shape) == 1,
          "unsupported", "static integer broadcasting requires equal shapes or scalar");
  Shape result;
  for (Bytes i = 0; i < elements(out.shape); ++i)
    result.push_back(integerValue(values[0]->at(values[0]->size() == 1 ? 0 : i),
                                  values[1]->at(values[1]->size() == 1 ? 0 : i), '/'));
  return {result};
}
} // namespace opfusion::operators
