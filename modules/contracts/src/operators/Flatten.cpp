// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Flatten.h"

#include "IndexHelpers.h"
#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Flatten::Flatten(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                 FlattenAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Flatten::semanticIdentity() {
  return identity("Flatten");
}
void Flatten::validateAttributes() const {
  ports(*this, 1);
}
std::vector<TensorSignature> Flatten::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  auto axis = mAttrs.axis;
  const auto rank = static_cast<std::int64_t>(inputs[0].shape.size());
  if (axis < 0) {
    axis += rank;
  }
  require(axis >= 0 && axis <= rank, "invalid_axis", "Flatten axis invalid");
  Shape a(inputs[0].shape.begin(), inputs[0].shape.begin() + axis),
      b(inputs[0].shape.begin() + axis, inputs[0].shape.end());
  return {{{shapeProduct(a), shapeProduct(b)}, inputs[0].dtype}};
}
Traits Flatten::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> Flatten::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"axis", mAttrs.axis}};
}
IndexSemantics Flatten::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  return reshapeIndex(inputs[0], output);
}
std::vector<std::optional<Shape>>
Flatten::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                          const std::vector<std::optional<Shape>> &values) const {
  return foldCopyIntegers(*this, inputs, values);
}
} // namespace opfusion::operators
