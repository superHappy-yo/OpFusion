// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Reshape.h"

#include "IndexHelpers.h"
#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Reshape::Reshape(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                 ReshapeAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Reshape::semanticIdentity() {
  return identity("Reshape");
}
void Reshape::validateAttributes() const {
  variablePorts(*this, 1, 2);
  for (auto d : mAttrs.target) {
    require(d >= -1, "invalid_attribute", "Reshape target dimension invalid");
  }
}
std::vector<TensorSignature> Reshape::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  Shape output = mAttrs.target;
  std::optional<std::size_t> missing;
  std::int64_t product = 1;
  if (inputs.size() == 2) {
    require(inputs[1].dtype == DataType::Int64 &&
                inputs[1].shape == Shape{static_cast<std::int64_t>(output.size())},
            "invalid_shape", "Reshape shape tensor invalid");
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] == -1) {
      require(!missing, "invalid_shape", "multiple inferred Reshape dimensions");
      missing = i;
      continue;
    }
    if (output[i] == 0 && !mAttrs.allowZero) {
      require(i < inputs[0].shape.size(), "invalid_shape", "Reshape zero axis invalid");
      output[i] = inputs[0].shape[i];
    }
    product = multiply(product, output[i]);
  }
  const auto total = shapeProduct(inputs[0].shape);
  if (missing) {
    require(product > 0 && total % product == 0, "invalid_shape",
            "Reshape inferred dimension invalid");
    output[*missing] = total / product;
  } else {
    require(product == total, "invalid_shape", "Reshape element count mismatch");
  }
  return {{output, inputs[0].dtype}};
}
Traits Reshape::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> Reshape::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"shape", mAttrs.target}, {"allowzero", mAttrs.allowZero}};
}
IndexSemantics Reshape::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  return reshapeIndex(inputs[0], output);
}
std::vector<std::optional<Shape>>
Reshape::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                          const std::vector<std::optional<Shape>> &values) const {
  infer(inputs);
  return {values.empty() ? std::nullopt : values[0]};
}
} // namespace opfusion::operators
