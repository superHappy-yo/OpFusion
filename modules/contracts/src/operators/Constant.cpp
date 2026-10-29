// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Constant.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Constant::Constant(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                   ConstantAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Constant::semanticIdentity() {
  return identity("Constant");
}
void Constant::validateAttributes() const {
  variablePorts(*this, 0, 0);
  (void)elements(mAttrs.signature.shape);
  require(bits(mAttrs.signature.dtype).has_value(), "invalid_dtype", "Constant type unknown");
  if (mAttrs.floating) {
    require(mAttrs.floating->size() == elements(mAttrs.signature.shape), "invalid_shape",
            "Constant data length mismatch");
  }
  if (mAttrs.integers) {
    require(mAttrs.integers->size() == elements(mAttrs.signature.shape), "invalid_shape",
            "Constant data length mismatch");
  }
}
std::vector<TensorSignature> Constant::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  return {mAttrs.signature};
}
Traits Constant::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> Constant::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  std::vector<Attribute> result{{"shape", mAttrs.signature.shape},
                                {"dtype", toString(mAttrs.signature.dtype)},
                                {"content_sha256", mAttrs.contentDigest}};
  if (mAttrs.integers) {
    result.push_back({"integers", *mAttrs.integers});
  }
  if (mAttrs.floating) {
    result.push_back({"floating", *mAttrs.floating});
  }
  return result;
}
IndexSemantics Constant::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.outputs = {{0, identityCoordinates(output.shape.size())}};
  return known(std::move(piece));
}
std::vector<std::optional<Shape>>
Constant::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                           const std::vector<std::optional<Shape>> &values) const {
  infer(inputs);
  (void)values;
  return {mAttrs.integers};
}
} // namespace opfusion::operators
