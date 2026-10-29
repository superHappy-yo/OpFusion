// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/ShapeOf.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
ShapeOf::ShapeOf(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                 ShapeOfAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity ShapeOf::semanticIdentity() {
  return identity("Shape");
}
void ShapeOf::validateAttributes() const {
  ports(*this, 1);
}
std::vector<TensorSignature> ShapeOf::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  const auto rank = static_cast<std::int64_t>(inputs[0].shape.size());
  auto start = mAttrs.start, end = mAttrs.end.value_or(rank);
  if (start < 0) {
    start += rank;
  }
  if (end < 0) {
    end += rank;
  }
  start = std::clamp(start, std::int64_t{0}, rank);
  end = std::clamp(end, std::int64_t{0}, rank);
  return {{{std::max(std::int64_t{0}, end - start)}, DataType::Int64}};
}
Traits ShapeOf::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> ShapeOf::attributes(const std::vector<TensorSignature> &inputs) const {
  const auto values =
      evaluateIntegers(inputs, std::vector<std::optional<Shape>>(inputs.size())).front();
  return {{"start", mAttrs.start},
          {"end", mAttrs.end.value_or(static_cast<std::int64_t>(inputs[0].shape.size()))},
          {"values", *values}};
}
IndexSemantics ShapeOf::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto rank = static_cast<std::int64_t>(inputs[0].shape.size());
  auto start = mAttrs.start;
  if (start < 0) {
    start += rank;
  }
  start = std::clamp(start, std::int64_t{0}, rank);
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.shapeReads = {{0, index(0, 1, 1, start)}};
  piece.outputs = {{0, identityCoordinates(1)}};
  return known(std::move(piece));
}
std::vector<std::optional<opfusion::Shape>>
ShapeOf::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                          const std::vector<std::optional<opfusion::Shape>> &values) const {
  const auto out = infer(inputs);
  (void)values;
  const auto rank = static_cast<std::int64_t>(inputs[0].shape.size());
  auto start = mAttrs.start;
  if (start < 0) {
    start += rank;
  }
  start = std::clamp(start, std::int64_t{0}, rank);
  return {opfusion::Shape(inputs[0].shape.begin() + start,
                          inputs[0].shape.begin() + start + out[0].shape[0])};
}
} // namespace opfusion::operators
