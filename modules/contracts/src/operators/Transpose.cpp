// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Transpose.h"

#include "IndexHelpers.h"
#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Transpose::Transpose(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                     TransposeAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Transpose::semanticIdentity() {
  return identity("Transpose");
}
void Transpose::validateAttributes() const {
  ports(*this, 1);
}
std::vector<TensorSignature> Transpose::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  Shape perm = mAttrs.permutation;
  const auto rank = inputs[0].shape.size();
  if (perm.empty()) {
    for (std::size_t i = rank; i > 0; --i) {
      perm.push_back(static_cast<std::int64_t>(i - 1));
    }
  }
  require(perm.size() == rank, "invalid_attribute", "Transpose permutation length mismatch");
  std::set<std::size_t> seen;
  Shape output;
  std::optional<LogicalAxes> axes;
  if (inputs[0].axes) {
    axes = LogicalAxes{};
  }
  for (auto p : perm) {
    require(p >= 0, "invalid_axis", "negative permutation axis");
    const auto d = axisIndex(p, rank);
    require(seen.insert(d).second, "invalid_attribute", "duplicate permutation axis");
    output.push_back(inputs[0].shape[d]);
    if (axes) {
      axes->push_back((*inputs[0].axes)[d]);
    }
  }
  return {{output, inputs[0].dtype, axes}};
}
Traits Transpose::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> Transpose::attributes(const std::vector<TensorSignature> &inputs) const {
  infer(inputs);
  auto permutation = mAttrs.permutation;
  if (permutation.empty()) {
    for (std::size_t d = inputs[0].shape.size(); d > 0; --d) {
      permutation.push_back(static_cast<std::int64_t>(d - 1));
    }
  }
  return {{"perm", permutation}};
}
IndexSemantics Transpose::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  auto perm = mAttrs.permutation;
  if (perm.empty()) {
    for (std::size_t d = inputs[0].shape.size(); d > 0; --d) {
      perm.push_back(static_cast<std::int64_t>(d - 1));
    }
  }
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  std::vector<IndexExpression> read(perm.size());
  for (std::size_t d = 0; d < perm.size(); ++d) {
    read[static_cast<std::size_t>(perm[d])] = index(d);
  }
  piece.inputs = {{0, read, FillValue::None, inputs[0].axes, true}};
  piece.outputs = {{0, identityCoordinates(output.shape.size())}};
  return known(std::move(piece));
}
std::vector<std::optional<Shape>>
Transpose::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                            const std::vector<std::optional<Shape>> &values) const {
  return foldCopyIntegers(*this, inputs, values);
}
} // namespace opfusion::operators
