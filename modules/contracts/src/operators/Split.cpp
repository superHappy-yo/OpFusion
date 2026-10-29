// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Split.h"

#include "IndexHelpers.h"
#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Split::Split(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
             SplitAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Split::semanticIdentity() {
  return identity("Split");
}
void Split::validateAttributes() const {
  require(!outputs().empty(), "invalid_ports", "Split needs outputs");
  variablePorts(*this, 1, 2, outputs().size());
  for (auto s : mAttrs.sizes) {
    require(s >= 0, "invalid_attribute", "negative Split size");
  }
}
std::vector<TensorSignature> Split::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  const auto d = axisIndex(mAttrs.axis, inputs[0].shape.size());
  Shape sizes = mAttrs.sizes;
  if (sizes.empty()) {
    require(inputs[0].shape[d] % static_cast<std::int64_t>(outputs().size()) == 0, "invalid_shape",
            "Split axis not divisible");
    sizes.assign(outputs().size(),
                 inputs[0].shape[d] / static_cast<std::int64_t>(outputs().size()));
  }
  require(sizes.size() == outputs().size(), "invalid_shape", "Split output count mismatch");
  std::int64_t sum = 0;
  std::vector<TensorSignature> result;
  if (inputs.size() == 2) {
    require(inputs[1].dtype == DataType::Int64, "invalid_dtype", "Split sizes must be INT64");
  }
  for (auto s : sizes) {
    sum = add(sum, s);
    auto out = inputs[0];
    out.shape[d] = s;
    result.push_back(std::move(out));
  }
  require(sum == inputs[0].shape[d], "invalid_shape", "Split sizes do not cover axis");
  return result;
}
Traits Split::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> Split::attributes(const std::vector<TensorSignature> &inputs) const {
  const auto inferred = infer(inputs);
  const auto axis = axisIndex(mAttrs.axis, inputs[0].shape.size());
  Shape sizes;
  for (const auto &out : inferred) {
    sizes.push_back(out.shape[axis]);
  }
  return {{"axis", mAttrs.axis}, {"split", sizes}};
}
IndexSemantics Split::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto outputs = infer(inputs);
  const auto axis = axisIndex(mAttrs.axis, inputs[0].shape.size());
  IndexSemantics result{SemanticStatus::Known, {}, {}};
  std::int64_t offset = 0;
  for (std::size_t port = 0; port < outputs.size(); ++port) {
    IndexPiece piece;
    piece.domain = iteration(outputs[port].shape);
    auto read = identityCoordinates(outputs[port].shape.size());
    read[axis].offset = offset;
    piece.inputs = {{0, read, FillValue::None, inputs[0].axes, true}};
    piece.outputs = {{port, identityCoordinates(outputs[port].shape.size())}};
    result.pieces.push_back(std::move(piece));
    offset = add(offset, outputs[port].shape[axis]);
  }
  return result;
}
std::vector<std::optional<Shape>>
Split::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                        const std::vector<std::optional<Shape>> &values) const {
  return foldCopyIntegers(*this, inputs, values);
}
} // namespace opfusion::operators
