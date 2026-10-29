// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Gather.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Gather::Gather(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
               GatherAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Gather::semanticIdentity() {
  return identity("Gather");
}
void Gather::validateAttributes() const {
  ports(*this, 2);
}
std::vector<TensorSignature> Gather::infer(const std::vector<TensorSignature> &inputs) const {
  validate(inputs);
  require(inputs[1].dtype == DataType::Int64 || inputs[1].dtype == DataType::Int32, "invalid_dtype",
          "Gather indices type invalid");
  const auto d = axisIndex(mAttrs.axis, inputs[0].shape.size());
  Shape out(inputs[0].shape.begin(), inputs[0].shape.begin() + d);
  out.insert(out.end(), inputs[1].shape.begin(), inputs[1].shape.end());
  out.insert(out.end(), inputs[0].shape.begin() + d + 1, inputs[0].shape.end());
  return {{out, inputs[0].dtype}};
}
Traits Gather::traits() const {
  return {true, false, false, false};
}
std::vector<Attribute> Gather::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"axis", mAttrs.axis}};
}
IndexSemantics Gather::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto axis = axisIndex(mAttrs.axis, inputs[0].shape.size());
  const auto q = inputs[1].shape.size();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  std::vector<IndexExpression> indices;
  for (std::size_t d = 0; d < q; ++d) {
    indices.push_back(index(axis + d));
  }
  std::vector<IndexExpression> read;
  for (std::size_t d = 0; d < axis; ++d) {
    read.push_back(index(d));
  }
  IndexExpression indirect;
  indirect.lookups = {{1, indices, inputs[0].shape[axis]}};
  read.push_back(std::move(indirect));
  for (std::size_t d = axis + 1; d < inputs[0].shape.size(); ++d) {
    read.push_back(index(d - 1 + q));
  }
  piece.inputs = {{1, indices, FillValue::None, inputs[1].axes, false},
                  {0, read, FillValue::None, inputs[0].axes, false}};
  piece.outputs = {{0, identityCoordinates(output.shape.size())}};
  return known(std::move(piece));
}
std::vector<std::optional<Shape>>
Gather::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                         const std::vector<std::optional<Shape>> &values) const {
  infer(inputs);
  const auto d = axisIndex(mAttrs.axis, inputs[0].shape.size());
  // 即使数据内容未知，已知的 indices 也必须先验证范围。
  if (values[1]) {
    for (auto index : *values[1]) {
      const auto extent = inputs[0].shape[d];
      require(index >= -extent && index < extent, "index_out_of_bounds",
              "Gather index outside input");
    }
  }
  if (!values[0] || !values[1]) {
    return {std::nullopt};
  }
  std::size_t outer = 1, inner = 1;
  for (std::size_t i = 0; i < d; ++i) {
    outer *= static_cast<std::size_t>(inputs[0].shape[i]);
  }
  for (std::size_t i = d + 1; i < inputs[0].shape.size(); ++i) {
    inner *= static_cast<std::size_t>(inputs[0].shape[i]);
  }
  Shape out;
  const auto extent = inputs[0].shape[d];
  for (std::size_t o = 0; o < outer; ++o) {
    for (auto i : *values[1]) {
      if (i < 0) {
        i += extent;
      }
      require(i >= 0 && i < extent, "index_out_of_bounds", "Gather index outside input");
      for (std::size_t j = 0; j < inner; ++j) {
        out.push_back(values[0]->at(
            (o * static_cast<std::size_t>(extent) + static_cast<std::size_t>(i)) * inner + j));
      }
    }
  }
  return {out};
}
} // namespace opfusion::operators
