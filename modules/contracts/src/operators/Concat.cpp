#include "opfusion/operators/Concat.h"

#include "IndexHelpers.h"
#include "SemanticChecks.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace opfusion::operators {
using namespace detail;

Concat::Concat(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
               ConcatAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(attrs) {
  validateAttributes();
}

SemanticIdentity Concat::semanticIdentity() {
  return identity("Concat");
}

void Concat::validateAttributes() const {
  require(!inputs().empty() && outputs().size() == 1, "invalid_ports",
          "Concat needs inputs and one output");
}

std::vector<TensorSignature> Concat::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, this->inputs().size());
  Shape output = inputs[0].shape;
  require(!output.empty() &&
              output.size() <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
          "invalid_shape", "Concat requires a non-scalar input rank");
  const auto rank = static_cast<std::int64_t>(output.size());
  require(mAttrs.axis >= -rank && mAttrs.axis < rank, "invalid_attribute",
          "Concat axis out of range");
  const auto axis = static_cast<std::size_t>(mAttrs.axis < 0 ? mAttrs.axis + rank : mAttrs.axis);
  output[axis] = 0;
  std::optional<LogicalAxes> axes;
  for (const auto &input : inputs) {
    require(input.shape.size() == output.size(), "invalid_shape", "Concat input rank mismatch");
    for (std::size_t i = 0; i < output.size(); ++i) {
      require(i == axis || input.shape[i] == output[i], "invalid_shape",
              "Concat non-axis dimensions must match");
    }
    output[axis] = add(output[axis], input.shape[axis]);
    axes = mergeAxes(axes, input.axes);
  }
  return {{std::move(output), inputs[0].dtype, axes}};
}

Traits Concat::traits() const {
  return {true, false, false, false};
}

IndexSemantics Concat::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  const auto axis = static_cast<std::size_t>(
      mAttrs.axis < 0 ? mAttrs.axis + static_cast<std::int64_t>(output.shape.size()) : mAttrs.axis);
  IndexSemantics result{SemanticStatus::Known, {}, {}};
  std::int64_t offset = 0;
  // 各输入占用独立半开输出区间；通过坐标偏移回到该输入，而非物化复制任务。
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    IndexPiece piece;
    piece.domain = iteration(output.shape);
    piece.domain[axis].lower = offset;
    piece.domain[axis].upper = add(offset, inputs[port].shape[axis]);
    auto coordinates = identityCoordinates(output.shape.size());
    coordinates[axis].offset = -offset;
    piece.inputs = {{port, std::move(coordinates), FillValue::None, output.axes, true}};
    piece.outputs = {{0, identityCoordinates(output.shape.size())}};
    result.pieces.push_back(std::move(piece));
    offset = add(offset, inputs[port].shape[axis]);
  }
  return result;
}

std::vector<Attribute> Concat::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"axis", mAttrs.axis}};
}

std::vector<std::optional<Shape>>
Concat::evaluateIntegers(const std::vector<TensorSignature> &inputs,
                         const std::vector<std::optional<Shape>> &values) const {
  return foldCopyIntegers(*this, inputs, values);
}
} // namespace opfusion::operators
