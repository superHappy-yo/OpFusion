#include "opfusion/graph/Operator.h"

#include "opfusion/common/Error.h"

#include <algorithm>
#include <utility>

// 公共校验入口只借用签名；数学推导及属性限制仍由具体算子负责。
namespace opfusion {

Operator::Operator(OperatorId id, SemanticIdentity semantic, std::vector<TensorId> inputs,
                   std::vector<TensorId> outputs, NumericalRequirements requirements)
    : mId(id), mSemantic(std::move(semantic)), mInputs(std::move(inputs)),
      mOutputs(std::move(outputs)), mRequirements(std::move(requirements)) {
  require(id != 0 && !mSemantic.name.empty() && !mSemantic.nameSpace.empty(), "id_invalid",
          "operator identity is incomplete");
  require(!mOutputs.empty(), "port_invalid", "operator must have an output");
  for (auto value : mInputs) {
    require(value != 0, "port_invalid", "null input tensor");
  }
  for (auto value : mOutputs) {
    require(value != 0, "port_invalid", "null output tensor");
  }
  mRequirements.validate();
}

BroadcastInfo Operator::broadcasting(const std::vector<TensorSignature> &inputs) const {
  BroadcastInfo result;
  const auto facts = traits();
  if (!facts.elementwise.value_or(false) || !facts.broadcasting.value_or(false)) {
    return result;
  }
  const auto output = infer(inputs).front();
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    PortBroadcast entry{port, {}};
    const auto offset = output.shape.size() - inputs[port].shape.size();
    for (std::size_t axis = 0; axis < output.shape.size(); ++axis) {
      if (axis < offset || inputs[port].shape[axis - offset] != output.shape[axis]) {
        entry.axes.push_back(axis);
      }
    }
    result.applied = result.applied || !entry.axes.empty();
    result.inputs.push_back(std::move(entry));
  }
  return result;
}

void Operator::validate(const std::vector<TensorSignature> &inputs) const {
  require(inputs.size() == mInputs.size(), "port_invalid",
          "input signature count differs from ports");
  for (const auto &input : inputs) {
    (void)elements(input.shape);
    validateAxes(input.axes, input.shape.size());
  }
  // 先确立输入静态/数值前提，再进入派生类钩子；不把未知 dtype 当默认浮点。
  validateNumerics(inputs);
  validateAttributes();
}

void Operator::validateNumerics(const std::vector<TensorSignature> &signatures) const {
  for (const auto &signature : signatures) {
    require(bits(signature.dtype).has_value(), "dtype_unresolved", "operator dtype is unknown");
    const auto &allowed = mRequirements.allowedTypes;
    require(allowed.empty() ||
                std::find(allowed.begin(), allowed.end(), signature.dtype) != allowed.end(),
            "numeric_mismatch", "tensor dtype violates operator numerical requirements");
  }
}

} // namespace opfusion
