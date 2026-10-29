#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Softmax 的静态逻辑属性，来源适配负责读取格式字段。
struct SoftmaxAttrs {
  std::int64_t axis = -1;
};
/// Softmax 的输入驱动形状推导；不执行设备工作。
class Softmax final : public Operator {
public:
  Softmax(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
          SoftmaxAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const SoftmaxAttrs &attrs() const noexcept { return mAttrs; }
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  Traits traits() const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  void validateAttributes() const override;

private:
  SoftmaxAttrs mAttrs;
};
} // namespace opfusion::operators
