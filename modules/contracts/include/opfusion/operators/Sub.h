#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Sub 的静态逻辑属性，来源适配负责读取格式字段。
struct SubAttrs {};
/// Sub 的输入驱动形状推导；不执行设备工作。
class Sub final : public Operator {
public:
  Sub(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
      SubAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const SubAttrs &attrs() const noexcept { return mAttrs; }
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  Traits traits() const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;
  std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<Shape>> &values) const override;

protected:
  void validateAttributes() const override;

private:
  SubAttrs mAttrs;
};
} // namespace opfusion::operators
