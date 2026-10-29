#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Gather 的静态逻辑属性，来源适配负责读取格式字段。
struct GatherAttrs {
  std::int64_t axis = 0;
};
/// Gather 的输入驱动形状推导；不执行设备工作。
class Gather final : public Operator {
public:
  Gather(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
         GatherAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const GatherAttrs &attrs() const noexcept { return mAttrs; }
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
  GatherAttrs mAttrs;
};
} // namespace opfusion::operators
