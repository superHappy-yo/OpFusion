#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Flatten 的静态逻辑属性，来源适配负责读取格式字段。
struct FlattenAttrs {
  std::int64_t axis = 1;
};
/// Flatten 的输入驱动形状推导；不执行设备工作。
class Flatten final : public Operator {
public:
  Flatten(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
          FlattenAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const FlattenAttrs &attrs() const noexcept { return mAttrs; }
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  Traits traits() const override;
  std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<Shape>> &values) const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  void validateAttributes() const override;

private:
  FlattenAttrs mAttrs;
};
} // namespace opfusion::operators
