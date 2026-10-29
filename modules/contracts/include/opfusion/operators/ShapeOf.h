#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Shape 的静态逻辑属性，来源适配负责读取格式字段。
struct ShapeOfAttrs {
  std::int64_t start = 0;
  std::optional<std::int64_t> end;
};
/// Shape 的输入驱动形状推导；不执行设备工作。
class ShapeOf final : public Operator {
public:
  ShapeOf(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
          ShapeOfAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const ShapeOfAttrs &attrs() const noexcept { return mAttrs; }
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  Traits traits() const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;
  std::vector<std::optional<opfusion::Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<opfusion::Shape>> &values) const override;

protected:
  void validateAttributes() const override;

private:
  ShapeOfAttrs mAttrs;
};
} // namespace opfusion::operators
