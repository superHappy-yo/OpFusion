#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// GlobalAveragePool 的静态逻辑属性，来源适配负责读取格式字段。
struct GlobalAveragePoolAttrs {};
/// GlobalAveragePool 的输入驱动形状推导；不执行设备工作。
class GlobalAveragePool final : public Operator {
public:
  GlobalAveragePool(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                    GlobalAveragePoolAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const GlobalAveragePoolAttrs &attrs() const noexcept { return mAttrs; }
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  Traits traits() const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  void validateAttributes() const override;

private:
  GlobalAveragePoolAttrs mAttrs;
};
} // namespace opfusion::operators
