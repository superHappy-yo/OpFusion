#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Gemm 的静态逻辑属性，来源适配负责读取格式字段。
struct GemmAttrs {
  double alpha = 1.0, beta = 1.0;
  bool transposeA = false, transposeB = false;
};
/// Gemm 的输入驱动形状推导；不执行设备工作。
class Gemm final : public Operator {
public:
  Gemm(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
       GemmAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const GemmAttrs &attrs() const noexcept { return mAttrs; }
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  Traits traits() const override;
  BroadcastInfo broadcasting(const std::vector<TensorSignature> &inputs) const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  void validateAttributes() const override;

private:
  GemmAttrs mAttrs;
};
} // namespace opfusion::operators
