#pragma once
#include "opfusion/graph/Operator.h"
namespace opfusion::operators {
/// Constant 的静态逻辑属性，来源适配负责读取格式字段。
struct ConstantAttrs {
  TensorSignature signature;
  /// 原始常量内容摘要；大常量不通过属性展示复制全部字节。
  std::string contentDigest;
  std::optional<Shape> integers;
  std::optional<std::vector<double>> floating;
};
/// Constant 的输入驱动形状推导；不执行设备工作。
class Constant final : public Operator {
public:
  Constant(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
           ConstantAttrs attrs = {}, NumericalRequirements requirements = {});
  static SemanticIdentity semanticIdentity();
  const ConstantAttrs &attrs() const noexcept { return mAttrs; }
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
  ConstantAttrs mAttrs;
};
} // namespace opfusion::operators
