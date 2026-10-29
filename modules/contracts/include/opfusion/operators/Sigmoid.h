#pragma once

#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// Sigmoid 当前没有近似算法或额外属性。
struct SigmoidAttrs {};

/// 浮点逐元素 1/(1+exp(-x)) 语义；不提供数值 kernel。
class Sigmoid final : public Operator {
 public:
  /// 按值保存单输入、单输出和数值要求，不选择近似实现或执行后端。
  Sigmoid(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
          SigmoidAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例属性，引用限实例存续期。
  const SigmoidAttrs& attrs() const noexcept { return mAttrs; }
  /// 仅借用已知浮点输入，保持其逻辑签名；未知/整数类型明确拒绝。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature>& inputs) const override;
  /// 声明纯逐元素语义，不给出近似误差或设备执行保证。
  Traits traits() const override;
  /// 描述恒等坐标访问，函数值仍由 Sigmoid 数学语义定义。
  IndexSemantics indexSemantics(const std::vector<TensorSignature>& inputs) const override;

 protected:
  /// 校验单输入/单输出，浮点签名限制由 infer 检查。
  void validateAttributes() const override;

 private:
  SigmoidAttrs mAttrs;
};

}  // namespace opfusion::operators
