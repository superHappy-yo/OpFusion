#pragma once

#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// 当前 MatMul 不支持转置或 batch 属性。
struct MatMulAttrs {};

/// 浮点二维 [M,K] @ [K,N] -> [M,N]；K=0 表示零和归约。
class MatMul final : public Operator {
public:
  /// 按值保存端口 [A,B] -> [Y] 及数值要求，不隐式转置或提升为 batch 运算。
  MatMul(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
         MatMulAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不隐含任何来源绑定。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例属性，引用限实例存续期。
  const MatMulAttrs &attrs() const noexcept { return mAttrs; }
  /// 仅据借用的两个二维浮点签名推导 [M,N]，K 必须相等，允许 K=0。
  /// 逻辑轴须兼容 Row/Column，未知类型拒绝，不按声明输出补形状。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯求和归约语义，不承诺设备实现、误差或性能。
  Traits traits() const override;
  BroadcastInfo broadcasting(const std::vector<TensorSignature> &inputs) const override;
  /// 构造 [M,N,K] 迭代域；空 K 保留非归约输出域和零和语义。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验双输入/单输出，不恢复隐含 transpose/batch 属性。
  void validateAttributes() const override;

private:
  MatMulAttrs mAttrs;
};

} // namespace opfusion::operators
