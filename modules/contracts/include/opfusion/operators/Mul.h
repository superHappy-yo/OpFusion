#pragma once

#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// Mul 当前没有额外属性；广播与数值要求由公共协议表达。
struct MulAttrs {};

/// 同 dtype、尾轴对齐的多向广播乘法；不引入隐式类型转换。
class Mul final : public Operator {
public:
  /// 取得有序端口和属性的值；必须双输入、单输出，不创建输入副本。
  Mul(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
      MulAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例属性，引用限实例存续期。
  const MulAttrs &attrs() const noexcept { return mAttrs; }
  /// 仅据借用输入推导尾轴广播；dtype 须相同且已知，不依据声明输出补答案。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯逐元素广播语义，不推断时延、流量或设备支持。
  Traits traits() const override;
  std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<Shape>> &values) const override;
  /// 描述推导输出域上的双输入读取；广播单例维固定读取坐标零。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验双输入/单输出，实际签名的广播合法性留到 infer。
  void validateAttributes() const override;

private:
  MulAttrs mAttrs;
};

} // namespace opfusion::operators
