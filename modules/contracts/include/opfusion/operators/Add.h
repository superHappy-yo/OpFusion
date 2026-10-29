#pragma once

#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// Add 当前没有额外属性；广播与数值要求由公共协议表达。
struct AddAttrs {};

/// 同 dtype、尾轴对齐的多向广播加法；只描述逻辑语义，不执行计算。
class Add final : public Operator {
public:
  /// 取得有序端口和属性的值；必须双输入、单输出，重复输入引用合法。
  Add(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
      AddAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例属性，引用限实例存续期，不持有调用方容器。
  const AddAttrs &attrs() const noexcept { return mAttrs; }
  /// 从借用的两个输入推导广播 shape/dtype/轴；0 与 1 广播为 0，未知类型拒绝。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯逐元素广播语义，不代表设备能力或融合支持。
  Traits traits() const override;
  std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<Shape>> &values) const override;
  /// 在推导输出域内读取各输入；广播单例维用常量零坐标。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验双输入/单输出，签名依赖的广播条件由 infer 检查。
  void validateAttributes() const override;

private:
  AddAttrs mAttrs;
};

} // namespace opfusion::operators
