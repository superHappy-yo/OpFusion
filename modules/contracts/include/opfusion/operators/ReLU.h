#pragma once

#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// ReLU 当前没有裁剪或斜率属性。
struct ReLUAttrs {};

/// 保持 shape/dtype 的逐元素 max(x, 0) 语义。
class ReLU final : public Operator {
 public:
  /// 按值保存单输入、单输出和数值要求；不隐含来源格式的裁剪或斜率默认值。
  ReLU(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
       ReLUAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例属性，引用限实例存续期。
  const ReLUAttrs& attrs() const noexcept { return mAttrs; }
  /// 仅借用输入，返回相同逻辑签名的副本；未知 dtype 或非静态维度拒绝。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature>& inputs) const override;
  /// 声明纯逐元素、非归约且无广播，不等同于数值 kernel 已存在。
  Traits traits() const override;
  /// 描述输入到输出的恒等坐标关系，保留单例维的轴含义。
  IndexSemantics indexSemantics(const std::vector<TensorSignature>& inputs) const override;

 protected:
  /// 校验单输入/单输出，不增加未声明的裁剪参数。
  void validateAttributes() const override;

 private:
  ReLUAttrs mAttrs;
};

}  // namespace opfusion::operators
