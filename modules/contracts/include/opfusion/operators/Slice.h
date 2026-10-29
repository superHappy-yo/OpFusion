#pragma once

#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// 每个数组覆盖完整 rank；边界非负、limits 右开、strides 为正。
struct SliceAttrs {
  Shape starts;
  Shape limits;
  Shape strides;
};

/// 完整 rank 的正步长切片，不隐式截断越界值或归一化负索引。
class Slice final : public Operator {
public:
  /// 按值保存单输入/输出及完整切片属性；构造时校验数组长度和边界顺序。
  Slice(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
        SliceAttrs attrs, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例切片属性，引用限实例存续期。
  const SliceAttrs &attrs() const noexcept { return mAttrs; }
  /// 仅借用输入推导各维 ceil((limit-start)/stride)，保持 dtype 和轴。
  /// 要求属性覆盖完整 rank 且 limits 不越界，未知类型/负维度明确拒绝。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯、非归约、非广播语义，不视为同坐标逐元素运算。
  Traits traits() const override;
  std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<Shape>> &values) const override;
  /// 返回已验证的具体算子属性，不暴露来源或 JSON 类型。
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  /// 描述 output 迭代器到 start + stride * iterator 的输入访问。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验单输入/输出及非负有序边界、正 stride，不隐式修复属性。
  void validateAttributes() const override;

private:
  SliceAttrs mAttrs;
};

} // namespace opfusion::operators
