#pragma once

#include "opfusion/graph/Operator.h"

#include <cstdint>
#include <vector>

namespace opfusion::operators {

/// axis 按输入 rank 解释，允许 [-rank, rank) 内的负轴。
struct ConcatAttrs {
  std::int64_t axis = 1;
};

/// 同 rank/dtype 张量沿一个轴连接，其他维必须相等。
class Concat final : public Operator {
public:
  /// 按值保存非空有序输入和单输出；顺序决定连接区间，不排序或去重输入。
  Concat(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
         ConcatAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例 axis 声明；负轴尚须结合具体 rank 解释。
  const ConcatAttrs &attrs() const noexcept { return mAttrs; }
  /// 只据借用输入累加连接维，合并兼容轴知识；检查 rank/dtype/其余维及溢出。
  /// 未知类型不默认支持，不读取声明输出决定连接长度。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯、非归约、非广播连接，不暗示免复制或零成本。
  Traits traits() const override;
  std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &inputs,
                   const std::vector<std::optional<Shape>> &values) const override;
  /// 返回已验证的具体算子属性，不暴露来源或 JSON 类型。
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  /// 按端口顺序构造互斥输出区间，各段只读取对应输入。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验非空输入与单输出；axis 范围由输入 rank 决定，留到 infer 检查。
  void validateAttributes() const override;

private:
  ConcatAttrs mAttrs;
};

} // namespace opfusion::operators
