#pragma once

#include "opfusion/graph/Axis.h"
#include "opfusion/graph/Operator.h"

#include <array>
#include <cstdint>
#include <vector>

namespace opfusion::operators {

/// 池化的边界填充值；当前语义仅接受 NegativeInfinity。
enum class PoolPadding { NegativeInfinity, Zero };

/// 空间数组为 [height, width]，pads 为 [top, left, bottom, right]。
struct MaxPoolAttrs {
  Shape window{5, 5};
  Shape strides{1, 1};
  Shape pads{2, 2, 2, 2};
  PoolPadding padding = PoolPadding::NegativeInfinity;
  bool returnIndices = false;
  AxisOrder dataAxes = AxisOrder::NCHW;
  std::size_t spatialDimensions = 2;
  Shape dilations{1, 1};
  bool ceilMode = false;
};

/// 浮点 NCHW 最大池化；floor 窗口规则、负无穷 padding，不返回 indices。
class MaxPool final : public Operator {
public:
  /// 按值保存单输入/输出、窗口及数值要求；窗口/stride 为正、padding 非负。
  /// 只接受 NCHW、负无穷填充且不返回 indices，其他政策构造时拒绝。
  MaxPool(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
          MaxPoolAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例池化属性，引用限实例存续期。
  const MaxPoolAttrs &attrs() const noexcept { return mAttrs; }
  /// 仅据借用浮点输入和窗口属性推导 floor 输出尺寸，检查通道、空间及溢出。
  /// 未知类型或不兼容逻辑轴明确拒绝，不从输出 shape 反推窗口。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯最大值归约语义，不代表已存在目标实现。
  Traits traits() const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  /// 描述空间窗口及 Maximum 归约，越界输入与归约初值均为负无穷。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 检查端口、窗口/步长/填充及当前固定政策，不做来源字段解析。
  void validateAttributes() const override;

private:
  MaxPoolAttrs mAttrs;
};

} // namespace opfusion::operators
