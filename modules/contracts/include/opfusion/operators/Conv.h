#pragma once

#include "opfusion/graph/Axis.h"
#include "opfusion/graph/Operator.h"

#include <array>
#include <cstdint>
#include <vector>

namespace opfusion::operators {

/// 一至三维空间数组；pads 按所有 begin 轴、所有 end 轴排列。
struct ConvAttrs {
  Shape strides{1, 1};
  Shape dilations{1, 1};
  Shape pads{0, 0, 0, 0};
  std::int64_t groups = 1;
  AxisOrder dataAxes = AxisOrder::NCHW;
  AxisOrder weightAxes = AxisOrder::OIHW;
  std::size_t spatialDimensions = 2;
  Shape kernelShape;
};

/// 浮点 NC.../OI... 分组 cross-correlation；不含 bias，越界输入填零。
class Conv final : public Operator {
public:
  /// 按值保存端口 [X,W] -> [Y]、属性及数值要求，构造时校验端口和固定政策。
  /// stride/dilation/groups 必须为正，padding 非负；输入 shape 约束留到 infer。
  Conv(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
       ConvAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例属性；轴约定是逻辑顺序，不是物理 stride。
  const ConvAttrs &attrs() const noexcept { return mAttrs; }
  /// 仅借用 X/W 签名与属性推导 Y；校验浮点、通道分组和有效空间窗口。
  /// 未知 dtype/静态维度或不兼容轴明确拒绝，不通过声明 Y 反推答案。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯归约语义，不把区域融合或设备支持写成固有事实。
  Traits traits() const override;
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  /// 描述分组通道、空间窗口与求和归约，越界 X 显式填零。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验两输入/单输出、逻辑轴约定、正 stride/dilation/groups 和非负 pads。
  void validateAttributes() const override;

private:
  ConvAttrs mAttrs;
};

} // namespace opfusion::operators
