#pragma once

#include "opfusion/graph/Axis.h"
#include "opfusion/graph/Operator.h"

#include <vector>

namespace opfusion::operators {

/// 插值方式；当前 Resize 语义只接受 Nearest。
enum class ResizeMode { Nearest, Linear, Cubic };
/// 采样坐标变换；当前只接受 Asymmetric。
enum class CoordinateTransform { Asymmetric, HalfPixel, AlignCorners };
/// 最近邻舍入方式；当前只接受 Floor。
enum class NearestMode { Floor, Ceil, RoundPreferFloor, RoundPreferCeil };
/// 目标比例策略；当前只接受 Stretch。
enum class AspectPolicy { Stretch, NotLarger, NotSmaller };

/// 完整重采样政策；显式记录未支持选项，校验时拒绝而非忽略。
struct ResizeAttrs {
  ResizeMode mode = ResizeMode::Nearest;
  CoordinateTransform coordinateTransform = CoordinateTransform::Asymmetric;
  NearestMode nearestMode = NearestMode::Floor;
  std::vector<double> scales{1.0, 1.0, 2.0, 2.0};
  Shape sizes;
  Shape axes;
  std::vector<double> roi;
  AspectPolicy aspectPolicy = AspectPolicy::Stretch;
  bool antialias = false;
  bool excludeOutside = false;
  double cubicCoefficient = -0.75;
  double extrapolationValue = 0.0;
  AxisOrder dataAxes = AxisOrder::NCHW;
};

/// 仅 NCHW nearest/floor/asymmetric 空间 2 倍缩放的逻辑语义。
class Resize final : public Operator {
public:
  /// 按值保存单输入/输出及采样政策；构造时拒绝不属于当前固定子集的组合。
  Resize(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
         ResizeAttrs attrs = {}, NumericalRequirements requirements = {});
  /// 返回数学语义身份，不包含来源格式或注册副作用。
  static SemanticIdentity semanticIdentity();
  /// 借用本实例采样属性，引用限实例存续期；不触发重采样计算。
  const ResizeAttrs &attrs() const noexcept { return mAttrs; }
  /// 仅据借用的 rank-4 输入将 H/W 乘二，检查静态维度、dtype、轴及溢出。
  /// 未知 dtype 明确拒绝；输出不参与推导，轴约定不转置输入。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override;
  /// 声明纯、非归约、非广播，不把重复采样误当作二元广播。
  Traits traits() const override;
  /// 返回已验证的具体算子属性，不暴露来源或 JSON 类型。
  std::vector<Attribute> attributes(const std::vector<TensorSignature> &inputs) const override;
  /// 描述输出空间坐标 floor(i/2) 的输入访问，不生成数值 kernel。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override;

protected:
  /// 校验当前固定政策组合；包括未支持的枚举值在内均拒绝而非静默忽略。
  void validateAttributes() const override;

private:
  ResizeAttrs mAttrs;
};

} // namespace opfusion::operators
