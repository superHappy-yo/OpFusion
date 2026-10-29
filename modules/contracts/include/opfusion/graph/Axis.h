#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

/// 逻辑维的语义角色；Unknown 保留未知，不根据维度大小或名称猜测。
enum class AxisKind {
  Unknown,
  Batch,
  Channel,
  Depth,
  Height,
  Width,
  OutputChannel,
  InputChannel,
  Row,
  Column
};
/// 轴知识的来源；Mixed 表示声明与推导共同补全，不代表物理布局。
enum class AxisOrigin { Unknown, Declared, Derived, Mixed };
/// 逻辑维序约定，与候选物理排列 Layout 分离。
enum class AxisOrder { NCHW, NHWC, OIHW, HWIO, NCW, OIW, NCDHW, OIDHW };
/// 一至三维空间算子的显式逻辑维序；不涉及物理存储排列。
AxisOrder spatialDataOrder(std::size_t dimensions);
AxisOrder spatialWeightOrder(std::size_t dimensions);

/// 一个逻辑维的角色与可选标签；空标签不声明额外身份。
struct Axis {
  AxisKind kind = AxisKind::Unknown;
  std::string label;
};
/// 按张量维度排列的轴记录；整体缺省与逐维 Unknown 均允许保留未知。
using LogicalAxes = std::vector<Axis>;

/// 返回稳定诊断文本；非法枚举值抛出 axis_invalid。
std::string toString(AxisKind kind);
/// 返回轴来源文本；不将非法枚举归入 Unknown。
std::string toString(AxisOrigin origin);
/// 按逻辑维序构造新轴记录；不重排张量或推断物理 stride。
LogicalAxes logicalAxes(AxisOrder order);
/// 只借用 axes；已知时要求长度等于 rank、已命名角色和非空标签各自唯一。
/// 缺省轴不报错，非法枚举或重复声明抛出 axis_invalid。
void validateAxes(const std::optional<LogicalAxes> &axes, std::size_t rank);
/// 比较借用的轴知识是否冲突；缺省、Unknown 和空标签不制造约束。
/// 本函数不替代 validateAxes 对单份记录的合法性检查。
bool compatibleAxes(const std::optional<LogicalAxes> &a, const std::optional<LogicalAxes> &b);
/// 返回两份兼容轴知识的值副本，只补全未知项；冲突抛出 axis_mismatch。
std::optional<LogicalAxes> mergeAxes(const std::optional<LogicalAxes> &a,
                                     const std::optional<LogicalAxes> &b);
/// 精确比较角色及标签，不把未知项当通配符。
bool operator==(const Axis &a, const Axis &b);

} // namespace opfusion
