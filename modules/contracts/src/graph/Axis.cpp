#include "opfusion/graph/Axis.h"

#include "opfusion/common/Error.h"

#include <set>

// 只维护逻辑轴知识的一致性；不从张量大小或候选物理排列反推语义。
namespace opfusion {
std::string toString(AxisKind kind) {
  switch (kind) {
  case AxisKind::Unknown:
    return "unknown";
  case AxisKind::Batch:
    return "batch";
  case AxisKind::Channel:
    return "channel";
  case AxisKind::Depth:
    return "depth";
  case AxisKind::Height:
    return "height";
  case AxisKind::Width:
    return "width";
  case AxisKind::OutputChannel:
    return "output_channel";
  case AxisKind::InputChannel:
    return "input_channel";
  case AxisKind::Row:
    return "row";
  case AxisKind::Column:
    return "column";
  }
  throw Error("axis_invalid", "invalid logical axis kind");
}
std::string toString(AxisOrigin origin) {
  switch (origin) {
  case AxisOrigin::Unknown:
    return "unknown";
  case AxisOrigin::Declared:
    return "declared";
  case AxisOrigin::Derived:
    return "derived";
  case AxisOrigin::Mixed:
    return "mixed";
  }
  throw Error("axis_invalid", "invalid axis origin");
}
LogicalAxes logicalAxes(AxisOrder order) {
  switch (order) {
  case AxisOrder::NCW:
    return {{AxisKind::Batch, {}}, {AxisKind::Channel, {}}, {AxisKind::Width, {}}};
  case AxisOrder::OIW:
    return {{AxisKind::OutputChannel, {}}, {AxisKind::InputChannel, {}}, {AxisKind::Width, {}}};
  case AxisOrder::NCDHW:
    return {{AxisKind::Batch, {}},
            {AxisKind::Channel, {}},
            {AxisKind::Depth, {}},
            {AxisKind::Height, {}},
            {AxisKind::Width, {}}};
  case AxisOrder::OIDHW:
    return {{AxisKind::OutputChannel, {}},
            {AxisKind::InputChannel, {}},
            {AxisKind::Depth, {}},
            {AxisKind::Height, {}},
            {AxisKind::Width, {}}};
  case AxisOrder::NCHW:
    return {{AxisKind::Batch, {}},
            {AxisKind::Channel, {}},
            {AxisKind::Height, {}},
            {AxisKind::Width, {}}};
  case AxisOrder::NHWC:
    return {{AxisKind::Batch, {}},
            {AxisKind::Height, {}},
            {AxisKind::Width, {}},
            {AxisKind::Channel, {}}};
  case AxisOrder::OIHW:
    return {{AxisKind::OutputChannel, {}},
            {AxisKind::InputChannel, {}},
            {AxisKind::Height, {}},
            {AxisKind::Width, {}}};
  case AxisOrder::HWIO:
    return {{AxisKind::Height, {}},
            {AxisKind::Width, {}},
            {AxisKind::InputChannel, {}},
            {AxisKind::OutputChannel, {}}};
  }
  throw Error("axis_invalid", "invalid logical axis order");
}
AxisOrder spatialDataOrder(std::size_t dimensions) {
  require(dimensions >= 1 && dimensions <= 3, "unsupported_dimensions",
          "spatial dimensions must be 1, 2 or 3");
  return dimensions == 1 ? AxisOrder::NCW : dimensions == 2 ? AxisOrder::NCHW : AxisOrder::NCDHW;
}
AxisOrder spatialWeightOrder(std::size_t dimensions) {
  require(dimensions >= 1 && dimensions <= 3, "unsupported_dimensions",
          "spatial dimensions must be 1, 2 or 3");
  return dimensions == 1 ? AxisOrder::OIW : dimensions == 2 ? AxisOrder::OIHW : AxisOrder::OIDHW;
}
bool operator==(const Axis &a, const Axis &b) {
  return a.kind == b.kind && a.label == b.label;
}
void validateAxes(const std::optional<LogicalAxes> &axes, std::size_t rank) {
  if (!axes)
    return;
  require(axes->size() == rank, "axis_invalid", "logical axis count differs from rank");
  std::set<AxisKind> kinds;
  std::set<std::string> labels;
  for (const auto &axis : *axes) {
    (void)toString(axis.kind);
    if (axis.kind != AxisKind::Unknown)
      require(kinds.insert(axis.kind).second, "axis_invalid", "duplicate named logical axis");
    if (!axis.label.empty())
      require(labels.insert(axis.label).second, "axis_invalid", "duplicate logical axis label");
  }
}
// 缺省或逐维未知表示没有声明约束；不能因为知识不足就报告冲突。
bool compatibleAxes(const std::optional<LogicalAxes> &a, const std::optional<LogicalAxes> &b) {
  if (!a || !b)
    return true;
  if (a->size() != b->size())
    return false;
  for (std::size_t i = 0; i < a->size(); ++i) {
    const auto &x = (*a)[i];
    const auto &y = (*b)[i];
    if (x.kind != AxisKind::Unknown && y.kind != AxisKind::Unknown && x.kind != y.kind)
      return false;
    if (!x.label.empty() && !y.label.empty() && x.label != y.label)
      return false;
  }
  return true;
}
std::optional<LogicalAxes> mergeAxes(const std::optional<LogicalAxes> &a,
                                     const std::optional<LogicalAxes> &b) {
  require(compatibleAxes(a, b), "axis_mismatch", "conflicting logical axis meanings");
  if (!a)
    return b;
  if (!b)
    return a;
  // 返回新记录，只补未知项；绝不原地改写调用方已发布的轴知识。
  auto result = *a;
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (result[i].kind == AxisKind::Unknown)
      result[i].kind = (*b)[i].kind;
    if (result[i].label.empty())
      result[i].label = (*b)[i].label;
  }
  validateAxes(result, result.size());
  return result;
}
} // namespace opfusion
