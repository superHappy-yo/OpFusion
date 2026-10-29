// 把逻辑轴映射到声明的物理存储顺序；不凭布局名称改变逻辑 shape 或移动数据。
#include "RepresentationChecks.h"
#include "opfusion/common/Error.h"
#include <algorithm>
#include <numeric>

namespace opfusion::detail {
Validation validateStorageLayout(const Representation& r, const Tensor& tensor) {
  Validation v;
  if (!r.layout) return v;
  std::vector<std::size_t> order;
  std::vector<AxisKind> names;
  switch (*r.layout) {
    case Layout::RowMajor: case Layout::ColumnMajor:
      order.resize(tensor.shape().size()); std::iota(order.begin(), order.end(), 0);
      if (*r.layout == Layout::ColumnMajor) std::reverse(order.begin(), order.end());
      break;
    case Layout::NCHW: names = {AxisKind::Batch, AxisKind::Channel, AxisKind::Height, AxisKind::Width}; break;
    case Layout::NHWC: names = {AxisKind::Batch, AxisKind::Height, AxisKind::Width, AxisKind::Channel}; break;
    case Layout::OIHW: names = {AxisKind::OutputChannel, AxisKind::InputChannel, AxisKind::Height, AxisKind::Width}; break;
    case Layout::HWIO: names = {AxisKind::Height, AxisKind::Width, AxisKind::InputChannel, AxisKind::OutputChannel}; break;
    case Layout::Strided: case Layout::Opaque: return v;
    default: v.invalid("layout_invalid", r.id, "Invalid physical storage layout"); return v;
  }
  // 具名物理排列必须经逻辑轴身份建立置换，不能仅根据 rank 猜测轴含义。
  if (!names.empty()) {
    if (tensor.shape().size() != names.size()) return v;
    const auto& axes = tensor.signature().axes;
    if (!axes) { v.unresolved("layout_axes_unknown", r.id, "Named storage order needs known logical axes"); return v; }
    for (auto name : names) {
      const auto found = std::find_if(axes->begin(), axes->end(), [&](const Axis& axis) { return axis.kind == name; });
      if (found == axes->end()) {
        const bool partial = std::any_of(axes->begin(), axes->end(), [](const Axis& axis) { return axis.kind == AxisKind::Unknown; });
        if (partial) v.unresolved("layout_axes_unknown", r.id, "Logical axes do not yet establish this storage permutation");
        else v.invalid("layout_axes", r.id, "Storage order names different logical axes");
        return v;
      }
      order.push_back(static_cast<std::size_t>(found - axes->begin()));
    }
  }
  if (r.packing != PackingKind::Dense || !r.stridesBytes || !r.paddingElementsByAxis ||
      r.stridesBytes->size() != order.size() || r.paddingElementsByAxis->size() != order.size()) return v;
  try {
    const auto width = bits(r.dtype); if (!width) return v;
    for (auto axis : order)
      if (checkedAdd(static_cast<Bytes>(tensor.shape()[axis]), (*r.paddingElementsByAxis)[axis]) == 0) return v;
    // 从物理最内层累积可寻址跨度，包含 padding；逻辑元素数不能替代物理占用。
    Bytes span = *width / 8;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      const auto axis = *it;
      const auto extent = checkedAdd(static_cast<Bytes>(tensor.shape()[axis]), (*r.paddingElementsByAxis)[axis]);
      if (extent == 0) return v;
      if (extent <= 1) continue;
      if ((*r.stridesBytes)[axis] < span) v.invalid("layout_strides", r.id, "Strides contradict the declared physical axis order");
      span = checkedAdd(span, checkedMultiply(extent - 1, (*r.stridesBytes)[axis]));
    }
  } catch (const Error& e) { v.invalid(e.code(), r.id, e.what()); }
  return v;
}
}  // namespace opfusion::detail
