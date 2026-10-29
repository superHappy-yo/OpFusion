#pragma once

// Plan 内部存活区间与容量核验；没有 PlanMemory 管理类，也不分配设备内存。
#include "opfusion/plan/Plan.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace opfusion::detail {

/// 分配中的一段字节及其全部开始/结束事件；不能只选预测最晚的一个读者。
struct LiveSpan {
  std::string owner;
  AllocationId allocation;
  Bytes offset = 0;
  Bytes bytes = 0;
  std::vector<EventId> starts;
  std::vector<EventId> finishes;
  std::optional<std::string> binding;
};

/// 借用平台、部署限额和唯一 Allocation，核验初始占用、偏序上界与给定时间表。
/// ordered 仅在同步调用期间借用；保守上界超限返回 Unresolved，不证明全局不可行。
Validation validateAllocationCapacity(const Platform& platform, const Boundary& boundary,
                                      const std::vector<Allocation>& allocations,
                                      const std::vector<Event>& events,
                                      const std::function<bool(const EventId&, const EventId&)>&
                                          ordered);

}  // namespace opfusion::detail
