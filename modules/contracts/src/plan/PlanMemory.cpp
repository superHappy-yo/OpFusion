// 唯一 Allocation 的分池容量核验；分别处理偏序保守上界与给定时间表。
#include "PlanMemory.h"
#include "opfusion/common/Error.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace opfusion::detail {
Validation validateAllocationCapacity(const Platform& platform, const Boundary& boundary,
                                      const std::vector<Allocation>& allocations,
                                      const std::vector<Event>& events,
                                      const std::function<bool(const EventId&, const EventId&)>& ordered) {
  Validation v;
  std::map<EventId, const Event*> eventIndex;
  for (const auto& e : events) eventIndex.emplace(e.id, &e);
  const auto event = [&](const EventId& id) -> const Event& {
    const auto found = eventIndex.find(id);
    require(found != eventIndex.end(), "event_missing", "Unknown allocation event: " + id); return *found->second;
  };
  for (const auto& pool : platform.pools()) {
    std::vector<const Allocation*> used;
    for (const auto& a : allocations) if (a.pool == pool.id) used.push_back(&a);
    if (used.empty()) continue;
    const auto available = boundary.availableBytesByPool.find(pool.id);
    if (!pool.physicalCapacityBytes || available == boundary.availableBytesByPool.end() || !available->second) {
      v.unresolved("pool_capacity_unknown", pool.id, "Physical and deployment capacities are both required"); continue;
    }
    const Bytes limit = std::min(*pool.physicalCapacityBytes, *available->second);
    const auto add = [&](Bytes& sum, Bytes bytes) {
      if (bytes > limit - sum) return false;
      sum += bytes; return true;
    };
    Bytes initial = 0;
    bool initialFits = true, complete = true;
    for (const auto* a : used) {
      if (!a->sizeBytes) { complete = false; v.unresolved("allocation_size_unknown", a->id, "Unknown bytes cannot prove capacity"); continue; }
      if (*a->sizeBytes > limit) v.invalid("allocation_capacity", a->id, "One allocation exceeds physical/deployment capacity, including zero-duration allocations");
      if (a->preexisting && initialFits) initialFits = add(initial, *a->sizeBytes);
    }
    if (!initialFits) v.invalid("pool_initial_capacity", pool.id, "All preexisting allocations are simultaneously charged at plan start");
    bool partialOrderFits = true;
    for (const auto* current : used) {
      if (current->preexisting) continue;
      if (!current->allocate) { complete = false; continue; }
      event(*current->allocate);
      Bytes upper = 0;
      for (const auto* a : used) {
        if (a != current) {
          if (a->release && ordered(*a->release, *current->allocate)) continue;
          if (!a->preexisting && a->allocate && *a->allocate != *current->allocate &&
              ordered(*current->allocate, *a->allocate)) continue;
        }
        // 不能由事件偏序排除的其他分配都可能在此切点存活；
        // 预测时间戳不能把它们从保守上界中删除。
        if (!a->sizeBytes) { complete = false; continue; }
        if (!add(upper, *a->sizeBytes)) { partialOrderFits = false; break; }
      }
    }
    if (!partialOrderFits)
      v.unresolved("pool_capacity_order_unproven", pool.id,
                   "Event-order upper bound exceeds capacity; this is not proof that every schedule is infeasible");
    if (!complete) continue;
    // 独立核验给定完整时间表本身是否超额；这不替代上面的偏序保守核验。
    // 零时长分配已在分配切点计入，不能因扫描半开区间而丢弃容量要求。
    std::vector<std::pair<double, std::pair<bool, Bytes>>> changes;
    bool timed = true;
    for (const auto* a : used) {
      std::optional<double> start = a->preexisting ? std::optional<double>{0.0}
          : a->allocate ? event(*a->allocate).timeUs : std::nullopt;
      const auto finish = a->release ? event(*a->release).timeUs : std::nullopt;
      if (!start || !std::isfinite(*start) || *start < 0 ||
          (a->release && (!finish || !std::isfinite(*finish) || *finish < *start))) { timed = false; break; }
      if (!a->release || *finish > *start) {
        changes.push_back({*start, {true, *a->sizeBytes}});
        if (finish) changes.push_back({*finish, {false, *a->sizeBytes}});
      }
    }
    if (!timed) continue;
    std::sort(changes.begin(), changes.end(), [](const auto& a, const auto& b) {
      return a.first < b.first || (a.first == b.first && a.second.first < b.second.first);
    });
    Bytes live = 0;
    for (const auto& change : changes) {
      if (change.second.first) {
        if (!add(live, change.second.second)) {
          v.invalid("pool_capacity", pool.id, "The supplied timed schedule exceeds available physical bytes"); break;
        }
      } else {
        require(live >= change.second.second, "capacity_accounting", "Release precedes counted allocation");
        live -= change.second.second;
      }
    }
  }
  return v;
}
}  // namespace opfusion::detail
