// 给定时间表的容量、排他复用顺序及偏序保守上界核验；不搜索或执行调度。
#include "PlanValidation.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace opfusion::detail {
namespace {

double peakUsage(const std::vector<const ResourceActivity*>& active) {
  std::vector<std::pair<double, double>> changes;
  for (const auto* activity : active) {
    changes.emplace_back(activity->startUs, activity->use.amount);
    changes.emplace_back(activity->endUs, -activity->use.amount);
  }
  // 同时刻先释放再申请，按半开区间累计已声明活动；不凭时间戳补事件依赖。
  std::sort(changes.begin(), changes.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
  });
  double usage = 0;
  double peak = 0;
  for (const auto& change : changes) {
    usage += change.second;
    peak = std::max(peak, usage);
  }
  return peak;
}

void validateScheduledUsage(const ResourceId& resource, double limit,
                             const std::vector<const ResourceActivity*>& active,
                             Validation& validation) {
  if (peakUsage(active) > limit + 1e-9) {
    validation.invalid("resource_conflict", resource,
                       "Concurrent scheduled activities exceed capacity");
  }
}

// 对两活动合量超过上限的排他复用要求显式顺序；不是任意多活动偏序的容量证明。
void validateResourceOrdering(const EventOrder& order, const ResourceId& resource, double limit,
                               const std::vector<const ResourceActivity*>& active,
                               Validation& validation) {
  for (std::size_t i = 0; i < active.size(); ++i) {
    for (std::size_t j = i + 1; j < active.size(); ++j) {
      const auto& from = *active[i];
      const auto& to = *active[j];
      if (from.owner == to.owner || from.use.amount + to.use.amount <= limit) {
        continue;
      }
      if (from.endUs <= to.startUs && !order.precedes(from.endEvent, to.startEvent)) {
        validation.invalid("resource_order_missing", resource,
                           "Exclusive resource reuse needs explicit event ordering, "
                           "not timestamps alone");
      }
      if (to.endUs <= from.startUs && !order.precedes(to.endEvent, from.startEvent)) {
        validation.invalid("resource_order_missing", resource,
                           "Exclusive resource reuse needs explicit event ordering, "
                           "not timestamps alone");
      }
    }
  }
}

// 每个任务在同一资源上的内部串行阶段只计局部峰值；不同任务不按预测时间互斥。
// 在任一实际超额时刻，最后启动的任务对应一个切点；该切点的上界包含全部存活任务。
// 上界超限只表示现有偏序不足以证明容量，不能据此断言所有时间表均不可行。
void validateResourceCapacityOrder(const EventOrder& order, const ResourceId& resource,
                                    double limit,
                                    const std::vector<const ResourceActivity*>& active,
                                    Validation& validation) {
  std::map<std::string, std::vector<const ResourceActivity*>> tasks;
  for (const auto* activity : active) {
    tasks[activity->owner].push_back(activity);
  }
  std::map<std::string, double> peaks;
  for (const auto& task : tasks) {
    peaks.emplace(task.first, peakUsage(task.second));
  }
  for (const auto& current : tasks) {
    const auto& anchor = *current.second.front();
    double upper = 0;
    for (const auto& task : tasks) {
      const auto& other = *task.second.front();
      if (task.first != current.first) {
        if (order.precedes(other.endEvent, anchor.startEvent)) {
          continue;
        }
        if (other.startEvent != anchor.startEvent &&
            order.precedes(anchor.startEvent, other.startEvent)) {
          continue;
        }
      }
      upper += peaks.at(task.first);
      if (upper > limit + 1e-9) {
        validation.unresolved("resource_capacity_order_unproven", resource,
                              "Task-peak upper bound exceeds capacity; add resource "
                              "ordering or establish a tighter capacity model");
        return;
      }
    }
  }
}

}  // namespace

void validateResources(const PlanValidationView& view, const PlanUsage& usage,
                        Validation& validation) {
  for (const auto& resource : view.model.platform().resources()) {
    std::vector<const ResourceActivity*> active;
    for (const auto& activity : usage.activities) {
      if (activity.use.resource == resource.id && activity.endUs > activity.startUs) {
        active.push_back(&activity);
      }
    }
    if (active.empty()) {
      continue;
    }
    // 库存与本次部署动态限额同时约束同一资源，不相加，也不重新登记规则下限。
    std::optional<double> limit = resource.capacity;
    const auto available = view.model.boundary().availableResources.find(resource.id);
    if (available != view.model.boundary().availableResources.end() && available->second) {
      limit = limit ? std::min(*limit, *available->second) : available->second;
    }
    if (!limit) {
      validation.unresolved("resource_capacity_unknown", resource.id,
                            "No resource conflict proof without a capacity");
      continue;
    }
    validateScheduledUsage(resource.id, *limit, active, validation);
    validateResourceOrdering(view.order, resource.id, *limit, active, validation);
    validateResourceCapacityOrder(view.order, resource.id, *limit, active, validation);
  }
}

}  // namespace opfusion::detail
