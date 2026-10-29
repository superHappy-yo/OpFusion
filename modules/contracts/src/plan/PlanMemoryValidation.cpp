// Plan 的工作区/绑定存活区间与既有别名规则；不执行真实内存管理。
#include "PlanValidation.h"

#include "PlanMemory.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace opfusion::detail {
namespace {

bool overlaps(Bytes offsetA, Bytes sizeA, Bytes offsetB, Bytes sizeB) {
  return sizeA != 0 && sizeB != 0 && offsetA < checkedAdd(offsetB, sizeB) &&
         offsetB < checkedAdd(offsetA, sizeA);
}

std::vector<LiveSpan> workspaceSpans(const PlanValidationView& view, const PlanUsage& usage,
                                    Validation& validation) {
  std::set<std::pair<TaskId, PoolId>> reservations;
  std::vector<LiveSpan> spans;
  for (const auto& workspace : view.plan.workspaces()) {
    const auto& task = lookupPlanEntry(view.index.tasks, workspace.task, "task");
    const auto& allocation =
        lookupPlanEntry(view.index.allocations, workspace.allocation, "allocation");
    if (!reservations.emplace(workspace.task, workspace.pool).second) {
      validation.invalid("workspace_duplicate", workspace.task,
                         "One reservation per task/pool; do not duplicate workspace");
    }
    if (allocation.pool != workspace.pool) {
      validation.invalid("workspace_pool", workspace.task,
                         "Workspace allocation is in another pool");
    }
    const auto requirements = usage.workspaceRequirements.find(workspace.task);
    const auto* required = requirements == usage.workspaceRequirements.end()
        ? nullptr : &requirements->second;
    if (!required || !required->count(workspace.pool)) {
      validation.invalid("workspace_unexpected", workspace.task,
                         "Workspace was not declared by the selected q/stages");
    } else if (required->at(workspace.pool) &&
               workspace.sizeBytes < *required->at(workspace.pool)) {
      validation.invalid("workspace_too_small", workspace.task,
                         "Reservation is smaller than this candidate's workspace requirement");
    }
    checkAllocationBounds(allocation, workspace.offsetBytes, workspace.sizeBytes,
                          workspace.task, validation);
    useAllocation(view, allocation.id, task.start, task.complete, workspace.task, validation);
    // 工作区按全任务保留，阶段内峰值不授权在任务完成前复用同一段字节。
    spans.push_back({workspace.task, allocation.id, workspace.offsetBytes, workspace.sizeBytes,
                     {task.start}, {task.complete}, std::nullopt});
  }
  for (const auto& requirement : usage.workspaceRequirements) {
    for (const auto& pool : requirement.second) {
      if (!pool.second) {
        validation.unresolved("workspace_unknown", requirement.first,
                              "Selected workspace requirement is unknown");
      } else if (*pool.second > 0 && !reservations.count({requirement.first, pool.first})) {
        validation.invalid("workspace_reservation_missing", requirement.first,
                           "Workspace must be allocated and reserved for "
                           "the whole region/adapter interval");
      }
    }
  }
  return spans;
}

void appendBindingSpans(const PlanValidationView& view, const PlanUsage& usage,
                         std::vector<LiveSpan>& spans) {
  for (const auto& entry : usage.bindingUses) {
    const auto& binding = lookupPlanEntry(view.index.bindings, entry.first, "binding");
    const auto& representation = view.model.representation(binding.representation);
    lookupPlanEntry(view.index.events, binding.ready, "event");
    // ready 之外还保留每一个读者/传输区间，避免把“已就绪”误当“可释放”。
    std::vector<EventId> starts{binding.ready};
    std::vector<EventId> finishes{binding.ready};
    for (const auto& use : entry.second) {
      lookupPlanEntry(view.index.events, use.first, "event");
      lookupPlanEntry(view.index.events, use.second, "event");
      starts.push_back(use.first);
      finishes.push_back(use.second);
    }
    if (representation.actualBytes) {
      spans.push_back({binding.id, binding.allocation, binding.offsetBytes,
                       *representation.actualBytes, std::move(starts), std::move(finishes),
                       binding.id});
    }
  }
}

bool sameBoundValue(const PlanValidationView& view, const LiveSpan& lhs, const LiveSpan& rhs) {
  if (!lhs.binding || !rhs.binding) {
    return false;
  }
  const auto& from = lookupPlanEntry(view.index.bindings, *lhs.binding, "binding");
  const auto& to = lookupPlanEntry(view.index.bindings, *rhs.binding, "binding");
  return from.tensor == to.tensor && from.offsetBytes == to.offsetBytes &&
         sameRepresentation(view.model.representation(from.representation),
                              view.model.representation(to.representation));
}

bool spanBefore(const EventOrder& order, const LiveSpan& from, const LiveSpan& to) {
  // 所有读者/传输均须先于复用完成；不能按预测时间只挑一个“最后读者”。
  return std::all_of(from.finishes.begin(), from.finishes.end(), [&](const EventId& end) {
    return std::all_of(to.starts.begin(), to.starts.end(),
                       [&](const EventId& start) { return order.precedes(end, start); });
  });
}

}  // namespace

void validateWorkspaceAndAliases(const PlanValidationView& view, const PlanUsage& usage,
                                 Validation& validation) {
  auto spans = workspaceSpans(view, usage, validation);
  appendBindingSpans(view, usage, spans);
  for (std::size_t i = 0; i < spans.size(); ++i) {
    for (std::size_t j = i + 1; j < spans.size(); ++j) {
      const auto& from = spans[i];
      const auto& to = spans[j];
      if (from.allocation != to.allocation ||
          !overlaps(from.offset, from.bytes, to.offset, to.bytes) ||
          sameBoundValue(view, from, to)) {
        continue;
      }
      if (!spanBefore(view.order, from, to) && !spanBefore(view.order, to, from)) {
        validation.invalid("allocation_alias_conflict", from.allocation,
                           "Different live values/workspaces overlap the same allocation bytes");
      }
    }
  }
}

}  // namespace opfusion::detail
