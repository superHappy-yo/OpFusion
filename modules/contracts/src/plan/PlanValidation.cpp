// Plan 的选择、事件偏序、物理绑定及来源/交付核验；索引只在本次校验内借用。
#include "PlanValidation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace opfusion::detail {
namespace {

template <class T>
std::map<std::string, const T*> indexRecords(const std::vector<T>& records,
                                            const std::string& kind, Validation& validation) {
  std::map<std::string, const T*> result;
  for (const auto& record : records) {
    if (record.id.empty() || !result.emplace(record.id, &record).second) {
      validation.invalid("identity_invalid", record.id, "Empty or duplicate " + kind + " identity");
    }
  }
  return result;
}

void validateSelectedCost(const Model& model, const Selection& selection,
                          const Candidate& candidate, std::optional<Statistic>& statistic,
                          Validation& validation) {
  if (selection.cost.empty()) {
    validation.unresolved("selected_cost_unknown", candidate.id,
                          "No cost was selected; unknown is not zero");
    return;
  }
  const auto& cost = model.cost(selection.cost);
  if (cost.candidate != candidate.id) {
    validation.invalid("selection_cost", candidate.id,
                       "Cannot splice another q's duration/workspace into this candidate");
  }
  validation.append(model.validateCost(cost));
  if (statistic && (statistic->kind != cost.statistic.kind ||
                    statistic->quantile != cost.statistic.quantile)) {
    validation.invalid("plan_statistic_mixed", candidate.id,
                       "Selected service times have different target statistics");
  }
  statistic = cost.statistic;
}

void validateEvent(const Event& event, Validation& validation) {
  switch (event.kind) {
    case EventKind::Allocate:
    case EventKind::Release:
    case EventKind::TaskStart:
    case EventKind::TaskComplete:
    case EventKind::SourceReady:
    case EventKind::SinkComplete:
    case EventKind::TimingStart:
    case EventKind::TimingFinish:
      break;
    default:
      validation.invalid("event_kind", event.id, "Invalid event kind");
  }
  if (!event.timeUs) {
    validation.unresolved("event_time_unknown", event.id,
                          "No schedule/resource/capacity proof without event times");
  } else if (!std::isfinite(*event.timeUs) || *event.timeUs < 0) {
    validation.invalid("event_time", event.id, "Event time must be finite and nonnegative");
  }
}

}  // namespace

PlanIndex indexPlan(const Plan& plan, Validation& validation) {
  return {indexRecords(plan.events(), "event", validation),
          indexRecords(plan.tasks(), "task", validation),
          indexRecords(plan.allocations(), "allocation", validation),
          indexRecords(plan.tensorBindings(), "tensor binding", validation)};
}

bool EventOrder::precedes(const EventId& before, const EventId& after) const {
  const auto found = ancestors.find(after);
  return before == after || (found != ancestors.end() && found->second.count(before) != 0);
}

bool nearPlanTime(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= 1e-9 * std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
}

void validatePlanModel(const Plan& plan, Validation& validation) {
  const auto& model = *plan.model();
  if (plan.id().empty()) {
    validation.invalid("plan_identity", plan.id(), "Plan identity is required");
  }
  // 未选候选的未知条件不直接污染当前选择；保留全部非法项及部署 IO 未知项。
  // 后续仍逐项验证真正选中的候选、成本和交接，不能因此跳过生产准入。
  const auto modelValidation = model.validate();
  validation.append(modelValidation, false);
  for (const auto& issue : modelValidation.issues) {
    if (issue.kind == IssueKind::Unresolved &&
        (issue.code == "input_residency_unknown" || issue.code == "output_target_unknown")) {
      validation.issues.push_back(issue);
    }
  }
  validation.append(model.boundary().validate(model.platform()));
  if (!model.dag().modelOutputs()) {
    validation.unresolved("model_outputs_unknown", plan.id(),
                          "Terminal outputs do not establish the full model IO contract");
  }
  if (!model.dag().modelInputs()) {
    validation.unresolved("model_inputs_unknown", plan.id(),
                          "Ordered model inputs and parameter roles are not established");
  }
  if (model.numericPolicy().id.empty() || model.dag().registryId().empty() ||
      model.platform().registryId().empty()) {
    validation.unresolved("model_conditions_unknown", plan.id(),
                          "Numeric and registry conditions must be frozen");
  }
}

void validateSelections(const Plan& plan, PlanUsage& usage, Validation& validation) {
  const auto& model = *plan.model();
  const auto& partition = model.partition(plan.partition());
  std::optional<Statistic> statistic;
  for (const auto& selection : plan.selections()) {
    if (!usage.selections.emplace(selection.region, &selection).second) {
      validation.invalid("selection_duplicate", selection.region,
                         "One candidate per region is required");
    }
    if (std::find(partition.regions().begin(), partition.regions().end(), selection.region) ==
        partition.regions().end()) {
      validation.invalid("selection_region", selection.region,
                         "Selected region is not in the chosen partition");
    }
    const auto& candidate = model.candidate(selection.candidate);
    if (candidate.region != selection.region) {
      validation.invalid("selection_candidate", selection.region,
                         "Candidate belongs to another region");
    }
    validation.append(model.validateCandidate(candidate));
    if (candidate.support == Support::Unsupported ||
        candidate.validation == ExecutionValidation::Failed) {
      validation.invalid("selection_not_executable", candidate.id,
                         "Selected candidate is unsupported or failed");
    }
    if (candidate.support != Support::Supported ||
        candidate.validation != ExecutionValidation::Verified) {
      validation.unresolved("selection_unverified", candidate.id,
                            "Production feasibility needs supported and verified candidates");
    }
    validateSelectedCost(model, selection, candidate, statistic, validation);
  }
  for (const auto& region : partition.regions()) {
    if (!usage.selections.count(region)) {
      validation.unresolved("selection_missing", region,
                            "Each chosen partition region requires a candidate and cost");
    }
  }
}

bool buildEventOrder(const Plan& plan, const PlanIndex& index, EventOrder& order,
                      Validation& validation) {
  std::map<EventId, std::size_t> indegree;
  std::map<EventId, std::vector<EventId>> next;
  for (const auto& entry : index.events) {
    indegree.emplace(entry.first, 0);
  }
  for (const auto& event : plan.events()) {
    validateEvent(event, validation);
    std::set<EventId> seen;
    for (const auto& dependency : event.dependsOn) {
      if (!seen.insert(dependency).second) {
        validation.invalid("event_dependency_duplicate", event.id, "Repeated dependency");
      }
      const auto& before = lookupPlanEntry(index.events, dependency, "event");
      next[dependency].push_back(event.id);
      ++indegree[event.id];
      if (before.timeUs && event.timeUs && *before.timeUs > *event.timeUs) {
        validation.invalid("event_time_order", event.id,
                           "Dependent event precedes its prerequisite");
      }
    }
  }
  std::vector<EventId> ready;
  for (const auto& entry : indegree) {
    if (entry.second == 0) {
      ready.push_back(entry.first);
    }
  }
  // 祖先关系只由显式依赖传播；即使两个预测时间相同，也不自动形成先后关系。
  std::size_t visited = 0;
  while (!ready.empty()) {
    const auto event = ready.back();
    ready.pop_back();
    ++visited;
    for (const auto& child : next[event]) {
      order.ancestors[child].insert(event);
      order.ancestors[child].insert(order.ancestors[event].begin(), order.ancestors[event].end());
      if (--indegree[child] == 0) {
        ready.push_back(child);
      }
    }
  }
  if (visited != index.events.size()) {
    validation.invalid("event_cycle", plan.id(), "Event dependency graph is cyclic");
    return false;
  }
  for (const auto& event : plan.events()) {
    if (event.timeUs && std::isfinite(*event.timeUs)) {
      order.horizonUs = std::max(order.horizonUs, *event.timeUs);
    }
  }
  return true;
}

const Event& eventFor(const PlanValidationView& view, const EventId& id, EventKind kind,
                       const std::string& subject, Validation& validation) {
  const auto& event = lookupPlanEntry(view.index.events, id, "event");
  if (event.kind != kind || event.subject != subject) {
    validation.invalid("event_owner", id, "Event kind or subject does not match its owner");
  }
  return event;
}

void useAllocation(const PlanValidationView& view, const AllocationId& allocationId,
                    const EventId& first, const EventId& last, const std::string& owner,
                    Validation& validation) {
  const auto& allocation = lookupPlanEntry(view.index.allocations, allocationId, "allocation");
  if (!allocation.preexisting) {
    if (!allocation.allocate) {
      validation.unresolved("allocation_event_unknown", allocation.id,
                            "Nonresident allocation requires an allocation event");
    } else if (!view.order.precedes(*allocation.allocate, first)) {
      validation.invalid("output_preallocation", owner,
                         "All inputs, outputs and workspace must be allocated "
                         "before activity start");
    }
  }
  // 完整输出就绪后仍可能被读取或传输；释放必须依赖本次使用的结束事件。
  if (allocation.release && !view.order.precedes(last, *allocation.release)) {
    validation.invalid("allocation_early_release", owner,
                       "Allocation released before the last read/transfer/operation completed");
  }
}

void useBinding(const PlanValidationView& view, PlanUsage& usage, const std::string& bindingId,
                 const EventId& first, const EventId& last, const std::string& owner,
                 bool input, Validation& validation) {
  const auto& binding = lookupPlanEntry(view.index.bindings, bindingId, "binding");
  lookupPlanEntry(view.index.events, binding.ready, "event");
  if (input && !view.order.precedes(binding.ready, first)) {
    validation.invalid("input_not_ready", owner,
                       "Consumer must depend on the actual complete-ready event, "
                       "even for identity");
  }
  useAllocation(view, binding.allocation, first, last, owner, validation);
  usage.bindingUses[bindingId].push_back({first, last});
}

void checkAllocationBounds(const Allocation& allocation, Bytes offset, std::optional<Bytes> bytes,
                            const std::string& owner, Validation& validation) {
  if (!allocation.sizeBytes || !bytes) {
    validation.unresolved("allocation_size_unknown", owner,
                          "Binding/reservation physical bytes must be known");
    return;
  }
  if (checkedAdd(offset, *bytes) > *allocation.sizeBytes) {
    validation.invalid("allocation_bounds", owner, "Binding/reservation exceeds its allocation");
  }
}

void validateAllocations(const PlanValidationView& view, Validation& validation) {
  for (const auto& allocation : view.plan.allocations()) {
    const auto& pool = view.model.platform().pool(allocation.pool);
    if (pool.physicalIdentity.empty() || !pool.physicalCapacityBytes) {
      validation.unresolved("physical_capacity_unknown", allocation.id,
                            "Used pools need known physical identity/capacity");
    }
    const auto available = view.model.boundary().availableBytesByPool.find(allocation.pool);
    if (available == view.model.boundary().availableBytesByPool.end() || !available->second) {
      validation.unresolved("available_capacity_unknown", allocation.id,
                            "Used pools need explicit dynamic available capacity");
    }
    if (!allocation.sizeBytes) {
      validation.unresolved("allocation_size_unknown", allocation.id,
                            "Unknown allocation bytes are not zero");
    }
    if (!allocation.alignmentBytes) {
      validation.unresolved("allocation_alignment_unknown", allocation.id,
                            "Allocation alignment is required");
    } else if (*allocation.alignmentBytes == 0 ||
               (*allocation.alignmentBytes & (*allocation.alignmentBytes - 1)) != 0) {
      validation.invalid("allocation_alignment", allocation.id,
                         "Alignment must be a positive power of two");
    }
    double begin = 0;
    double end = view.order.horizonUs;
    if (allocation.preexisting) {
      if (allocation.allocate) {
        validation.invalid("resident_allocation", allocation.id,
                           "A preexisting allocation cannot be allocated again");
      }
    } else if (!allocation.allocate) {
      validation.unresolved("allocation_event_unknown", allocation.id,
                            "Allocation event is required");
    } else {
      const auto& event = eventFor(view, *allocation.allocate, EventKind::Allocate,
                                  allocation.id, validation);
      if (event.timeUs) {
        begin = *event.timeUs;
      }
    }
    if (allocation.release) {
      const auto& event = eventFor(view, *allocation.release, EventKind::Release,
                                  allocation.id, validation);
      if (event.timeUs) {
        end = *event.timeUs;
      }
      if (allocation.allocate && !view.order.precedes(*allocation.allocate, *allocation.release)) {
        validation.invalid("allocation_order", allocation.id, "Release must depend on allocation");
      }
    }
    if (end < begin) {
      validation.invalid("allocation_lifetime", allocation.id, "Negative allocation lifetime");
    }
  }
}

void validateTensorBindings(const PlanValidationView& view, Validation& validation) {
  for (const auto& binding : view.plan.tensorBindings()) {
    const auto& representation = view.model.representation(binding.representation);
    if (representation.tensor != binding.tensor) {
      validation.invalid("binding_tensor", binding.id,
                         "Binding uses another logical value's representation");
    }
    validation.append(representation.validate(view.model.dag().tensor(binding.tensor),
                                               view.model.platform()));
    const auto& allocation =
        lookupPlanEntry(view.index.allocations, binding.allocation, "allocation");
    if (allocation.pool != representation.pool) {
      validation.invalid("binding_pool", binding.id,
                         "Physical allocation and representation pool mismatch");
    }
    checkAllocationBounds(allocation, binding.offsetBytes, representation.actualBytes,
                          binding.id, validation);
    if (representation.alignmentBytes &&
        (*representation.alignmentBytes == 0 ||
         binding.offsetBytes % *representation.alignmentBytes != 0 ||
         (allocation.alignmentBytes &&
          *allocation.alignmentBytes < *representation.alignmentBytes))) {
      validation.invalid("binding_alignment", binding.id,
                         "Allocation/offset cannot satisfy representation alignment");
    }
    lookupPlanEntry(view.index.events, binding.ready, "event");
  }
}

void validateSources(const PlanValidationView& view, PlanUsage& usage, Validation& validation) {
  std::set<std::string> entries;
  const auto& sources = view.model.boundary().inputs;
  for (const auto& source : view.plan.sources()) {
    if (!entries.insert(source.boundaryEntry).second) {
      validation.invalid("source_duplicate", source.boundaryEntry, "Duplicate source binding");
    }
    const auto residency = std::find_if(sources.begin(), sources.end(),
        [&](const Residency& entry) { return entry.id == source.boundaryEntry; });
    require(residency != sources.end(), "source_missing",
            "Source must be derived from Boundary residency");
    const auto& binding = lookupPlanEntry(view.index.bindings, source.tensorBinding, "binding");
    if (binding.tensor != residency->tensor ||
        binding.representation != residency->representation || binding.ready != source.event) {
      validation.invalid("source_binding", source.boundaryEntry,
                         "Source binding must match the precise initial residency");
    }
    const auto& event = eventFor(view, source.event, EventKind::SourceReady,
                                source.boundaryEntry, validation);
    if (event.timeUs && residency->readyUs && !nearPlanTime(*event.timeUs, *residency->readyUs)) {
      validation.invalid("source_time", source.boundaryEntry,
                         "Source time differs from the residency condition");
    }
    ++usage.providers[binding.id];
    useBinding(view, usage, binding.id, source.event, source.event,
               source.boundaryEntry, false, validation);
  }
  for (const auto& source : sources) {
    if (!entries.count(source.id)) {
      validation.unresolved("source_binding_unknown", source.id,
                            "Every declared resident allocation must be bound and counted");
    }
  }
}

void validateSinks(const PlanValidationView& view, PlanUsage& usage, Validation& validation) {
  std::set<std::string> entries;
  const auto& targets = view.model.boundary().outputs;
  for (const auto& sink : view.plan.sinks()) {
    if (!entries.insert(sink.boundaryEntry).second) {
      validation.invalid("sink_duplicate", sink.boundaryEntry, "Duplicate sink binding");
    }
    const auto target = std::find_if(targets.begin(), targets.end(),
        [&](const OutputTarget& entry) { return entry.id == sink.boundaryEntry; });
    require(target != targets.end(), "sink_missing",
            "Sink must be derived from Boundary output targets");
    const auto& binding = lookupPlanEntry(view.index.bindings, sink.tensorBinding, "binding");
    if (binding.tensor != target->tensor || binding.representation != target->representation) {
      validation.invalid("sink_binding", sink.boundaryEntry,
                         "Sink must reach the requested value, pool and representation, "
                         "not merely the last kernel");
    }
    const auto& delivered = eventFor(view, sink.event, EventKind::SinkComplete,
                                     sink.boundaryEntry, validation);
    if (!view.order.precedes(binding.ready, sink.event)) {
      validation.invalid("sink_not_ready", sink.boundaryEntry,
                         "Sink depends on complete output delivery");
    }
    const auto& ready = lookupPlanEntry(view.index.events, binding.ready, "event");
    if (delivered.timeUs && ready.timeUs && !nearPlanTime(*delivered.timeUs, *ready.timeUs)) {
      validation.invalid("sink_completion_time", sink.boundaryEntry,
                         "A virtual sink completes when its target binding is ready; "
                         "additional delivery work must be an explicit task");
    }
    useBinding(view, usage, binding.id, sink.event, sink.event,
               sink.boundaryEntry, true, validation);
  }
  for (const auto& target : targets) {
    if (!entries.count(target.id)) {
      validation.unresolved("sink_binding_unknown", target.id,
                            "Every required output target must be delivered");
    }
  }
  // ready 引用必须由真实来源或任务输出提供；单独声明一个事件不产生值。
  for (const auto& binding : view.plan.tensorBindings()) {
    if (usage.providers[binding.id] == 0) {
      validation.invalid("binding_ready_unproven", binding.id,
                         "Ready event is not supplied by a real source or task output");
    }
    if (usage.providers[binding.id] > 1) {
      validation.invalid("binding_multiple_writers", binding.id,
                         "Binding has multiple writers/sources");
    }
  }
}

void validateEventOwnership(const PlanValidationView& view, Validation& validation) {
  for (const auto& event : view.plan.events()) {
    bool owned = false;
    if (event.kind == EventKind::TaskStart || event.kind == EventKind::TaskComplete) {
      const auto task = view.index.tasks.find(event.subject);
      owned = task != view.index.tasks.end() &&
              (event.kind == EventKind::TaskStart ? task->second->start : task->second->complete) ==
                  event.id;
    } else if (event.kind == EventKind::Allocate || event.kind == EventKind::Release) {
      const auto allocation = view.index.allocations.find(event.subject);
      owned = allocation != view.index.allocations.end() &&
              (event.kind == EventKind::Allocate ? allocation->second->allocate
                                                 : allocation->second->release) ==
                  std::optional<EventId>{event.id};
    } else if (event.kind == EventKind::TimingStart || event.kind == EventKind::TimingFinish) {
      const auto& timing = view.model.boundary().timing;
      owned = timing && event.subject == timing->id &&
              event.id == (event.kind == EventKind::TimingStart ? timing->startEvent
                                                               : timing->finishEvent);
    } else if (event.kind == EventKind::SourceReady || event.kind == EventKind::SinkComplete) {
      const auto& entries = event.kind == EventKind::SourceReady ? view.plan.sources()
                                                                : view.plan.sinks();
      owned = std::any_of(entries.begin(), entries.end(), [&](const BoundaryBinding& entry) {
        return entry.event == event.id && entry.boundaryEntry == event.subject;
      });
    }
    if (!owned) {
      validation.invalid("event_orphan", event.id,
                         "Event is not an actual declared "
                         "task/allocation/source/sink/timing event");
    }
  }
}

}  // namespace opfusion::detail
