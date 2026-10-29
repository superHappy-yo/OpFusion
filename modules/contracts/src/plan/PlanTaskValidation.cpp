// 把已选候选/适配器映射为任务使用区间；核验端口和依赖，不补资源或执行工作。
#include "PlanValidation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace opfusion::detail {
namespace {

struct TaskSemantics {
  std::optional<double> serviceTimeUs;
  bool aliasOnly = false;
};

struct AdapterStages {
  std::vector<const Stage*> stages;
  std::optional<double> serviceTimeUs;
};

void addResourceActivities(const PlanValidationView& view, PlanUsage& usage,
                            const std::vector<ResourceUse>& resources, const Task& task,
                            double offset, const std::optional<double>& length,
                            Validation& validation) {
  const auto& start = lookupPlanEntry(view.index.events, task.start, "event");
  // 仅登记候选/阶段的实际请求；设备最低规则已在 Model 核验，不能重复生成活动。
  for (const auto& resource : resources) {
    const auto& described = view.model.platform().resource(resource.resource);
    if (described.scope == ResourceScope::Thread || described.scope == ResourceScope::Block) {
      continue;
    }
    const auto available = view.model.boundary().availableResources.find(resource.resource);
    if (available == view.model.boundary().availableResources.end() || !available->second) {
      validation.unresolved("available_resource_unknown", task.id,
                            "Explicit available limit required for resource " + resource.resource);
    } else if (resource.amount > *available->second) {
      validation.invalid("resource_available", task.id,
                         "Activity exceeds the deployment resource limit");
    }
    if (start.timeUs && length) {
      const double begin = *start.timeUs + offset;
      const double end = begin + *length;
      if (!std::isfinite(begin) || !std::isfinite(end)) {
        validation.invalid("schedule_overflow", task.id, "Stage timing overflow");
      } else {
        usage.activities.push_back({task.id, task.start, task.complete, begin, end, resource});
      }
    }
  }
}

void validateCandidatePorts(const PlanValidationView& view, const Task& task,
                             const std::vector<std::string>& actual,
                             const std::vector<PortRepresentation>& ports, Validation& validation) {
  if (actual.size() != ports.size()) {
    validation.invalid("task_ports", task.id, "Task port count mismatch");
  }
  for (const auto& port : ports) {
    if (port.port >= actual.size()) {
      continue;
    }
    const auto& binding = lookupPlanEntry(view.index.bindings, actual[port.port], "binding");
    if (binding.tensor != port.tensor || binding.representation != port.representation) {
      validation.invalid("task_port_representation", task.id,
                         "Task cannot substitute a different candidate port representation");
    }
  }
}

TaskSemantics validateRegionTask(const PlanValidationView& view, PlanUsage& usage,
                                  const Task& task, Validation& validation) {
  const auto& candidate = view.model.candidate(task.reference);
  const auto selection = usage.selections.find(candidate.region);
  if (selection == usage.selections.end() || selection->second->candidate != candidate.id) {
    validation.invalid("task_selection", task.id, "Task must execute exactly the selected q");
  }
  if (!usage.computeTasks.emplace(candidate.id, &task).second) {
    validation.invalid("compute_duplicate", task.id, "Region candidate executed more than once");
  }
  if (task.reuse || !task.stageBindings.empty()) {
    validation.invalid("region_task_fields", task.id,
                       "Region tasks cannot reuse a handoff or carry adapter stage bindings");
  }
  TaskSemantics result;
  if (selection != usage.selections.end() && !selection->second->cost.empty()) {
    const auto& cost = view.model.cost(selection->second->cost);
    result.serviceTimeUs = cost.serviceTimeUs;
    usage.workspaceRequirements[task.id] = cost.workspaceBytesByPool;
    addResourceActivities(view, usage, candidate.config.resources, task,
                           0, result.serviceTimeUs, validation);
    addResourceActivities(view, usage, cost.scopedResources, task,
                           0, result.serviceTimeUs, validation);
  }
  validateCandidatePorts(view, task, task.inputs, candidate.inputs, validation);
  validateCandidatePorts(view, task, task.outputs, candidate.outputs, validation);
  return result;
}

AdapterStages selectAdapterStages(const PlanValidationView& view, PlanUsage& usage,
                                   const Task& task, Validation& validation) {
  AdapterStages result;
  if (task.kind == TaskKind::Handoff) {
    const auto& handoff = view.model.handoff(task.reference);
    validation.append(view.model.validateHandoff(handoff));
    usage.handoffTasks.push_back(&task);
    if (handoff.validation == ExecutionValidation::Failed) {
      validation.invalid("handoff_failed", task.id, "Failed handoff cannot be selected");
    }
    const auto& producer = view.model.candidate(handoff.producer);
    const auto& consumer = view.model.candidate(handoff.consumer);
    const auto producerSelection = usage.selections.find(producer.region);
    const auto consumerSelection = usage.selections.find(consumer.region);
    if (producerSelection == usage.selections.end() ||
        consumerSelection == usage.selections.end() ||
        producerSelection->second->candidate != handoff.producer ||
        consumerSelection->second->candidate != handoff.consumer) {
      validation.invalid("handoff_selection", task.id,
                         "Handoff must join the exact selected candidates");
    }
    for (const auto& stage : handoff.stages) {
      result.stages.push_back(&stage);
    }
    result.serviceTimeUs = handoff.serviceTimeUs();
  } else {
    const auto& stage = view.model.boundaryStage(task.reference);
    validation.append(view.model.validateStage(stage));
    result.stages.push_back(&stage);
    result.serviceTimeUs = stage.durationUs;
    if (task.reuse) {
      validation.invalid("boundary_stage_reuse", task.id,
                         "Replica reuse is an explicit handoff join");
    }
  }
  return result;
}

void validateReplicaReuse(const PlanValidationView& view, const Task& task,
                           const std::vector<const Stage*>& stages, Validation& validation) {
  const auto& prior = lookupPlanEntry(view.index.tasks, *task.reuse, "task");
  if (prior.kind != TaskKind::Handoff || prior.id == task.id ||
      prior.outputs.size() != 1 || task.outputs.size() != 1) {
    validation.invalid("handoff_reuse", task.id,
                       "Reuse must reference another completed handoff replica");
  } else {
    const auto& handoff = view.model.handoff(task.reference);
    const auto& previous = view.model.handoff(prior.reference);
    const auto& wanted = lookupPlanEntry(view.index.bindings, task.outputs[0], "binding");
    const auto& actual = lookupPlanEntry(view.index.bindings, prior.outputs[0], "binding");
    if (handoff.producer != previous.producer || wanted.tensor != actual.tensor ||
        wanted.allocation != actual.allocation || wanted.offsetBytes != actual.offsetBytes ||
        !sameRepresentation(view.model.representation(wanted.representation),
                             view.model.representation(actual.representation))) {
      validation.invalid("handoff_reuse_replica", task.id,
                         "Existing replica must have identical value, representation "
                         "and physical location");
    }
    if (!view.order.precedes(prior.complete, task.start) ||
        !view.order.precedes(wanted.ready, task.start)) {
      validation.invalid("handoff_reuse_ready", task.id,
                         "Reusing a replica still requires its actual completed-ready dependency");
    }
  }
  if (!task.stageBindings.empty()) {
    validation.invalid("handoff_reuse_stages", task.id,
                       "Reused stages must not be re-executed or charged");
  }
  if (!stages.empty() && task.inputs.size() == 1 && task.outputs.size() == 1 &&
      (lookupPlanEntry(view.index.bindings, task.inputs[0], "binding").representation !=
           stages.front()->from ||
       lookupPlanEntry(view.index.bindings, task.outputs[0], "binding").representation !=
           stages.back()->to)) {
    validation.invalid("handoff_reuse_ports", task.id,
                       "Reuse join must still match both selected candidate interfaces");
  }
}

// 顺序阶段的工作区取分池峰值，并保留未知；实际预留仍覆盖整个适配任务。
void accumulateWorkspace(PlanUsage& usage, const Task& task, const Stage& stage) {
  for (const auto& entry : stage.workspaceBytesByPool) {
    auto& required = usage.workspaceRequirements[task.id];
    const auto old = required.find(entry.first);
    if (old == required.end()) {
      required.emplace(entry.first, entry.second);
    } else if (!old->second || !entry.second) {
      old->second = std::nullopt;
    } else {
      old->second = std::max(*old->second, *entry.second);
    }
  }
}

void validateStageBinding(const PlanValidationView& view, PlanUsage& usage, const Task& task,
                           const Stage& stage, std::size_t index, std::size_t stageCount,
                           bool noCopy, Validation& validation) {
  const auto& stageBinding = task.stageBindings[index];
  const auto& from = lookupPlanEntry(view.index.bindings, stageBinding.inputBinding, "binding");
  const auto& to = lookupPlanEntry(view.index.bindings, stageBinding.outputBinding, "binding");
  if (stageBinding.stage != stage.id || from.representation != stage.from ||
      to.representation != stage.to) {
    validation.invalid("stage_binding_representation", task.id,
                       "Stage bindings must match its actual endpoints");
  }
  if (index > 0 && task.stageBindings[index - 1].outputBinding != stageBinding.inputBinding) {
    validation.invalid("stage_binding_continuity", task.id,
                       "Intermediate stage buffer chain is discontinuous");
  }
  if (noCopy && (from.allocation != to.allocation || from.offsetBytes != to.offsetBytes)) {
    validation.invalid("identity_allocation", task.id,
                       "Identity must reuse the same physical allocation, not conjure a copy");
  }
  // 首版保守保留中间与目标缓冲至整个适配任务结束，不提前发布阶段输出。
  useAllocation(view, from.allocation, task.start, task.complete, task.id, validation);
  useAllocation(view, to.allocation, task.start, task.complete, task.id, validation);
  usage.bindingUses[from.id].push_back({task.start, task.complete});
  usage.bindingUses[to.id].push_back({task.start, task.complete});
  if (index > 0 && from.ready != task.complete) {
    validation.invalid("stage_intermediate_ready", task.id,
                       "Private intermediates are visible only at adapter completion");
  }
  if (index + 1 < stageCount) {
    if (to.ready != task.complete) {
      validation.invalid("stage_intermediate_ready", task.id,
                         "Intermediate cannot be published early");
    }
    ++usage.providers[to.id];
  }
}

bool validateAdapterStages(const PlanValidationView& view, PlanUsage& usage, const Task& task,
                            const std::vector<const Stage*>& stages, Validation& validation) {
  if (stages.size() != task.stageBindings.size()) {
    validation.invalid("stage_bindings", task.id,
                       "Every executed stage requires source/target bindings");
  }
  double offset = 0;
  bool knownOffset = true;
  bool aliasOnly = !stages.empty();
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = *stages[i];
    const bool noCopy = stage.kind == StageKind::Identity || stage.kind == StageKind::Synchronize;
    aliasOnly = aliasOnly && noCopy;
    if (knownOffset) {
      addResourceActivities(view, usage, stage.resources, task, offset,
                            stage.durationUs, validation);
    }
    if (stage.durationUs) {
      offset += *stage.durationUs;
    } else {
      knownOffset = false;
    }
    accumulateWorkspace(usage, task, stage);
    if (i < task.stageBindings.size()) {
      validateStageBinding(view, usage, task, stage, i, stages.size(), noCopy, validation);
    }
  }
  if (!task.stageBindings.empty() && task.inputs.size() == 1 && task.outputs.size() == 1 &&
      (task.inputs[0] != task.stageBindings.front().inputBinding ||
       task.outputs[0] != task.stageBindings.back().outputBinding)) {
    validation.invalid("adapter_binding_endpoints", task.id,
                       "Task bindings do not enclose the complete stage chain");
  }
  return aliasOnly;
}

TaskSemantics validateAdapterTask(const PlanValidationView& view, PlanUsage& usage,
                                   const Task& task, Validation& validation) {
  const auto selected = selectAdapterStages(view, usage, task, validation);
  if (task.inputs.size() != 1 || task.outputs.size() != 1) {
    validation.invalid("adapter_task_ports", task.id,
                       "An adapter task has one value input and one output");
  }
  if (task.reuse) {
    validateReplicaReuse(view, task, selected.stages, validation);
    return {0.0, true};
  }
  const bool aliasOnly = validateAdapterStages(view, usage, task, selected.stages, validation);
  return {selected.serviceTimeUs, aliasOnly};
}

void validateTaskBindings(const PlanValidationView& view, PlanUsage& usage, const Task& task,
                           bool aliasOnly, Validation& validation) {
  for (const auto& binding : task.inputs) {
    useBinding(view, usage, binding, task.start, task.complete, task.id, true, validation);
  }
  for (const auto& bindingId : task.outputs) {
    const auto& binding = lookupPlanEntry(view.index.bindings, bindingId, "binding");
    useBinding(view, usage, bindingId, task.start, task.complete, task.id, false, validation);
    if (binding.ready == task.complete) {
      ++usage.providers[bindingId];
    } else if (!aliasOnly || !view.order.precedes(binding.ready, task.start)) {
      validation.invalid("output_ready_event", task.id,
                         "A written output becomes ready only at complete; "
                         "aliases need an earlier actual-ready event");
    }
  }
}

void validateHandoffConnection(const PlanValidationView& view, const Task& task, const Edge& edge,
                               const Task& producer, const Task& consumer,
                               Validation& validation) {
  if (!view.order.precedes(producer.complete, task.start) ||
      !view.order.precedes(task.complete, consumer.start)) {
    validation.invalid("handoff_ready_dependency", task.id,
                       "Producer -> handoff -> consumer dependencies are mandatory, "
                       "including zero-cost identity");
  }
  bool linkedInput = false;
  bool linkedOutput = false;
  for (const auto& bindingId : producer.outputs) {
    const auto& binding = lookupPlanEntry(view.index.bindings, bindingId, "binding");
    if (binding.tensor == edge.tensor && !task.inputs.empty() && bindingId == task.inputs[0]) {
      linkedInput = true;
    }
  }
  for (const auto& bindingId : consumer.inputs) {
    const auto& binding = lookupPlanEntry(view.index.bindings, bindingId, "binding");
    if (binding.tensor == edge.tensor && !task.outputs.empty() && bindingId == task.outputs[0]) {
      linkedOutput = true;
    }
  }
  if (!linkedInput || !linkedOutput) {
    validation.invalid("handoff_binding_join", task.id,
                       "Handoff task must connect the actual selected "
                       "producer/consumer tensor bindings");
  }
}

}  // namespace

void validateTasks(const PlanValidationView& view, PlanUsage& usage, Validation& validation) {
  for (const auto& task : view.plan.tasks()) {
    switch (task.kind) {
      case TaskKind::Region:
      case TaskKind::Handoff:
      case TaskKind::BoundaryStage:
        break;
      default:
        validation.invalid("task_kind", task.id, "Invalid task kind");
    }
    const auto& start = eventFor(view, task.start, EventKind::TaskStart, task.id, validation);
    const auto& complete =
        eventFor(view, task.complete, EventKind::TaskComplete, task.id, validation);
    if (!view.order.precedes(task.start, task.complete)) {
      validation.invalid("task_completion_order", task.id, "Completion must depend on start");
    }
    const auto semantics = task.kind == TaskKind::Region
        ? validateRegionTask(view, usage, task, validation)
        : validateAdapterTask(view, usage, task, validation);
    if (semantics.serviceTimeUs && start.timeUs && complete.timeUs &&
        !nearPlanTime(*complete.timeUs - *start.timeUs, *semantics.serviceTimeUs)) {
      validation.invalid("task_service_time", task.id,
                         "Task interval must equal selected service time; "
                         "queueing belongs before start");
    }
    if (!semantics.serviceTimeUs) {
      validation.unresolved("task_service_unknown", task.id, "Task duration is unknown, not zero");
    }
    validateTaskBindings(view, usage, task, semantics.aliasOnly, validation);
  }
  for (const auto& selection : view.plan.selections()) {
    if (!usage.computeTasks.count(selection.candidate)) {
      validation.unresolved("compute_task_missing", selection.candidate,
                            "Selected candidate needs exactly one compute task");
    }
  }
}

void validateHandoffs(const PlanValidationView& view, const PlanUsage& usage,
                      Validation& validation) {
  std::map<OperatorId, RegionId> owner;
  const auto& partition = view.model.partition(view.plan.partition());
  for (const auto& region : partition.regions()) {
    for (auto member : view.model.region(region).members()) {
      owner.emplace(member, region);
    }
  }
  const auto& edges = view.model.dag().edges();
  // 重复消费端口可共享副本，但见证必须仍绑定所选的精确生产者/消费者候选。
  for (const auto& edge : edges) {
    const auto& from = owner.at(edge.producer);
    const auto& to = owner.at(edge.consumer);
    if (from == to || !usage.selections.count(from) || !usage.selections.count(to)) {
      continue;
    }
    const auto& producerId = usage.selections.at(from)->candidate;
    const auto& consumerId = usage.selections.at(to)->candidate;
    const auto producer = usage.computeTasks.find(producerId);
    const auto consumer = usage.computeTasks.find(consumerId);
    if (producer == usage.computeTasks.end() || consumer == usage.computeTasks.end()) {
      continue;
    }
    bool witnessed = false;
    for (const auto* task : usage.handoffTasks) {
      const auto& handoff = view.model.handoff(task->reference);
      const auto handoffEdge = std::find_if(edges.begin(), edges.end(),
          [&](const Edge& candidate) { return candidate.id == handoff.edge; });
      if (handoffEdge == edges.end() || handoffEdge->tensor != edge.tensor ||
          handoff.producer != producerId || handoff.consumer != consumerId) {
        continue;
      }
      witnessed = true;
      validateHandoffConnection(view, *task, edge, *producer->second, *consumer->second,
                                validation);
    }
    if (!witnessed) {
      validation.unresolved("handoff_missing", std::to_string(edge.id),
                            "No selected exact-candidate join witness; "
                            "same device/type is not a zero-cost default");
    }
  }
}

}  // namespace opfusion::detail
