// Plan 的纯校验阶段编排与派生任务图；具体内存/资源规则在同目录私有实现中。
#include "opfusion/plan/Plan.h"

#include "PlanConditions.h"
#include "PlanMemory.h"
#include "PlanValidation.h"
#include "opfusion/common/Error.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace opfusion {
namespace {

using NodeKey = std::pair<TaskNodeKind, std::string>;

NodeKey key(const TaskNode& node) { return {node.kind, node.id}; }

TaskNode taskNode(const Task& task) {
  switch (task.kind) {
    case TaskKind::Region:
      return {TaskNodeKind::Region, task.id};
    case TaskKind::Handoff:
      return {TaskNodeKind::Handoff, task.id};
    case TaskKind::BoundaryStage:
      return {TaskNodeKind::BoundaryStage, task.id};
  }
  return {TaskNodeKind::Unknown, task.id};
}

void appendHandoffArcs(const Plan& plan, const Task& task,
                        const std::map<CandidateId, TaskNode>& computes, TaskGraph& graph) {
  const auto& model = *plan.model();
  const auto& handoff = model.handoff(task.reference);
  const auto consumer = computes.find(handoff.consumer);
  if (consumer != computes.end()) {
    const auto& representation = model.representation(handoff.stages.at(0).from);
    graph.arcs.push_back({taskNode(task), consumer->second, representation.tensor});
  }
  if (task.reuse) {
    const auto prior = std::find_if(plan.tasks().begin(), plan.tasks().end(),
        [&](const Task& candidate) { return candidate.id == *task.reuse; });
    if (prior != plan.tasks().end()) {
      graph.arcs.push_back({taskNode(*prior), taskNode(task),
                            model.representation(handoff.stages.at(0).from).tensor});
    }
  }
}

}  // namespace

Validation TaskGraph::validate() const {
  Validation validation;
  std::map<NodeKey, std::size_t> indegree;
  std::map<NodeKey, std::set<NodeKey>> next;
  for (const auto& node : nodes) {
    switch (node.kind) {
      case TaskNodeKind::Source:
      case TaskNodeKind::Region:
      case TaskNodeKind::Handoff:
      case TaskNodeKind::BoundaryStage:
      case TaskNodeKind::Sink:
        break;
      case TaskNodeKind::Unknown:
      default:
        validation.invalid("task_node_kind", node.id, "Invalid derived task node kind");
    }
    if (node.id.empty() || !indegree.emplace(key(node), 0).second) {
      validation.invalid("task_graph_identity", node.id, "Empty or duplicate derived task node");
    }
  }
  for (const auto& arc : arcs) {
    if (!indegree.count(key(arc.from)) || !indegree.count(key(arc.to))) {
      validation.invalid("task_graph_reference", arc.from.id, "Arc endpoint is missing");
      continue;
    }
    if (next[key(arc.from)].insert(key(arc.to)).second) {
      ++indegree[key(arc.to)];
    }
  }
  std::vector<NodeKey> ready;
  for (const auto& entry : indegree) {
    if (entry.second == 0) {
      ready.push_back(entry.first);
    }
  }
  std::size_t visited = 0;
  while (!ready.empty()) {
    const auto node = ready.back();
    ready.pop_back();
    ++visited;
    for (const auto& target : next[node]) {
      if (--indegree[target] == 0) {
        ready.push_back(target);
      }
    }
  }
  if (visited != indegree.size()) {
    validation.invalid("task_graph_cycle", "task_graph",
                       "Derived Source/Task/Sink graph is cyclic");
  }
  return validation;
}

Plan::Plan(Builder builder) : mData(std::move(builder)) {}

TaskGraph Plan::taskGraph() const {
  require(model() != nullptr, "plan_model", "Plan needs a frozen model");
  const auto& model = *mData.model;
  TaskGraph graph;
  // Source/Sink 是部署边界的派生节点，原始逻辑 DAG 不因此增加设备或复制节点。
  std::map<EventId, TaskNode> providers;
  std::map<CandidateId, TaskNode> computes;
  std::map<std::string, const TensorBinding*> bindings;
  for (const auto& binding : tensorBindings()) {
    bindings.emplace(binding.id, &binding);
  }
  for (const auto& source : model.boundary().inputs) {
    graph.nodes.push_back({TaskNodeKind::Source, source.id});
  }
  for (const auto& sink : model.boundary().outputs) {
    graph.nodes.push_back({TaskNodeKind::Sink, sink.id});
  }
  for (const auto& source : sources()) {
    providers.emplace(source.event, TaskNode{TaskNodeKind::Source, source.boundaryEntry});
  }
  for (const auto& task : tasks()) {
    graph.nodes.push_back(taskNode(task));
    providers.emplace(task.complete, taskNode(task));
    if (task.kind == TaskKind::Region) {
      computes.emplace(task.reference, taskNode(task));
    }
  }
  for (const auto& task : tasks()) {
    for (const auto& input : task.inputs) {
      const auto& binding = detail::lookupPlanEntry(bindings, input, "binding");
      const auto provider = providers.find(binding.ready);
      if (provider != providers.end() && key(provider->second) != key(taskNode(task))) {
        graph.arcs.push_back({provider->second, taskNode(task), binding.tensor});
      }
    }
    if (task.kind == TaskKind::Handoff) {
      appendHandoffArcs(*this, task, computes, graph);
    }
  }
  for (const auto& sink : sinks()) {
    const auto& binding = detail::lookupPlanEntry(bindings, sink.tensorBinding, "binding");
    const auto provider = providers.find(binding.ready);
    if (provider != providers.end()) {
      graph.arcs.push_back(
          {provider->second, {TaskNodeKind::Sink, sink.boundaryEntry}, binding.tensor});
    }
  }
  return graph;
}

Validation Plan::validate() const {
  Validation validation;
  if (!model()) {
    validation.invalid("plan_model", id(), "A frozen model is required");
    return validation;
  }
  detail::validatePlanModel(*this, validation);
  const auto index = detail::indexPlan(*this, validation);
  detail::PlanUsage usage;
  detail::EventOrder order;
  try {
    detail::validateSelections(*this, usage, validation);
    // 后续生命周期核验依赖无环偏序；发现环后不能继续把祖先表当有效证明。
    if (!detail::buildEventOrder(*this, index, order, validation)) {
      return validation;
    }
    const detail::PlanValidationView view{*this, *model(), index, order};
    detail::validateAllocations(view, validation);
    detail::validateTensorBindings(view, validation);
    detail::validateSources(view, usage, validation);
    detail::validateTasks(view, usage, validation);
    detail::validateHandoffs(view, usage, validation);
    detail::validateSinks(view, usage, validation);
    // 先收集全部读者与实际请求，再核验生命周期/容量；不回写或优化输入计划。
    detail::validateWorkspaceAndAliases(view, usage, validation);
    detail::validateResources(view, usage, validation);
    const auto ordered = [&](const EventId& from, const EventId& to) {
      return order.precedes(from, to);
    };
    validation.append(detail::validateAllocationCapacity(
        model()->platform(), model()->boundary(), allocations(), events(), ordered));
    detail::validateEventOwnership(view, validation);
    validation.append(detail::validatePlanNumerics(*this));
    validation.append(detail::validatePlanTiming(*this, ordered));
    validation.append(taskGraph().validate());
  } catch (const Error& error) {
    validation.invalid(error.code(), id(), error.what());
  } catch (const std::out_of_range& error) {
    validation.invalid("plan_reference", id(), error.what());
  }
  return validation;
}

}  // namespace opfusion
