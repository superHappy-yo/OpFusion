// 核验计划引用的数值验收范围与计时事件覆盖，不产生数值观测或执行调度。
#include "PlanConditions.h"
#include "PlanValidation.h"
#include "opfusion/common/Error.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace opfusion::detail {
Validation validatePlanNumerics(const Plan& plan) {
  Validation v;
  const auto& m = *plan.model();
  std::set<CandidateId> candidates;
  std::set<HandoffId> handoffs;
  std::set<std::string> stages;
  bool approximate = false;
  for (const auto& s : plan.selections()) {
    candidates.insert(s.candidate);
    approximate = m.needsNumericAcceptance(m.candidate(s.candidate)) || approximate;
  }
  for (const auto& t : plan.tasks()) {
    if (t.kind == TaskKind::Handoff) {
      const auto& h = m.handoff(t.reference); handoffs.insert(h.id);
      for (const auto& s : h.stages) approximate = m.needsNumericAcceptance(s) || approximate;
    } else if (t.kind == TaskKind::BoundaryStage) {
      const auto& s = m.boundaryStage(t.reference); stages.insert(s.id);
      approximate = m.needsNumericAcceptance(s) || approximate;
    }
  }
  if (!approximate) return v;
  // 整图验收绑定精确组合；不能拼接来自其他候选或交接组合的观测声明。
  bool accepted = false;
  for (const auto& ref : m.numericPolicy().validationRefs) {
    const auto& a = m.numericAcceptance(ref);
    if (a.partition != plan.partition() ||
        std::set<CandidateId>(a.candidates.begin(), a.candidates.end()) != candidates ||
        std::set<HandoffId>(a.handoffs.begin(), a.handoffs.end()) != handoffs ||
        std::set<std::string>(a.boundaryStages.begin(), a.boundaryStages.end()) != stages) continue;
    const auto result = m.validateNumericAcceptance(a);
    if (result.valid()) accepted = true;
  }
  if (!accepted)
    v.unresolved("plan_numeric_acceptance_unknown", plan.id(),
                 "No passing scoped whole-graph acceptance binds this exact candidate/handoff/boundary-adapter combination");
  return v;
}
Validation validatePlanTiming(const Plan& plan,
                              const std::function<bool(const EventId&, const EventId&)>& ordered) {
  Validation v; const auto& m = *plan.model();
  if (!m.boundary().timing) return v;
  const auto& scope = *m.boundary().timing;
  if (scope.startEvent.empty() || scope.finishEvent.empty()) return v;
  std::map<EventId, const Event*> events;
  for (const auto& e : plan.events()) events.emplace(e.id, &e);
  const auto event = [&](const EventId& id) -> const Event& {
    const auto found = events.find(id); require(found != events.end(), "timing_event_missing", "Unknown timing event: " + id);
    return *found->second;
  };
  const auto& start = event(scope.startEvent);
  const auto& finish = event(scope.finishEvent);
  // 时间相等不能代替偏序；偏序覆盖也不能授权任意延后计时起止。
  // 未知/非法时间由事件校验保留，不能用忽略未知项的 max() 伪造完整边界。
  const auto checkExactBoundary = [&](const std::vector<BoundaryBinding>& entries,
                                       const Event& boundary, double emptyTime,
                                       const char* code) {
    double latest = emptyTime;
    for (const auto& entry : entries) {
      const auto& observed = event(entry.event);
      if (!observed.timeUs || !std::isfinite(*observed.timeUs)) {
        return;
      }
      latest = std::max(latest, *observed.timeUs);
    }
    if (boundary.timeUs && std::isfinite(*boundary.timeUs) &&
        !nearPlanTime(*boundary.timeUs, latest)) {
      v.invalid(code, scope.id,
                "Timing must equal the latest declared boundary event, without added delay");
    }
  };
  if (!ordered(scope.startEvent, scope.finishEvent)) v.invalid("timing_event_order", scope.id, "Timing finish must depend on start");
  // 计时只检查声明范围内工作的事件覆盖；本函数不移动事件来缩短指标。
  bool firstRegion = false;
  for (const auto& t : plan.tasks()) {
    bool include = t.kind == TaskKind::Region || t.kind == TaskKind::Handoff;
    if (t.kind == TaskKind::Region && t.start == scope.startEvent) firstRegion = true;
    if (t.kind == TaskKind::BoundaryStage) {
      const auto& stage = m.boundaryStage(t.reference);
      const auto tensor = m.representation(stage.from).tensor;
      const auto& inputs = m.dag().externalInputs();
      const bool input = std::find(inputs.begin(), inputs.end(), tensor) != inputs.end();
      const bool output = m.dag().modelOutputs() &&
          std::find(m.dag().modelOutputs()->begin(), m.dag().modelOutputs()->end(), tensor) != m.dag().modelOutputs()->end();
      include = (input && scope.includesInputTransfers) || (output && scope.includesOutputTransfers);
    }
    if (include && (!ordered(scope.startEvent, t.start) || !ordered(t.complete, scope.finishEvent)))
      v.invalid("timing_task_coverage", t.id, "The declared metric omits required region/handoff/adapter work");
  }
  if (scope.start == TimingStart::FirstRegionStart && !firstRegion)
    v.invalid("timing_compute_start", scope.id, "Compute-start timing must identify an actual first region-start event");
  if (scope.start == TimingStart::InputsAvailable) {
    for (const auto& source : plan.sources()) {
      if (!ordered(source.event, scope.startEvent))
        v.invalid("timing_inputs_available", scope.id, "Inputs-available timing starts only after every actual source is available");
    }
    checkExactBoundary(plan.sources(), start, 0.0, "timing_inputs_available_time");
  }
  if (scope.finish == TimingFinish::OutputsDelivered) {
    for (const auto& sink : plan.sinks()) {
      if (!ordered(sink.event, scope.finishEvent))
        v.invalid("timing_delivery_coverage", scope.id, "Timing cannot end before required output delivery");
    }
    if (!plan.sinks().empty() || start.timeUs) {
      checkExactBoundary(plan.sinks(), finish, plan.sinks().empty() ? *start.timeUs : 0.0,
                          "timing_outputs_delivered_time");
    }
  }
  if (scope.finish == TimingFinish::RegionOutputsReady && m.dag().modelOutputs()) {
    for (auto output : *m.dag().modelOutputs()) {
      bool ready = false;
      for (const auto& b : plan.tensorBindings())
        if (b.tensor == output && ordered(b.ready, scope.finishEvent)) ready = true;
      if (!ready) v.unresolved("timing_output_ready_unknown", std::to_string(output), "No actual complete-ready model output is included in the timing finish");
    }
  }
  return v;
}
}  // namespace opfusion::detail
