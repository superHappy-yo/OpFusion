// 计算精度政策与精确组合的数值验收声明核验；不计算误差或升级模拟证据。
#include "opfusion/model/Model.h"
#include "opfusion/common/Error.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace opfusion {
namespace {
bool changed(const Model& model, const Representation& r) {
  return (r.dtype != DataType::Unknown && r.dtype != model.dag().tensor(r.tensor).dtype()) ||
         (r.quantparams && r.quantparams->kind != QuantizationKind::None);
}
DataType baselineType(const Model& model, const Operator& op) {
  if (op.requirements().computeType) return *op.requirements().computeType;
  return op.inputs().empty() ? DataType::Unknown : model.dag().tensor(op.inputs().front()).dtype();
}
bool arithmeticChanges(const Model& model, const Candidate& c, const Operator& op) {
  if (!c.config.arithmetic) return false;
  const auto& a = *c.config.arithmetic;
  const auto expected = baselineType(model, op);
  if (a.mode == ArithmeticMode::ReducedPrecision ||
      (expected != DataType::Unknown && a.computeType != DataType::Unknown && a.computeType != expected)) return true;
  if (op.traits().reduction.value_or(false) || op.requirements().accumulationType) {
    const auto accumulation = op.requirements().accumulationType.value_or(expected);
    if (a.accumulationType && accumulation != DataType::Unknown && *a.accumulationType != accumulation) return true;
  }
  return false;
}
void checkBound(const std::optional<double>& observed, const std::optional<double>& bound,
                const std::string& id, const char* name, Validation& v) {
  if (observed && (!std::isfinite(*observed) || *observed < 0))
    v.invalid("numeric_error_invalid", id, name);
  if (bound) {
    if (!observed) v.unresolved("numeric_error_unknown", id, name);
    else if (*observed > *bound) v.invalid("numeric_error_exceeded", id, name);
  }
}
}
const NumericAcceptance& Model::numericAcceptance(const std::string& id) const {
  const auto found = std::find_if(mData.numericAcceptances.begin(), mData.numericAcceptances.end(),
      [&](const NumericAcceptance& a) { return a.id == id; });
  require(found != mData.numericAcceptances.end(), "numeric_acceptance_missing", "Numeric acceptance not found: " + id);
  return *found;
}
bool Model::needsNumericAcceptance(const Candidate& c) const {
  for (const auto& port : c.inputs) if (changed(*this, representation(port.representation))) return true;
  for (const auto& port : c.outputs) if (changed(*this, representation(port.representation))) return true;
  for (auto id : region(c.region).members()) if (arithmeticChanges(*this, c, dag().node(id))) return true;
  return false;
}
bool Model::needsNumericAcceptance(const Stage& stage) const {
  const auto& from = representation(stage.from); const auto& to = representation(stage.to);
  return changed(*this, from) || changed(*this, to) || from.dtype != to.dtype || !(from.quantparams == to.quantparams);
}
Validation Model::validateArithmetic(const Candidate& c) const {
  Validation v;
  if (!c.config.arithmetic) { v.unresolved("arithmetic_unknown", c.id, "Compute/accumulation arithmetic is not established"); return v; }
  const auto& a = *c.config.arithmetic;
  try {
    if (!bits(a.computeType)) v.unresolved("compute_type_unknown", c.id, "Compute dtype is unknown");
    if (a.accumulationType && !bits(*a.accumulationType)) v.unresolved("accumulation_type_unknown", c.id, "Accumulation dtype is unknown");
  } catch (const Error& e) { v.invalid(e.code(), c.id, e.what()); }
  switch (a.mode) {
    case ArithmeticMode::Strict: case ArithmeticMode::ReducedPrecision: break;
    default: v.invalid("arithmetic_mode", c.id, "Invalid arithmetic mode");
  }
  // 全局与逐算子要求都须满足；允许近似不等于允许覆盖显式计算/累加类型。
  const auto exact = [&](const NumericalRequirements& req, const std::string& object) {
    if (req.computeType && a.computeType != DataType::Unknown && a.computeType != *req.computeType)
      v.invalid("compute_requirement", object, "Execution cannot override an explicit compute dtype requirement");
    if (req.accumulationType) {
      if (!a.accumulationType) v.unresolved("accumulation_type_unknown", object, "Required accumulation dtype is missing");
      else if (*a.accumulationType != *req.accumulationType)
        v.invalid("accumulation_requirement", object, "Execution cannot override an explicit accumulation dtype requirement");
    }
    if (!req.allowedTypes.empty() && a.computeType != DataType::Unknown &&
        std::find(req.allowedTypes.begin(), req.allowedTypes.end(), a.computeType) == req.allowedTypes.end())
      v.invalid("compute_requirement", object, "Compute dtype is excluded by numerical requirements");
  };
  exact(numericPolicy().requirements, c.id);
  for (auto id : region(c.region).members()) {
    const auto& op = dag().node(id); exact(op.requirements(), c.id);
    if (op.traits().reduction.value_or(false) && !a.accumulationType)
      v.unresolved("accumulation_type_unknown", c.id, "A reduction needs explicit accumulation dtype");
    if (!op.traits().reduction)
      v.unresolved("accumulation_semantics_unknown", c.id, "Unknown reduction traits cannot establish accumulation requirements");
    if (arithmeticChanges(*this, c, op) &&
        (!op.requirements().allowApproximation || !numericPolicy().requirements.allowApproximation))
      v.invalid("arithmetic_numeric_policy", c.id, "Compute/accumulation changes violate operator or global numerical policy");
  }
  return v;
}
Validation Model::validateNumericAcceptance(const NumericAcceptance& a) const {
  Validation v;
  try {
    if (a.id.empty()) v.invalid("numeric_acceptance_identity", a.id, "Acceptance identity is required");
    if (a.graphId != dag().id() || a.platformId != platform().id() || a.boundaryId != boundary().id ||
        a.numericPolicyId != numericPolicy().id)
      v.invalid("numeric_acceptance_conditions", a.id, "Acceptance belongs to another graph/platform/boundary/policy snapshot");
    const auto& p = partition(a.partition);
    const std::set<RegionId> regions(p.regions().begin(), p.regions().end());
    std::set<CandidateId> candidates; std::set<RegionId> covered;
    for (const auto& id : a.candidates) {
      const auto& c = candidate(id);
      if (!candidates.insert(id).second || !covered.insert(c.region).second || !regions.count(c.region))
        v.invalid("numeric_acceptance_selection", a.id, "Acceptance must identify exactly one candidate per selected region");
    }
    if (covered != regions) v.invalid("numeric_acceptance_selection", a.id, "Acceptance does not cover the complete partition");
    std::set<HandoffId> handoffs;
    for (const auto& id : a.handoffs) {
      const auto& h = handoff(id);
      if (!handoffs.insert(id).second || !candidates.count(h.producer) || !candidates.count(h.consumer))
        v.invalid("numeric_acceptance_handoff", a.id, "Acceptance handoffs must bind its exact selected candidates");
    }
    std::set<std::string> stages;
    for (const auto& id : a.boundaryStages) {
      boundaryStage(id);
      if (!stages.insert(id).second) v.invalid("numeric_acceptance_stage", a.id, "Duplicate boundary adapter identity");
    }
    // 此处核验的是声明及其范围；模拟/解析结论不能转成生产实机数值证据。
    switch (a.kind) {
      case EvidenceKind::RealDevice: break;
      case EvidenceKind::Simulation: case EvidenceKind::Analytical:
        v.unresolved("numeric_acceptance_non_device", a.id, "Simulation/analysis is not production numerical execution evidence"); break;
      default: v.invalid("evidence_kind", a.id, "Invalid evidence kind");
    }
    switch (a.result) {
      case EvidenceResult::Passed: break;
      case EvidenceResult::Failed: case EvidenceResult::Unknown:
        v.unresolved("numeric_acceptance_not_passed", a.id, "This combination has no passing numerical acceptance"); break;
      default: v.invalid("evidence_result", a.id, "Invalid evidence result");
    }
    if (a.method.empty() || a.casesRef.empty() || a.observationRef.empty())
      v.unresolved("numeric_acceptance_scope_unknown", a.id, "Acceptance needs method, data split/case scope and observation reference");
    const auto& requirements = numericPolicy().requirements;
    if (!requirements.absoluteError && !requirements.relativeError)
      v.unresolved("numeric_tolerance_unknown", a.id, "Approximate production acceptance needs an explicit error criterion");
    checkBound(a.absoluteError, a.result == EvidenceResult::Passed ? requirements.absoluteError : std::nullopt, a.id, "absolute error", v);
    checkBound(a.relativeError, a.result == EvidenceResult::Passed ? requirements.relativeError : std::nullopt, a.id, "relative error", v);
    // 整图误差不证明更严格的局部算子误差；缺少独立范围观测时继续保持未知。
    for (const auto& id : a.candidates) {
      const auto& c = candidate(id);
      if (!needsNumericAcceptance(c)) continue;
      for (auto node : region(c.region).members()) {
        const auto& req = dag().node(node).requirements();
        if (req.absoluteError || req.relativeError)
          v.unresolved("operator_numeric_evidence_unknown", a.id, "Local operator error bounds need separately scoped observations");
      }
    }
  } catch (const Error& e) { v.invalid(e.code(), a.id, e.what()); }
  return v;
}
}  // namespace opfusion
