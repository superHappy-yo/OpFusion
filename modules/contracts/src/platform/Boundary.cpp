// 部署边界只约束本次输入、交付与动态容量；事件覆盖仍由 Plan 单独核验。
#include "opfusion/platform/Boundary.h"

#include "opfusion/common/Error.h"

#include <cmath>
#include <set>

namespace opfusion {
namespace {

bool nonnegative(double value) {
  return std::isfinite(value) && value >= 0;
}

} // namespace

Validation Boundary::validate(const Platform &p) const {
  Validation v;
  if (id.empty()) {
    v.unresolved("boundary_unknown", id, "Boundary identity is required");
  }
  if (platformId.empty()) {
    v.unresolved("boundary_platform_unknown", id, "Boundary must bind a platform");
  } else if (platformId != p.id()) {
    v.invalid("boundary_platform", id, "Boundary belongs to another platform");
  }
  if (!timing) {
    v.unresolved("timing_scope_unknown", id, "Timing boundaries are required");
  } else {
    const auto &scope = *timing;
    if (scope.id.empty() || scope.startEvent.empty() || scope.finishEvent.empty()) {
      v.unresolved("timing_scope_unknown", id,
                   "Timing identity and concrete event references are required");
    }
    switch (scope.start) {
    case TimingStart::Unknown:
      v.unresolved("timing_start_unknown", id, "Timing start meaning is unknown");
      break;
    case TimingStart::InputsAvailable:
    case TimingStart::FirstRegionStart:
    case TimingStart::BeforeInitialization:
      break;
    default:
      v.invalid("timing_start", id, "Invalid timing start");
    }
    switch (scope.finish) {
    case TimingFinish::Unknown:
      v.unresolved("timing_finish_unknown", id, "Timing finish meaning is unknown");
      break;
    case TimingFinish::RegionOutputsReady:
    case TimingFinish::OutputsDelivered:
      break;
    default:
      v.invalid("timing_finish", id, "Invalid timing finish");
    }
    if (scope.includesInputTransfers && scope.start == TimingStart::FirstRegionStart) {
      v.invalid("timing_input_coverage", id,
                "A compute-only start cannot promise all input transfer coverage");
    }
    if (scope.includesOutputTransfers && scope.finish != TimingFinish::OutputsDelivered) {
      v.invalid("timing_output_coverage", id, "Output transfers require delivered-output timing");
    }
    // 初始化尚无任务记录；明确保留未知，不能把冷启动开销隐含设为零。
    if (scope.includesInitialization || scope.start == TimingStart::BeforeInitialization) {
      v.unresolved("initialization_unmodeled", id,
                   "Initialization tasks are not modeled; their cost cannot be assumed zero");
    }
  }
  std::set<std::string> inputsSeen, outputsSeen;
  for (const auto &input : inputs) {
    if (input.id.empty() || !inputsSeen.insert(input.id).second) {
      v.invalid("residency_identity", input.id, "Empty or duplicate residency");
    }
    if (input.representation.empty() || input.dataRef.empty()) {
      v.unresolved("residency_unknown", input.id, "Representation and data reference are required");
    }
    if (!input.readyUs) {
      v.unresolved("source_ready_unknown", input.id, "Source availability time is required");
    } else if (!nonnegative(*input.readyUs)) {
      v.invalid("source_ready", input.id, "Ready time must be finite and nonnegative");
    }
  }
  for (const auto &output : outputs) {
    if (output.id.empty() || !outputsSeen.insert(output.id).second) {
      v.invalid("output_identity", output.id, "Empty or duplicate output target");
    }
    if (output.representation.empty()) {
      v.unresolved("output_representation_unknown", output.id,
                   "Output target needs a representation");
    }
  }
  for (const auto &available : availableBytesByPool) {
    try {
      const auto &pool = p.pool(available.first);
      if (!available.second) {
        v.unresolved("available_capacity_unknown", available.first,
                     "Dynamic available bytes are unknown");
      } else if (pool.physicalCapacityBytes && *available.second > *pool.physicalCapacityBytes) {
        v.invalid("available_exceeds_physical", available.first,
                  "Dynamic capacity cannot overwrite physical capacity");
      }
    } catch (const Error &error) {
      v.invalid(error.code(), id, error.what());
    }
  }
  for (const auto &available : availableResources) {
    try {
      const auto &resource = p.resource(available.first);
      if (!available.second) {
        v.unresolved("available_resource_unknown", available.first,
                     "Dynamic resource limit is unknown");
      } else if (!nonnegative(*available.second)) {
        v.invalid("available_resource", available.first, "Limit must be finite and nonnegative");
      } else if (resource.capacity && *available.second > *resource.capacity) {
        v.invalid("available_resource", available.first, "Dynamic limit exceeds physical capacity");
      }
    } catch (const Error &error) {
      v.invalid(error.code(), id, error.what());
    }
  }
  return v;
}

Validation TimingPolicy::validate() const {
  Validation v;
  if (start == TimingStart::Unknown)
    v.unresolved("timing_start_unknown", "timing", "timing start unspecified");
  else if (start != TimingStart::InputsAvailable && start != TimingStart::FirstRegionStart &&
           start != TimingStart::BeforeInitialization)
    v.invalid("timing_start", "timing", "invalid start");
  if (finish == TimingFinish::Unknown)
    v.unresolved("timing_finish_unknown", "timing", "timing finish unspecified");
  else if (finish != TimingFinish::RegionOutputsReady && finish != TimingFinish::OutputsDelivered)
    v.invalid("timing_finish", "timing", "invalid finish");
  if (includesInputTransfers && start == TimingStart::FirstRegionStart)
    v.invalid("timing_input_coverage", "timing", "compute start excludes initial transfer");
  if (includesOutputTransfers && finish != TimingFinish::OutputsDelivered)
    v.invalid("timing_output_coverage", "timing", "transfers require delivered-output timing");
  return v;
}

} // namespace opfusion
