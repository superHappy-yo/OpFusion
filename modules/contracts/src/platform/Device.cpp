// 设备描述的公共不变量与声明规则查询；不因派生类型不同而绕过 CPU 分配约束。
#include "opfusion/platform/Device.h"

#include "opfusion/common/Error.h"

#include <cmath>
#include <set>
#include <utility>

namespace opfusion {

Device::Device(DeviceDescription description) : mDescription(std::move(description)) {
  require(!mDescription.id.empty(), "identity_invalid", "Device identity must not be empty");
}

ResourceRequirements DeviceResourceRules::requirements(DeviceWork work,
                                                       const DeviceId& device) const {
  ResourceRequirements result;
  switch (work) {
    case DeviceWork::Compute:
      result.minimumUses = compute;
      break;
    case DeviceWork::LocalConversion:
      result.minimumUses = localConversion;
      break;
    default:
      result.validation.invalid("device_work", device, "Invalid device work category");
      return result;
  }
  if (!result.minimumUses) {
    result.validation.unresolved("device_resource_rules_unknown", device,
                                 "Required resources for this work are not established");
  }
  return result;
}

Validation validateDeviceDescription(const DeviceDescription& d) {
  Validation v;
  if (d.id.empty()) {
    v.invalid("identity_invalid", d.id, "Device identity must not be empty");
  }
  switch (d.kind) {
    case DeviceKind::Unknown:
      v.unresolved("device_kind_unknown", d.id, "Device classification is unknown");
      break;
    case DeviceKind::Cpu:
    case DeviceKind::Gpu:
    case DeviceKind::Accelerator:
    case DeviceKind::Other:
      break;
    default:
      v.invalid("device_kind", d.id, "Invalid device classification");
  }
  if (d.nodeId.empty() || d.architecture.empty() || d.instance.empty() || d.configuration.empty()) {
    v.unresolved("device_conditions_unknown", d.id,
                 "Node, architecture, instance and configuration must be explicit");
  }
  std::set<std::string> backends;
  for (const auto& backend : d.backends) {
    if (backend.empty() || !backends.insert(backend).second) {
      v.invalid("backend_identity", d.id, "Empty or duplicate backend");
    }
  }
  if (d.backends.empty()) {
    v.unresolved("backends_unknown", d.id, "No backend catalog was established");
  }
  if (d.kind == DeviceKind::Cpu && !d.cpuAllocation) {
    v.unresolved("cpu_allocation_unknown", d.id, "CPU quota, affinity and NUMA placement are unknown");
  }
  if (d.cpuAllocation) {
    if (d.kind != DeviceKind::Cpu) {
      v.invalid("cpu_allocation_device", d.id, "CPU allocation belongs only to a CPU device");
    }
    const auto& cpu = *d.cpuAllocation;
    if (!cpu.quotaCores) {
      v.unresolved("cpu_quota_unknown", d.id, "CPU quota is unknown");
    } else if (!std::isfinite(*cpu.quotaCores) || *cpu.quotaCores < 0) {
      v.invalid("cpu_quota", d.id, "CPU quota must be finite and nonnegative");
    }
    if (!cpu.affinity) {
      v.unresolved("cpu_affinity_unknown", d.id, "CPU affinity is unknown");
    } else {
      const std::set<std::uint32_t> cores(cpu.affinity->begin(), cpu.affinity->end());
      if (cores.size() != cpu.affinity->size()) {
        v.invalid("cpu_affinity", d.id, "Duplicate affinity CPU");
      }
      if (cpu.quotaCores && *cpu.quotaCores > cores.size()) {
        v.invalid("cpu_quota", d.id, "Quota exceeds assigned affinity CPUs");
      }
    }
    // “不适用”必须显式声明；没有节点号不能自动当作非 NUMA 环境。
    switch (cpu.numaBinding) {
      case NumaBinding::Unknown:
        v.unresolved("numa_unknown", d.id, "NUMA applicability must be explicit");
        break;
      case NumaBinding::NotApplicable:
        if (cpu.numaNode) {
          v.invalid("numa_binding", d.id, "Non-NUMA placement cannot name a NUMA node");
        }
        break;
      case NumaBinding::Node:
        if (!cpu.numaNode) {
          v.unresolved("numa_unknown", d.id, "NUMA node is unknown");
        }
        break;
      default:
        v.invalid("numa_binding", d.id, "Invalid NUMA binding");
    }
  }
  std::set<std::string> environments;
  for (const auto& environment : d.environments) {
    if (environment.backend.empty() || !environments.insert(environment.backend).second ||
        !backends.count(environment.backend)) {
      v.invalid("backend_environment", d.id,
                "Environment must name a unique declared backend");
    }
  }
  return v;
}

}  // namespace opfusion
