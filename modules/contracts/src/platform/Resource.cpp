// 校验唯一资源库存、实际请求与设备下限；所有验证均不修改或预留资源。
#include "opfusion/platform/Resource.h"

#include "ResourceChecks.h"
#include "opfusion/common/Error.h"
#include "opfusion/platform/Device.h"
#include "opfusion/platform/Platform.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace opfusion {
namespace {

bool includesDevice(const Resource& resource, const DeviceId& device) {
  return std::find(resource.sharedDevices.begin(), resource.sharedDevices.end(), device) !=
         resource.sharedDevices.end();
}

bool taskCompute(const Resource& resource) {
  return resource.kind == ResourceKind::Compute && resource.scope == ResourceScope::Device;
}

}  // namespace

Validation validateResourceDescription(const Platform& p, const Resource& r) {
  Validation v;
  switch (r.kind) {
    case ResourceKind::Compute:
    case ResourceKind::Copy:
    case ResourceKind::Link:
    case ResourceKind::Threads:
    case ResourceKind::Registers:
    case ResourceKind::OnChip:
    case ResourceKind::Other:
      break;
    default:
      v.invalid("resource_kind", r.id, "Invalid resource kind");
  }
  switch (r.scope) {
    case ResourceScope::Device:
    case ResourceScope::Pool:
    case ResourceScope::Link:
    case ResourceScope::Thread:
    case ResourceScope::Block:
      break;
    default:
      v.invalid("resource_scope", r.id, "Invalid resource scope");
  }
  if (r.unit.empty()) {
    v.unresolved("resource_unit_unknown", r.id, "Resource units must be explicit");
  }
  if (!r.capacity) {
    v.unresolved("resource_capacity_unknown", r.id, "Unknown capacity is not zero");
  } else if (!std::isfinite(*r.capacity) || *r.capacity < 0) {
    v.invalid("resource_capacity", r.id, "Capacity must be finite and nonnegative");
  }
  try {
    if (r.device) {
      p.device(*r.device);
    }
    if (r.pool) {
      p.pool(*r.pool);
    }
    if (r.device && !r.sharedDevices.empty()) {
      v.invalid("resource_ownership", r.id, "Exclusive and shared owners are mutually exclusive");
    }
    std::set<DeviceId> seen;
    for (const auto& device : r.sharedDevices) {
      p.device(device);
      if (!seen.insert(device).second) {
        v.invalid("resource_ownership", r.id, "Duplicate shared owner");
      }
    }
    if (!r.sharedDevices.empty() && r.sharingRef.empty()) {
      v.unresolved("resource_sharing_unknown", r.id,
                   "Shared device access requires an explicit sharing reference");
    }
    if (r.scope == ResourceScope::Pool && !r.pool) {
      v.unresolved("resource_pool_unknown", r.id, "Pool-scoped resource needs a pool");
    }
  } catch (const Error& error) {
    v.invalid(error.code(), r.id, error.what());
  }
  return v;
}

Validation validateResourceUses(const Platform& p, const std::vector<ResourceUse>& uses,
                                const std::string& object,
                                const std::optional<DeviceId>& device) {
  Validation v;
  std::set<ResourceId> seen;
  for (const auto& use : uses) {
    if (!seen.insert(use.resource).second) {
      v.invalid("resource_duplicate", object, "Repeated resource request");
    }
    if (!std::isfinite(use.amount) || use.amount <= 0) {
      v.invalid("resource_amount", object, "Amount must be finite and positive");
    }
    try {
      const auto& resource = p.resource(use.resource);
      v.append(validateResourceDescription(p, resource));
      if (resource.capacity && use.amount > *resource.capacity) {
        v.invalid("resource_capacity", object, "Request exceeds physical limit: " + use.resource);
      }
      if (device) {
        p.device(*device);
        if (resource.device && *resource.device != *device) {
          v.invalid("resource_device", object, "Resource belongs to another device");
        }
        if (!resource.sharedDevices.empty() && !includesDevice(resource, *device)) {
          v.invalid("resource_device", object, "Device is not an explicitly shared owner");
        }
        if (resource.pool && !p.accessible(*resource.pool, *device)) {
          v.invalid("resource_pool_access", object, "Device cannot access the resource pool");
        }
        if (!resource.device && resource.sharedDevices.empty() && !resource.pool) {
          v.unresolved("resource_owner_unknown", object,
                       "Device-scoped use lacks exclusive, pool or proven shared ownership");
        }
      }
    } catch (const Error& error) {
      v.invalid(error.code(), object, error.what());
    }
  }
  return v;
}

namespace detail {

Validation validateCpuResourceSharing(const Platform& p, const DeviceId& id,
                                      const std::vector<ResourceUse>* actualUses) {
  Validation v;
  const auto& d = p.device(id).description();
  if (d.kind != DeviceKind::Cpu || d.nodeId.empty() || !d.cpuAllocation ||
      !d.cpuAllocation->affinity) {
    return v;
  }
  const auto& affinity = *d.cpuAllocation->affinity;
  for (const auto& entry : p.devices()) {
    if (!entry || entry->description().id == id) {
      continue;
    }
    const auto& other = entry->description();
    if (other.kind != DeviceKind::Cpu || other.nodeId != d.nodeId || !other.cpuAllocation ||
        !other.cpuAllocation->affinity) {
      continue;
    }
    const auto& assigned = *other.cpuAllocation->affinity;
    const bool overlaps = std::any_of(affinity.begin(), affinity.end(), [&](std::uint32_t core) {
      return std::find(assigned.begin(), assigned.end(), core) != assigned.end();
    });
    if (!overlaps) {
      continue;
    }
    bool declared = false;
    bool requested = false;
    for (const auto& resource : p.resources()) {
      if (!taskCompute(resource) || resource.sharingRef.empty() ||
          !includesDevice(resource, id) || !includesDevice(resource, other.id)) {
        continue;
      }
      declared = true;
      if (actualUses && std::any_of(actualUses->begin(), actualUses->end(),
          [&](const ResourceUse& use) { return use.resource == resource.id && use.amount > 0; })) {
        requested = true;
      }
    }
    // 实例配额不能证明交集核心相互独占；必须进入共同资源的同一容量账本。
    if (!declared) {
      v.unresolved("cpu_sharing_unknown", id,
                   "Overlapping CPU affinity lacks an explicit shared compute constraint: " +
                       other.id);
    } else if (actualUses && !requested) {
      v.invalid("cpu_shared_resource_missing", id,
                "Execution omits the shared compute constraint for overlapping CPU affinity");
    }
  }
  return v;
}

}  // namespace detail

Validation validateExecutionResourceUses(const Platform& p, const DeviceId& device,
                                         DeviceWork work, const std::vector<ResourceUse>& uses,
                                         const std::string& object) {
  Validation v;
  v.append(validateResourceUses(p, uses, object, device));
  switch (work) {
    case DeviceWork::Compute:
    case DeviceWork::LocalConversion:
      break;
    default:
      v.invalid("device_work", object, "Invalid device work category");
      return v;
  }
  try {
    const auto& d = p.device(device);
    v.append(p.validateDevice(device));
    const auto required = d.requiredResources(p, work);
    v.append(required.validation);
    if (!required.minimumUses) {
      v.unresolved("device_resource_rules_unknown", object,
                   "Minimum execution resource rules are not established");
    } else {
      v.append(validateResourceUses(p, *required.minimumUses, object, device));
      bool computeRequired = false;
      for (const auto& minimum : *required.minimumUses) {
        const auto& resource = p.resource(minimum.resource);
        computeRequired = taskCompute(resource) || computeRequired;
        const auto found = std::find_if(uses.begin(), uses.end(), [&](const ResourceUse& use) {
          return use.resource == minimum.resource && use.amount >= minimum.amount;
        });
        if (found == uses.end()) {
          v.invalid("execution_resource_missing", object,
                    "Execution omits a required resource or its minimum amount: " + minimum.resource);
        }
      }
      if (!computeRequired) {
        v.invalid("device_compute_requirement", object,
                  "A known compute/conversion rule must require device-scoped Compute");
      }
    }
    bool computeRequested = false;
    for (const auto& use : uses) {
      const auto& resource = p.resource(use.resource);
      computeRequested = taskCompute(resource) || computeRequested;
      const auto& description = d.description();
      if (description.kind == DeviceKind::Cpu && description.cpuAllocation &&
          description.cpuAllocation->quotaCores && taskCompute(resource) &&
          resource.unit == "cores" && use.amount > *description.cpuAllocation->quotaCores) {
        v.invalid("cpu_resource_quota", object, "Requested cores exceed the CPU allocation quota");
      }
    }
    // 局部 Registers/Thread/Block 即使非空，也不会进入整任务 Compute 活动。
    if (!computeRequested) {
      if (uses.empty() && !required.minimumUses) {
        v.unresolved("compute_resources_unknown", object, "Compute reservations are not established");
      } else {
        v.invalid("execution_compute_resource", object,
                  "Compute/conversion work needs an explicit device-scoped Compute request");
      }
    }
    const auto& description = d.description();
    if (description.kind == DeviceKind::Cpu && description.cpuAllocation) {
      const auto& cpu = *description.cpuAllocation;
      if ((cpu.quotaCores && *cpu.quotaCores == 0) || (cpu.affinity && cpu.affinity->empty())) {
        v.invalid("cpu_unavailable", object, "The selected CPU has no assigned execution capacity");
      }
    }
    v.append(detail::validateCpuResourceSharing(p, device, &uses));
  } catch (const Error& error) {
    v.invalid(error.code(), object, error.what());
  }
  return v;
}

}  // namespace opfusion
