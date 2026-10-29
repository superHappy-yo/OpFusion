// 平台快照的所有权、身份与库存核验；不探测设备、不执行资源预留。
#include "opfusion/platform/Platform.h"

#include "ResourceChecks.h"
#include "opfusion/common/Error.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace opfusion {
namespace {

template <class T>
const T& lookup(const std::vector<T>& values, const std::string& id, const std::string& kind) {
  const auto found = std::find_if(values.begin(), values.end(),
                                  [&](const T& value) { return value.id == id; });
  require(found != values.end(), kind + "_missing", kind + " not found: " + id);
  return *found;
}

template <class T>
void identities(const std::vector<T>& values, const std::string& kind, Validation& v) {
  std::set<std::string> seen;
  for (const auto& value : values) {
    if (value.id.empty() || !seen.insert(value.id).second) {
      v.invalid("identity_invalid", value.id, "Empty or duplicate " + kind + " identity");
    }
  }
}

void throwInvalid(const Validation& validation) {
  for (const auto& issue : validation.issues) {
    if (issue.kind == IssueKind::Invalid) {
      throw Error(issue.code, issue.object + ": " + issue.message);
    }
  }
}

}  // namespace

Platform::Platform(std::string id, std::string registryId,
                   std::vector<std::unique_ptr<const Device>> devices,
                   std::vector<MemoryPool> pools, std::vector<Resource> resources,
                   std::vector<Link> links)
    : mId(std::move(id)), mRegistryId(std::move(registryId)), mDevices(std::move(devices)),
      mPools(std::move(pools)), mResources(std::move(resources)), mLinks(std::move(links)) {
  throwInvalid(validate());
}

Platform::Platform(const Platform& other)
    : mId(other.mId), mRegistryId(other.mRegistryId), mPools(other.mPools),
      mResources(other.mResources), mLinks(other.mLinks) {
  mDevices.reserve(other.mDevices.size());
  for (const auto& device : other.mDevices) {
    auto copy = device->clone();
    require(copy != nullptr, "device_clone_null", "Device clone must not be null");
    require(copy->description().id == device->description().id, "device_clone_identity",
            "Device clone must retain its identity");
    mDevices.push_back(std::move(copy));
  }
  // clone 必须产生同一快照的独立规则；异常时已复制设备由 RAII 销毁。
  throwInvalid(validate());
}

const Device& Platform::device(const DeviceId& id) const {
  const auto found = std::find_if(mDevices.begin(), mDevices.end(), [&](const auto& device) {
    return device && device->description().id == id;
  });
  require(found != mDevices.end(), "device_missing", "device not found: " + id);
  return **found;
}

const MemoryPool& Platform::pool(const PoolId& id) const { return lookup(mPools, id, "pool"); }
const Resource& Platform::resource(const ResourceId& id) const {
  return lookup(mResources, id, "resource");
}
const Link& Platform::link(const LinkId& id) const { return lookup(mLinks, id, "link"); }

bool Platform::accessible(const PoolId& poolId, const DeviceId& deviceId) const {
  device(deviceId);
  const auto& devices = pool(poolId).accessibleDevices;
  return std::find(devices.begin(), devices.end(), deviceId) != devices.end();
}

Validation Platform::validateDevice(const DeviceId& id) const {
  Validation v;
  try {
    v.append(validateDeviceDescription(device(id).description()));
    v.append(detail::validateCpuResourceSharing(*this, id));
  } catch (const Error& error) {
    v.invalid(error.code(), id, error.what());
  }
  return v;
}

Validation Platform::validate() const {
  Validation v;
  if (mId.empty()) {
    v.invalid("platform_identity", mId, "Platform identity is required");
  }
  if (mRegistryId.empty()) {
    v.unresolved("registry_unknown", mId, "Explicit registry identity is required");
  }
  std::set<DeviceId> deviceIds;
  for (const auto& device : mDevices) {
    if (!device) {
      v.invalid("device_null", mId, "Platform must not contain a null device");
    } else if (device->description().id.empty() ||
               !deviceIds.insert(device->description().id).second) {
      v.invalid("identity_invalid", device->description().id, "Empty or duplicate device identity");
    }
  }
  identities(mPools, "pool", v);
  identities(mResources, "resource", v);
  identities(mLinks, "link", v);
  if (mDevices.empty()) {
    v.unresolved("devices_unknown", mId, "No devices described");
  }
  for (const auto& device : mDevices) {
    if (device) {
      v.append(validateDevice(device->description().id));
    }
  }
  std::set<std::string> physical;
  for (const auto& pool : mPools) {
    if (pool.physicalIdentity.empty()) {
      v.unresolved("physical_pool_unknown", pool.id, "Physical identity must be explicit");
    } else if (!physical.insert(pool.physicalIdentity).second) {
      v.invalid("physical_pool_duplicate", pool.id, "One physical pool must have one ID");
    }
    if (!pool.physicalCapacityBytes) {
      v.unresolved("physical_capacity_unknown", pool.id, "Unknown capacity is not zero");
    }
    std::set<DeviceId> seen;
    for (const auto& deviceId : pool.accessibleDevices) {
      if (!seen.insert(deviceId).second) {
        v.invalid("pool_access_duplicate", pool.id, deviceId);
      }
      try {
        device(deviceId);
      } catch (const Error& error) {
        v.invalid(error.code(), pool.id, error.what());
      }
    }
  }
  for (const auto& resource : mResources) {
    v.append(validateResourceDescription(*this, resource));
  }
  for (const auto& link : mLinks) {
    try {
      pool(link.sourcePool);
      pool(link.targetPool);
    } catch (const Error& error) {
      v.invalid(error.code(), link.id, error.what());
    }
    v.append(validateResourceUses(*this, link.resources, link.id));
    if (link.resources.empty()) {
      v.unresolved("link_resources_unknown", link.id, "A link needs explicit shared resources");
    }
    if (link.bandwidthBytesPerUs &&
        (!std::isfinite(*link.bandwidthBytesPerUs) || *link.bandwidthBytesPerUs <= 0)) {
      v.invalid("link_bandwidth", link.id, "Known bandwidth must be finite and positive");
    }
    if (link.startupUs && (!std::isfinite(*link.startupUs) || *link.startupUs < 0)) {
      v.invalid("link_startup", link.id, "Startup must be finite and nonnegative");
    }
  }
  return v;
}

}  // namespace opfusion
