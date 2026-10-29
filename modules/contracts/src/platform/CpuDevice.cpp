// CPU 最低规则只约束声明用量；物理容量仍来自 Platform，实际请求另行核验。
#include "opfusion/platform/CpuDevice.h"

#include "opfusion/common/Error.h"
#include "opfusion/platform/Platform.h"

#include <utility>

namespace opfusion {

CpuDevice::CpuDevice(DeviceDescription description, DeviceResourceRules rules)
    : Device(std::move(description)), mRules(std::move(rules)) {
  require(this->description().kind == DeviceKind::Cpu, "cpu_device_kind",
          "CpuDevice requires a CPU description");
}

ResourceRequirements CpuDevice::requiredResources(const Platform& platform,
                                                 DeviceWork work) const {
  auto result = mRules.requirements(work, description().id);
  // core 配额不能转换成 slots；只有规范库存明确使用 cores 时才能比较。
  if (result.minimumUses && description().cpuAllocation &&
      description().cpuAllocation->quotaCores) {
    for (const auto& use : *result.minimumUses) {
      try {
        const auto& resource = platform.resource(use.resource);
        if (resource.kind == ResourceKind::Compute && resource.unit == "cores" &&
            use.amount > *description().cpuAllocation->quotaCores) {
          result.validation.invalid("cpu_resource_quota", description().id,
                                    "Required cores exceed the CPU allocation quota");
        }
      } catch (const Error& error) {
        result.validation.invalid(error.code(), description().id, error.what());
      }
    }
  }
  return result;
}

std::unique_ptr<const Device> CpuDevice::clone() const {
  return std::make_unique<CpuDevice>(*this);
}

}  // namespace opfusion
