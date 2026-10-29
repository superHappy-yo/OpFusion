// 显式资源声明的值语义实现；查询规则不会探测或执行被描述的设备。
#include "opfusion/platform/DeclaredDevice.h"

#include "opfusion/common/Error.h"

#include <utility>

namespace opfusion {

DeclaredDevice::DeclaredDevice(DeviceDescription description, DeviceResourceRules rules)
    : Device(std::move(description)), mRules(std::move(rules)) {
  require(this->description().kind != DeviceKind::Cpu, "declared_cpu_device",
          "Use CpuDevice for a CPU description");
}

ResourceRequirements DeclaredDevice::requiredResources(const Platform&, DeviceWork work) const {
  return mRules.requirements(work, description().id);
}

std::unique_ptr<const Device> DeclaredDevice::clone() const {
  return std::make_unique<DeclaredDevice>(*this);
}

}  // namespace opfusion
