#pragma once

// 显式声明式设备，不为 GPU/加速器描述伪造编译或执行能力。
#include "opfusion/platform/Device.h"

namespace opfusion {

/// 使用已有资源规则描述非 CPU 设备；新型号可复用本类型而无需增加名称分支。
/// 新资源行为可直接继承 Device，不要求把硬件方法塞入该声明类型。
class DeclaredDevice final : public Device {
 public:
  /// 按值拥有描述和规则；CPU 使用 CpuDevice，未知类别仍由校验保守处理。
  explicit DeclaredDevice(DeviceDescription description, DeviceResourceRules rules = {});
  /// 返回显式声明的工作下限；没有规则时不默认支持或填零。
  ResourceRequirements requiredResources(const Platform& platform,
                                         DeviceWork work) const override;
  /// 复制声明和规则值，保留同一设备身份而不借用原对象。
  std::unique_ptr<const Device> clone() const override;

 private:
  DeviceResourceRules mRules;
};

}  // namespace opfusion
