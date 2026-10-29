#pragma once

// CPU 的纯资源规则适配；配额/亲和性是声明条件，不在构造中探测机器。
#include "opfusion/platform/Device.h"

namespace opfusion {

/// CPU 实例描述与最低资源规则；具体候选仍须独立申报完整实际用量。
class CpuDevice final : public Device {
 public:
  /// 拥有描述和规则；非 CPU 类别抛出 Error，未提供规则保持未知。
  explicit CpuDevice(DeviceDescription description, DeviceResourceRules rules = {});
  /// 查询工作下限并检查量纲明确的 core 请求与分配配额；不推定默认核心数。
  ResourceRequirements requiredResources(const Platform& platform,
                                         DeviceWork work) const override;
  /// 独立复制描述与规则，不复制硬件句柄（本类型不持有句柄）。
  std::unique_ptr<const Device> clone() const override;

 private:
  DeviceResourceRules mRules;
};

}  // namespace opfusion
