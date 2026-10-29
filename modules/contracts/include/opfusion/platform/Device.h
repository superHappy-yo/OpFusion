#pragma once

// 设备的不可变描述与薄资源规则端口；不包含 SDK、执行句柄或自动注册。
#include "opfusion/platform/Resource.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

class Platform;

/// 封闭的粗粒度类别；新设备型号仍用开放的架构/实例身份表达。
enum class DeviceKind { Unknown, Cpu, Gpu, Accelerator, Other };
/// NUMA 未知、不适用与已绑定节点不能混为同一种缺省状态。
enum class NumaBinding { Unknown, NotApplicable, Node };

/// 实例的 CPU 分配条件；不是一份额外的物理核心容量账本。
struct CpuAllocation {
  /// 单位为 core，允许分数；不默认一个资源 slot 等于一个 core。
  std::optional<double> quotaCores;
  std::optional<std::vector<std::uint32_t>> affinity;
  NumaBinding numaBinding = NumaBinding::Unknown;
  std::optional<std::uint32_t> numaNode;
};

/// 明确声明的后端运行环境；存在描述不代表该环境已探测或初始化。
struct BackendEnvironment {
  std::string backend;
  std::string version;
  std::string abi;
  std::string runtimeVersion;
  std::optional<bool> requiresDriver;
  std::optional<std::string> driverVersion;
};

/// 设备实例的纯值描述；传入 Device 后独立保存，不保留调用方可变引用。
struct DeviceDescription {
  DeviceId id;
  std::string nodeId;
  DeviceKind kind = DeviceKind::Unknown;
  std::string architecture;
  std::string instance;
  std::string configuration;
  std::vector<std::string> backends;
  std::optional<CpuAllocation> cpuAllocation;
  std::vector<BackendEnvironment> environments;
};

/// 一类工作的最低请求与诊断；nullopt 表示规则未知，不是零资源工作。
struct ResourceRequirements {
  std::optional<std::vector<ResourceUse>> minimumUses;
  Validation validation;
};

/// 两类现有工作的显式规则；只拥有资源 ID/最低量，不复制物理或动态容量。
struct DeviceResourceRules {
  std::optional<std::vector<ResourceUse>> compute;
  std::optional<std::vector<ResourceUse>> localConversion;
  /// 查询一类规则；未知返回 Unresolved，非法 work 返回 Invalid。
  ResourceRequirements requirements(DeviceWork work, const DeviceId& device) const;
};

/// 拥有不可变描述的资源协议；新规则通过直接继承替换，不负责设备 I/O。
/// 派生类必须按值保存不可变规则，不能捕获外部可变配置或运行句柄。
/// const 不自动提供线程安全；并发查询还依赖派生实现遵守纯查询约定。
class Device {
 public:
  virtual ~Device() = default;
  Device& operator=(const Device&) = delete;
  /// 借用本实例描述；只在该设备对象存活期间有效。
  const DeviceDescription& description() const noexcept { return mDescription; }
  /// 只借用平台，返回当前工作最低资源要求；不得探测、回写或递归完整校验。
  virtual ResourceRequirements requiredResources(const Platform& platform,
                                                 DeviceWork work) const = 0;
  /// 复制完整动态类型及规则，不共享可变状态；返回非空独占所有权。
  virtual std::unique_ptr<const Device> clone() const = 0;

 protected:
  /// 取得描述值，不在基类构造期间调用派生虚函数。
  explicit Device(DeviceDescription description);
  Device(const Device&) = default;

 private:
  DeviceDescription mDescription;
};

/// 核对所有派生类型都应遵守的描述、CPU 分配和后端目录不变量。
/// 不判断某项工作是否支持，也不调用资源规则或初始化设备。
Validation validateDeviceDescription(const DeviceDescription& description);

}  // namespace opfusion
