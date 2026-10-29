#pragma once

// 平台资源的唯一库存与请求协议；设备规则只引用这些身份，不复制容量。
#include "opfusion/common/Validation.h"
#include "opfusion/graph/Tensor.h"

#include <optional>
#include <string>
#include <vector>

namespace opfusion {

class Platform;

using DeviceId = std::string;
using PoolId = std::string;
using ResourceId = std::string;
using LinkId = std::string;

/// 当前资源规则区分的工作类别；不是设备型号或实现提供者的中央名单。
enum class DeviceWork { Compute, LocalConversion };

/// 一次工作的实际请求或规则下限；amount 使用被引用 Resource 的单位。
struct ResourceUse {
  ResourceId resource;
  double amount = 1.0;
};

/// 资源用途与作用域相互独立；寄存器/线程额度不能代替任务级 Compute。
enum class ResourceKind { Compute, Copy, Link, Threads, Registers, OnChip, Other };
/// Thread/Block 是内部局部限制，不进入 Plan 的整任务并发活动求和。
enum class ResourceScope { Device, Pool, Link, Thread, Block };

/// 规范物理资源；Platform 独占保存库存，所有消费者按同一个 ID 计费。
struct Resource {
  ResourceId id;
  ResourceKind kind = ResourceKind::Other;
  ResourceScope scope = ResourceScope::Device;
  std::string unit;
  /// nullopt 是未知上限，已知零容量仍是有效的不可用资源描述。
  std::optional<double> capacity;
  std::optional<DeviceId> device;
  std::optional<PoolId> pool;
  /// 与 device 独占归属互斥；必须同时给出明确的共享依据。
  std::vector<DeviceId> sharedDevices;
  std::string sharingRef;
};

/// 唯一物理内存池；逻辑张量、候选表示和实际 Allocation 不在这里合并。
struct MemoryPool {
  PoolId id;
  std::string physicalIdentity;
  std::optional<Bytes> physicalCapacityBytes;
  std::vector<DeviceId> accessibleDevices;
};

/// 有向池间链路；两个方向可引用同一个共享资源，而非重复声明容量。
struct Link {
  LinkId id;
  PoolId sourcePool;
  PoolId targetPool;
  std::vector<ResourceUse> resources;
  std::optional<double> bandwidthBytesPerUs;
  std::optional<double> startupUs;
};

/// 核对库存自身的单位、容量、归属和引用；只借用平台，不探测硬件。
Validation validateResourceDescription(const Platform& platform, const Resource& resource);
/// 校验正数请求、唯一 ID、物理上限和可选设备归属；不创建资源活动。
Validation validateResourceUses(const Platform& platform,
                                const std::vector<ResourceUse>& uses,
                                const std::string& object,
                                const std::optional<DeviceId>& device = std::nullopt);
/// 追加设备工作规则及任务级 Compute 覆盖校验；未知规则保持 Unresolved。
/// 不修改实际请求、不按最低量自动预留；动态 Boundary 限额由 Plan 检查。
Validation validateExecutionResourceUses(const Platform& platform, const DeviceId& device,
                                         DeviceWork work,
                                         const std::vector<ResourceUse>& uses,
                                         const std::string& object);

}  // namespace opfusion
