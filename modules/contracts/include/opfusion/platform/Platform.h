#pragma once

// 不可变平台快照：独占多态设备，按唯一身份保存资源、池和链路。
#include "opfusion/platform/Device.h"
#include "opfusion/platform/Resource.h"

#include <memory>
#include <string>
#include <vector>

namespace opfusion {

/// 给定设备与资源的纯领域快照；没有探测、预留、编译或执行副作用。
/// 所有查询引用借用自本实例；设备规则必须遵守 Device 的不可变纯查询约定。
class Platform {
 public:
  /// 接管全部设备与值记录；空设备、重复身份或其他 Invalid 条件抛出 Error。
  /// Unresolved 可以保留在快照中，但不能据此宣布某个候选已可执行。
  Platform(std::string id, std::string registryId,
           std::vector<std::unique_ptr<const Device>> devices,
           std::vector<MemoryPool> pools, std::vector<Resource> resources,
           std::vector<Link> links);
  /// 逐设备 clone，保留动态行为并独立复制规则；clone 失败不发布部分平台。
  Platform(const Platform& other);
  Platform& operator=(const Platform&) = delete;
  /// 平台及其依赖目录的版本身份；不会随查询自动更新。
  const std::string& id() const noexcept { return mId; }
  /// 候选、图与实现必须使用相同的显式目录身份。
  const std::string& registryId() const noexcept { return mRegistryId; }
  /// 借用只读多态容器；不能复制该容器来绕过设备的 clone 协议。
  const std::vector<std::unique_ptr<const Device>>& devices() const noexcept { return mDevices; }
  /// 物理池库存，非运行时分配表。
  const std::vector<MemoryPool>& pools() const noexcept { return mPools; }
  /// 唯一物理资源库存；共享使用者引用同一条目，不复制容量。
  const std::vector<Resource>& resources() const noexcept { return mResources; }
  /// 有向链路目录，不代表传输已经发生。
  const std::vector<Link>& links() const noexcept { return mLinks; }
  /// 按 ID 借用设备；未知身份抛出 device_missing。
  const Device& device(const DeviceId& id) const;
  /// 按 ID 借用物理池；未知身份抛出 pool_missing。
  const MemoryPool& pool(const PoolId& id) const;
  /// 按 ID 借用规范资源；未知身份抛出 resource_missing。
  const Resource& resource(const ResourceId& id) const;
  /// 按 ID 借用有向链路；未知身份抛出 link_missing。
  const Link& link(const LinkId& id) const;
  /// 查询显式池访问权限；未知设备或池抛出 Error，不隐式建立共享。
  bool accessible(const PoolId& pool, const DeviceId& device) const;
  /// 核对库存、身份与描述；不把未使用工作类别的规则缺失当成失败。
  Validation validate() const;
  /// 核对指定设备的公共描述；工作资源规则在具体工作入口按需查询。
  Validation validateDevice(const DeviceId& device) const;

 private:
  std::string mId, mRegistryId;
  std::vector<std::unique_ptr<const Device>> mDevices;
  std::vector<MemoryPool> mPools;
  std::vector<Resource> mResources;
  std::vector<Link> mLinks;
};

}  // namespace opfusion
