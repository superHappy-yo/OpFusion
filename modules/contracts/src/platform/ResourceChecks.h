#pragma once

// 平台描述与执行请求共用的 CPU 共享约束，仅供本功能目录内部核验。
#include "opfusion/platform/Resource.h"

namespace opfusion::detail {

/// 检查同节点已知 affinity 的交集；没有显式共同资源时不证明隔离。
/// actualUses 可空且仅在调用期间借用；非空时还检查该工作确实请求共享资源。
Validation validateCpuResourceSharing(const Platform& platform, const DeviceId& device,
                                      const std::vector<ResourceUse>* actualUses = nullptr);

}  // namespace opfusion::detail
