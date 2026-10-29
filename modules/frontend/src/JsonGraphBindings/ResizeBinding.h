#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 ResizeAttributes 到 Resize 的绑定值，不自行登记或保存来源引用。
/// 工厂限 nearest/asymmetric/floor/stretch 及空 ROI；其余属性由公共类校验，失败抛出 Error。
SourceBinding makeResizeBinding();

}  // namespace opfusion::frontend::jsonGraph
