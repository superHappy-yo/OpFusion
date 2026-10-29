#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 PointwiseAttributes/mul 到 Mul 的绑定值，不自行登记或保存来源引用。
/// 工厂保留两个有序输入，属性/端口失败抛出 Error，不据名字推断设备支持。
SourceBinding makeMulBinding();

}  // namespace opfusion::frontend::jsonGraph
