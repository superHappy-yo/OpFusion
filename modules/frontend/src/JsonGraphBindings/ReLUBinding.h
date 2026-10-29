#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 PointwiseAttributes/relu_fwd 到标准 ReLU 的绑定值，不自行登记。
/// 工厂同步借用来源，裁剪/斜率参数及额外输入只能缺失或为 null，不静默近似其他激活。
SourceBinding makeReLUBinding();

}  // namespace opfusion::frontend::jsonGraph
