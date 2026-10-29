#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 ConvolutionFwdAttributes 到无偏置 Conv 的绑定值，不自行登记。
/// 工厂同步借用来源，限互相关、dilation=1、stride=1/2、padding=0/1；非法设置抛出 Error。
SourceBinding makeConvBinding();

} // namespace opfusion::frontend::jsonGraph
