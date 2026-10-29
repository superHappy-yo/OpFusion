#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回扁平 ResampleFwdAttributes/maxpool 到 MaxPool 的绑定值，不自行登记。
/// 工厂同步借用来源，要求负无穷填充；窗口、端口及索引输出限制由公共类校验。
SourceBinding makeMaxPoolBinding();

} // namespace opfusion::frontend::jsonGraph
