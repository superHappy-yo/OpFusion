#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 PointwiseAttributes/add 到 Add 的绑定值，不自行登记或保存来源引用。
/// 调用方先登记目标语义再 bind；工厂拒绝非空的未支持属性及非法端口。
SourceBinding makeAddBinding();

}  // namespace opfusion::frontend::jsonGraph
