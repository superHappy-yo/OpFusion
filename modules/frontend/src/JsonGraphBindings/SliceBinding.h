#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 SliceAttributes 到 Slice 的绑定值，不自行登记或保存来源引用。
/// 起止与步长保留为逻辑元素索引；公共类拒绝非法范围，不裁剪或猜测缺失值。
SourceBinding makeSliceBinding();

}  // namespace opfusion::frontend::jsonGraph
