#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 ConcatenateAttributes 到 Concat 的绑定值，不自行登记或保存来源引用。
/// 工厂限 2..4 个有序输入、逻辑 axis=1，拒绝 in_place_index 等未支持设置。
SourceBinding makeConcatBinding();

}  // namespace opfusion::frontend::jsonGraph
