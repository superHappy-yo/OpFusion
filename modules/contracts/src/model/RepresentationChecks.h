#pragma once

// Model 内部布局核验入口；不是另一个公开表示或转换器。
#include "opfusion/model/Model.h"

namespace opfusion::detail {
/// 借用候选表示和逻辑轴，核对存储排列与字节 stride；未知轴保留 Unresolved。
Validation validateStorageLayout(const Representation& representation, const Tensor& tensor);
}  // namespace opfusion::detail
