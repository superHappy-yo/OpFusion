#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend::jsonGraph {

/// 返回 PointwiseAttributes/sigmoid_fwd 到 Sigmoid 的绑定值，不自行登记。
/// 工厂同步借用来源并严格校验单输入字段；非法属性抛出 Error，不提供执行实现。
SourceBinding makeSigmoidBinding();

}  // namespace opfusion::frontend::jsonGraph
