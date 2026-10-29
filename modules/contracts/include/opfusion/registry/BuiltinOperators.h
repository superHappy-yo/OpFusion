#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion {

/// 向调用方拥有的可变 Registry 登记构建发现的内置语义。
/// 重复身份或冻结后登记由 Registry 抛出 Error；不登记来源绑定。
void registerBuiltinOperators(Registry& registry);
/// 返回覆盖领域、来源适配和构建条件的内容摘要；指针具有静态存储期。
const char* builtinOperatorsDigest() noexcept;

}  // namespace opfusion
