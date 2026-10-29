#pragma once

// Plan 内部的数值组合与计时条件核验；不执行验证样例，也不测量时间。
#include "opfusion/plan/Plan.h"

#include <functional>

namespace opfusion::detail {

/// 对需要近似验收的精确候选/交接/边界阶段组合查找有效整图验收声明。
/// 仅同步借用 plan；缺少覆盖该组合的验收时返回 Unresolved，不复用其他组合。
Validation validatePlanNumerics(const Plan& plan);

/// 根据事件偏序核验计时工作覆盖，并核对 InputsAvailable/OutputsDelivered 的准确时刻。
/// ordered 仅在调用期间借用；时间相等仍不能替代依赖，不核验真实执行时钟。
Validation validatePlanTiming(const Plan& plan,
                              const std::function<bool(const EventId&, const EventId&)>& ordered);

}  // namespace opfusion::detail
