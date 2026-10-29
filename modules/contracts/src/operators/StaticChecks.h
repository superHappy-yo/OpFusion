#pragma once
#include "SemanticChecks.h"

#include <algorithm>
#include <limits>
namespace opfusion::operators::detail {
// 来源中的负轴在具体算子内按已知 rank 规范化。
inline std::size_t axisIndex(std::int64_t axis, std::size_t rank) {
  require(rank <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
          "shape_overflow", "rank overflow");
  if (axis < 0)
    axis += static_cast<std::int64_t>(rank);
  require(axis >= 0 && static_cast<std::size_t>(axis) < rank, "invalid_axis",
          "axis is outside rank");
  return static_cast<std::size_t>(axis);
}
inline void variablePorts(const Operator &op, std::size_t low, std::size_t high,
                          std::size_t out = 1) {
  require(op.inputs().size() >= low && op.inputs().size() <= high && op.outputs().size() == out,
          "invalid_ports", "operator port count mismatch");
}
inline std::int64_t shapeProduct(const Shape &s) {
  const auto n = elements(s);
  require(n <= static_cast<Bytes>(std::numeric_limits<std::int64_t>::max()), "shape_overflow",
          "shape product overflow");
  return static_cast<std::int64_t>(n);
}
// 仅对有限静态整数求值；溢出和除零明确失败。
inline std::int64_t integerValue(std::int64_t a, std::int64_t b, char op) {
  std::int64_t result = 0;
  bool overflow = false;
  if (op == '+')
    overflow = __builtin_add_overflow(a, b, &result);
  if (op == '-')
    overflow = __builtin_sub_overflow(a, b, &result);
  if (op == '*')
    overflow = __builtin_mul_overflow(a, b, &result);
  if (op == '/') {
    require(b != 0 && !(a == std::numeric_limits<std::int64_t>::min() && b == -1),
            "invalid_integer", "integer division invalid");
    result = a / b;
  }
  require(!overflow, "shape_overflow", "static integer overflow");
  return result;
}
} // namespace opfusion::operators::detail
