#include "opfusion/graph/Tensor.h"

#include "opfusion/common/Error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// 逻辑值与数值约束的纯校验；字节计算不分配内存、不读取参数制品。
namespace opfusion {

std::string toString(DataType type) {
  switch (type) {
  case DataType::Float16:
    return "float16";
  case DataType::Float32:
    return "float32";
  case DataType::Float64:
    return "float64";
  case DataType::Int8:
    return "int8";
  case DataType::Int32:
    return "int32";
  case DataType::Int64:
    return "int64";
  case DataType::Bool:
    return "bool";
  case DataType::Unknown:
    return "unknown";
  }
  throw Error("dtype_invalid", "invalid data type");
}
std::string toString(Layout layout) {
  switch (layout) {
  case Layout::RowMajor:
    return "row_major";
  case Layout::ColumnMajor:
    return "column_major";
  case Layout::NCHW:
    return "NCHW";
  case Layout::NHWC:
    return "NHWC";
  case Layout::OIHW:
    return "OIHW";
  case Layout::HWIO:
    return "HWIO";
  case Layout::Strided:
    return "strided";
  case Layout::Opaque:
    return "opaque";
  }
  throw Error("layout_invalid", "invalid layout enum value");
}
std::optional<unsigned> bits(DataType type) {
  switch (type) {
  case DataType::Float16:
    return 16;
  case DataType::Float32:
  case DataType::Int32:
    return 32;
  case DataType::Float64:
  case DataType::Int64:
    return 64;
  case DataType::Bool:
    return 8;
  case DataType::Int8:
    return 8;
  case DataType::Unknown:
    return std::nullopt;
  }
  throw Error("dtype_invalid", "invalid data type");
}
Bytes checkedMultiply(Bytes a, Bytes b) {
  require(b == 0 || a <= std::numeric_limits<Bytes>::max() / b, "size_overflow",
          "tensor size multiplication overflow");
  return a * b;
}
Bytes checkedAdd(Bytes a, Bytes b) {
  require(a <= std::numeric_limits<Bytes>::max() - b, "size_overflow",
          "tensor size addition overflow");
  return a + b;
}
Bytes elements(const Shape &shape) {
  for (auto dim : shape)
    require(dim >= 0, "shape_unresolved", "static dimensions cannot be negative");
  // 先拒绝所有负维度，再处理已知空张量；避免空乘积前的大维度伪溢出。
  if (std::find(shape.begin(), shape.end(), 0) != shape.end())
    return 0;
  Bytes result = 1;
  for (auto dim : shape)
    result = checkedMultiply(result, static_cast<Bytes>(dim));
  return result;
}
std::optional<Bytes> logicalBytes(const Shape &shape, DataType type) {
  const auto count = elements(shape);
  auto width = bits(type);
  if (!width)
    return std::nullopt;
  return checkedMultiply(count, *width / 8);
}
bool operator==(const TensorSignature &a, const TensorSignature &b) {
  return a.shape == b.shape && a.dtype == b.dtype && a.axes == b.axes;
}
bool operator!=(const TensorSignature &a, const TensorSignature &b) {
  return !(a == b);
}
void NumericalRequirements::validate() const {
  for (auto type : allowedTypes)
    require(bits(type).has_value(), "numeric_invalid", "allowed type cannot be unknown");
  if (computeType)
    require(bits(*computeType).has_value(), "numeric_invalid", "unknown compute type");
  if (accumulationType)
    require(bits(*accumulationType).has_value(), "numeric_invalid", "unknown accumulation type");
  for (const auto &error : {absoluteError, relativeError})
    if (error)
      require(std::isfinite(*error) && *error >= 0, "numeric_invalid", "invalid error bound");
}
std::string toString(ValueKind kind) {
  switch (kind) {
  case ValueKind::Unknown:
    return "unknown";
  case ValueKind::Input:
    return "input";
  case ValueKind::Parameter:
    return "parameter";
  case ValueKind::OperatorResult:
    return "operator_result";
  }
  throw Error("value_kind_invalid", "invalid logical value kind");
}
Tensor::Tensor(TensorId id, std::string name, Shape shape, DataType dtype,
               std::optional<LogicalAxes> axes, ValueKind kind,
               std::optional<ParameterRef> parameter, AxisOrigin axesOrigin)
    : mId(id), mName(std::move(name)), mSignature{std::move(shape), dtype, std::move(axes)},
      mKind(kind), mParameter(std::move(parameter)), mAxesOrigin(axesOrigin) {
  require(id != 0, "id_invalid", "tensor ID must be nonzero");
  (void)elements(mSignature.shape);
  (void)opfusion::logicalBytes(mSignature.shape, dtype);
  (void)toString(kind);
  validateAxes(mSignature.axes, mSignature.shape.size());
  switch (axesOrigin) {
  case AxisOrigin::Unknown:
  case AxisOrigin::Declared:
  case AxisOrigin::Derived:
  case AxisOrigin::Mixed:
    break;
  default:
    throw Error("axis_invalid", "invalid axis origin");
  }
  // 缺轴时不能保留 Declared 的假知识；有轴才要求可追溯的已知来源。
  if (!mSignature.axes)
    mAxesOrigin = AxisOrigin::Unknown;
  else
    require(axesOrigin != AxisOrigin::Unknown, "axis_invalid", "known axes need an origin");
  if (mParameter)
    require(kind == ValueKind::Parameter && !mParameter->artifactId.empty(), "parameter_invalid",
            "parameter references require the parameter role and artifact identity");
}
std::optional<Bytes> Tensor::logicalBytes() const {
  return opfusion::logicalBytes(mSignature.shape, mSignature.dtype);
}

} // namespace opfusion
