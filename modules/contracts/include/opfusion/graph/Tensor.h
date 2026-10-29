#pragma once

#include "opfusion/graph/Axis.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace opfusion {

// 既有数值 ID 仍是别名；本接口不承诺不同身份间已有编译期隔离。
using TensorId = std::uint64_t;
using OperatorId = std::uint64_t;
using EdgeId = std::uint64_t;
using Bytes = std::uint64_t;
/// 静态逻辑维度；负值非法，零维度为空张量，空 shape 为标量。
using Shape = std::vector<std::int64_t>;

/// 逻辑元素类型；Unknown 不是零位宽，也不代表默认浮点类型。
enum class DataType { Unknown, Float16, Float32, Float64, Int8, Int32, Int64, Bool };
/// 返回稳定类型文本；非法枚举抛出 dtype_invalid。
std::string toString(DataType type);

/// 候选的物理排列，独立于逻辑 AxisOrder；不附加到原始 Tensor。
enum class Layout { RowMajor, ColumnMajor, NCHW, NHWC, OIHW, HWIO, Strided, Opaque };
/// 返回稳定排列文本；非法枚举抛出 layout_invalid。
std::string toString(Layout layout);
/// 返回每元素位数；Unknown 返回 nullopt，非法枚举抛出 dtype_invalid。
std::optional<unsigned> bits(DataType type);
/// 进行有溢出检查的字节/数量乘法；溢出抛出 size_overflow。
Bytes checkedMultiply(Bytes a, Bytes b);
/// 进行有溢出检查的字节/数量加法；溢出抛出 size_overflow。
Bytes checkedAdd(Bytes a, Bytes b);
/// 借用 shape 计算元素数；先拒绝负维度，含零时返回零，空 shape 返回一。
/// 非空乘积无法表示时抛出 size_overflow，不把未知维度当作零。
Bytes elements(const Shape &shape);
/// 返回稠密逻辑字节数，而非物理占用或访问流量；未知 dtype 返回 nullopt。
std::optional<Bytes> logicalBytes(const Shape &shape, DataType type);

/// 可独立复制的逻辑签名；允许携带未解析 dtype/轴，使用方负责相应校验。
struct TensorSignature {
  Shape shape;
  DataType dtype = DataType::Unknown;
  std::optional<LogicalAxes> axes;
  /// 创建默认未知类型的标量签名，不代表已验证的算子输入。
  TensorSignature() = default;
  /// 按值保存 shape/dtype/轴，不借用调用方容器，也不在此执行语义推导。
  TensorSignature(Shape shape, DataType dtype, std::optional<LogicalAxes> axes = std::nullopt)
      : shape(std::move(shape)), dtype(dtype), axes(std::move(axes)) {}
};
/// 精确比较 shape、dtype 和轴知识；兼容但不相同的轴仍不相等。
bool operator==(const TensorSignature &a, const TensorSignature &b);
/// 精确签名比较的否定，不执行广播或类型转换。
bool operator!=(const TensorSignature &a, const TensorSignature &b);

/// 逻辑值来源；Unknown 保留导入信息不足，不按名字推断参数。
enum class ValueKind { Unknown, Input, Parameter, OperatorResult };
/// 返回来源文本；非法枚举抛出 value_kind_invalid。
std::string toString(ValueKind kind);
/// 参数内容的外部制品引用，不拥有权重字节，也不在构造时读取制品。
struct ParameterRef {
  std::string artifactId;
  std::string member;
  std::string contentDigest;
};

/// 算子实例的数值约束声明，不是实际执行误差或设备能力证据。
struct NumericalRequirements {
  /// 空集合表示未额外限制存储 dtype；算子仍可限定自身支持类型。
  std::vector<DataType> allowedTypes;
  /// 缺省表示未声明，不从输入 dtype 自动补全计算或累加精度。
  std::optional<DataType> computeType;
  std::optional<DataType> accumulationType;
  bool allowApproximation = false;
  /// 可选非负有限误差界；缺省不等于零误差已被验证。
  std::optional<double> absoluteError;
  std::optional<double> relativeError;
  /// 拒绝未知/非法数值类型及非法误差界，不执行数值验证。
  void validate() const;
};

/// 逻辑值的只读领域记录，拥有签名、来源和参数引用，不保存设备或 Buffer。
/// 已知轴必须有来源；角色为 Parameter 才能携带非空 artifactId 的参数引用。
class Tensor {
public:
  /// 按值接收并校验非零 ID、静态 shape、dtype 枚举、轴及参数角色。
  /// Unknown dtype/角色可保留；缺省轴的来源规范化为 Unknown。
  Tensor(TensorId id, std::string name, Shape shape, DataType dtype,
         std::optional<LogicalAxes> axes = std::nullopt, ValueKind kind = ValueKind::Unknown,
         std::optional<ParameterRef> parameter = std::nullopt,
         AxisOrigin axesOrigin = AxisOrigin::Declared);
  /// 返回逻辑值身份，不是缓冲或分配身份。
  TensorId id() const noexcept { return mId; }
  /// 借用名称；引用只在本对象未被替换且仍存活时有效。
  const std::string &name() const noexcept { return mName; }
  /// 借用逻辑维度，不描述物理 stride 或 padding。
  const Shape &shape() const noexcept { return mSignature.shape; }
  /// 返回存储元素类型；Unknown 仍须由消费方保守处理。
  DataType dtype() const noexcept { return mSignature.dtype; }
  /// 借用本对象拥有的签名，生命周期同 name() 的借用约定。
  const TensorSignature &signature() const noexcept { return mSignature; }
  /// 返回来源角色，不将外部输入和固定参数合并。
  ValueKind kind() const noexcept { return mKind; }
  /// 返回已知轴知识的来源，不证明轴已获数值验证。
  AxisOrigin axesOrigin() const noexcept { return mAxesOrigin; }
  /// 借用参数引用；nullopt 不会触发制品读取或默认权重构造。
  const std::optional<ParameterRef> &parameter() const noexcept { return mParameter; }
  /// 返回稠密逻辑字节数；未知 dtype 返回 nullopt，不等于零分配。
  std::optional<Bytes> logicalBytes() const;

private:
  TensorId mId;
  std::string mName;
  TensorSignature mSignature;
  ValueKind mKind;
  std::optional<ParameterRef> mParameter;
  AxisOrigin mAxesOrigin;
};

} // namespace opfusion
