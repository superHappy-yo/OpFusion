#pragma once

#include "opfusion/common/Error.h"
#include "opfusion/graph/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace opfusion {

/// 来源边界的自有值树，不暴露解析库对象；null、缺字段及类型错误分别处理。
/// 字符串/容器访问器只借用本树，调用方修改、替换或销毁对应值后不得沿用引用。
struct SourceValue {
  using Array = std::vector<SourceValue>;
  using Object = std::map<std::string, SourceValue>;
  using Storage =
      std::variant<std::monostate, bool, std::int64_t, double, std::string, Array, Object>;
  Storage value;
  /// 默认构造显式 null 值，不表示字段不存在。
  SourceValue() = default;
  /// 按值取得一种 Storage 支持的来源类型，不执行 schema 或语义转换。
  template <class T> explicit SourceValue(T v) : value(std::move(v)) {}
  /// 区分显式 null 与其他已存储值，不做真值转换。
  bool isNull() const { return std::holds_alternative<std::monostate>(value); }
  /// 借用对象；类型不符抛出 schema_invalid，以下同类访问器不隐式转换。
  const Object &object() const;
  /// 借用数组，引用有效期受对应值树的存续和修改限制。
  const Array &array() const;
  /// 借用字符串，不把数字或 null 格式化为文本。
  const std::string &string() const;
  /// 仅接受 int64_t；浮点值即使数学上为整数也不接受。
  std::int64_t integer() const;
  /// 接受整数或浮点数并返回有限 double；可能发生整数到浮点的精度损失。
  double number() const;
  /// 仅接受 bool，不把零、空容器或 null 当作 false。
  bool boolean() const;
  /// 在对象中借用必需字段；非对象为 schema_invalid，缺字段为 field_missing。
  const SourceValue &at(const std::string &key) const;
  /// 在对象中借用可选字段；缺字段返回 nullptr，显式 null 仍返回非空指针。
  const SourceValue *find(const std::string &key) const;
};

/// 来源节点的值快照，供注册工厂同步借用；不等同于规范化算子身份。
/// format 含来源版本，operation 缺省保留未声明；fields 不自动匹配任何 schema。
struct SourceNode {
  std::string format;
  std::string type;
  std::optional<std::string> operation;
  std::string name;
  /// 来源顺序位置，不是 TensorId；具体工厂负责索引转换与溢出检查。
  std::size_t index = 0;
  SourceValue fields;
  /// 静态来源适配的有序端口、签名与可证明的整数内容。
  std::vector<TensorId> inputs, outputs;
  std::vector<TensorSignature> inputSignatures;
  std::vector<std::optional<Shape>> integerInputs;
  std::vector<std::optional<std::vector<double>>> floatingInputs;
  TensorId nextTensor = 0;
};

} // namespace opfusion
