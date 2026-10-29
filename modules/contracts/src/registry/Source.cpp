#include "opfusion/registry/Source.h"
#include <cmath>

// 自有来源值树的严格读取边界；返回的引用均借用树，不延长其生命周期。
namespace opfusion {
const SourceValue::Object& SourceValue::object() const {
  auto p = std::get_if<Object>(&value);
  require(p != nullptr, "schema_invalid", "expected object"); return *p;
}
const SourceValue::Array& SourceValue::array() const {
  auto p = std::get_if<Array>(&value);
  require(p != nullptr, "schema_invalid", "expected array"); return *p;
}
const std::string& SourceValue::string() const {
  auto p = std::get_if<std::string>(&value);
  require(p != nullptr, "schema_invalid", "expected string"); return *p;
}
std::int64_t SourceValue::integer() const {
  auto p = std::get_if<std::int64_t>(&value);
  require(p != nullptr, "schema_invalid", "expected signed integer"); return *p;
}
double SourceValue::number() const {
  // 仅 number() 允许整数提升；integer() 不接受浮点 UID，防止精度损失改变身份。
  double result;
  if (auto p = std::get_if<double>(&value)) result = *p;
  else if (auto integer = std::get_if<std::int64_t>(&value)) result = static_cast<double>(*integer);
  else throw Error("schema_invalid", "expected number");
  require(std::isfinite(result), "schema_invalid", "number must be finite"); return result;
}
bool SourceValue::boolean() const {
  auto p = std::get_if<bool>(&value);
  require(p != nullptr, "schema_invalid", "expected boolean"); return *p;
}
const SourceValue& SourceValue::at(const std::string& key) const {
  const auto& obj = object(); auto it = obj.find(key);
  require(it != obj.end(), "field_missing", "missing source field: " + key); return it->second;
}
const SourceValue* SourceValue::find(const std::string& key) const {
  const auto& obj = object(); auto it = obj.find(key);
  return it == obj.end() ? nullptr : &it->second;
}
}  // namespace opfusion
