#pragma once

#include "opfusion/graph/IndexSemantics.h"
#include "opfusion/graph/Tensor.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace opfusion {

/// 开放的 namespace/name 身份；兼容性由契约和不可变目录内容身份管理。
struct SemanticIdentity {
  std::string nameSpace = "opfusion";
  std::string name;
  /// 内部键保留命名空间，不拼接算子版本。
  std::string key() const { return nameSpace + ":" + name; }
};

/// 已验证属性的只读展示值；不是具体算子的内部配置或来源字段袋。
using AttributeValue =
    std::variant<bool, std::int64_t, double, std::string, Shape, std::vector<double>>;
struct Attribute {
  std::string name;
  AttributeValue value;
};
struct PortBroadcast {
  std::size_t port = 0;
  std::vector<std::size_t> axes;
};
/// applied 是本实例行为，不能由算子支持广播的 trait 替代。
struct BroadcastInfo {
  bool applied = false;
  std::vector<PortBroadcast> inputs;
};

/// 每项独立表达已知事实；nullopt 不等于 false 或不支持。
struct Traits {
  std::optional<bool> pure;
  std::optional<bool> elementwise;
  std::optional<bool> reduction;
  std::optional<bool> broadcasting;
};

/// 逻辑算子的公共行为协议，拥有身份、按端口有序的值引用和数值要求。
/// 不保存 Registry、来源模型、设备句柄或缓冲；DAG 取得实例的独占所有权。
/// 只读引用借用本实例存储，限实例存续期；具体提供者自行保证只读调用安全。
class Operator {
public:
  /// 校验公共身份/端口/数值属性；不在父类构造中调用虚函数。
  Operator(OperatorId id, SemanticIdentity semantic, std::vector<TensorId> inputs,
           std::vector<TensorId> outputs, NumericalRequirements requirements = {});
  /// 支持经公共协议销毁具体实例，不承担设备资源释放或执行工作。
  virtual ~Operator() = default;
  Operator(const Operator &) = delete;
  Operator &operator=(const Operator &) = delete;
  /// 返回逻辑节点身份，不是来源索引或设备任务身份。
  OperatorId id() const noexcept { return mId; }
  /// 借用完整数学语义身份，不按名称猜测实现支持。
  const SemanticIdentity &semantic() const noexcept { return mSemantic; }
  /// 借用有序输入值引用；允许重复，同值多次使用仍占不同端口。
  const std::vector<TensorId> &inputs() const noexcept { return mInputs; }
  /// 借用有序输出值引用；值是否唯一生产由 DAG 负责检查。
  const std::vector<TensorId> &outputs() const noexcept { return mOutputs; }
  /// 借用实例数值约束声明，不是数值验收证据。
  const NumericalRequirements &requirements() const noexcept { return mRequirements; }
  /// 仅按借用的有序输入签名及自身属性推导输出；不能读取声明输出作答案。
  /// 非法输入或未支持模式抛出 Error，不执行数值 kernel。
  virtual std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const = 0;
  /// 返回语义特征；未知事实保持 nullopt，不代表设备支持结论。
  virtual Traits traits() const = 0;
  /// 仅在整数输入内容已知时推导静态形状值；nullopt 不表示零。
  virtual std::vector<std::optional<Shape>>
  evaluateIntegers(const std::vector<TensorSignature> &,
                   const std::vector<std::optional<Shape>> &) const {
    return std::vector<std::optional<Shape>>(mOutputs.size());
  }
  /// 返回具体类已验证属性的值副本；无属性算子返回空列表。
  virtual std::vector<Attribute> attributes(const std::vector<TensorSignature> &) const {
    return {};
  }
  /// 描述实际广播；默认仅处理逐元素算子，矩阵批广播由 MatMul 独立提供。
  virtual BroadcastInfo broadcasting(const std::vector<TensorSignature> &inputs) const;
  /// 描述域、端口访问与归约；Unknown/Unsupported 需附原因，不伪造部分关系。
  virtual IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const = 0;
  /// 校验签名数量、静态形状、轴和数值要求，再调用具体类的属性钩子。
  /// 只借用输入，失败抛出 Error；不修改算子或声明张量。
  void validate(const std::vector<TensorSignature> &inputs) const;
  /// 核对给定签名的 dtype 是否满足实例数值要求，不验证设备执行误差。
  void validateNumerics(const std::vector<TensorSignature> &signatures) const;

protected:
  /// 具体类完成成员初始化后调用，检查本类端口、属性与支持模式。
  virtual void validateAttributes() const = 0;

private:
  OperatorId mId;
  SemanticIdentity mSemantic;
  std::vector<TensorId> mInputs;
  std::vector<TensorId> mOutputs;
  NumericalRequirements mRequirements;
};

} // namespace opfusion
