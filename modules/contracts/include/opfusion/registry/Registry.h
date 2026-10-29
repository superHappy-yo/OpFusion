#pragma once

#include "opfusion/graph/Operator.h"
#include "opfusion/registry/Source.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

/// 公共头路径分层后的 C++ 契约版本；与算子工厂 API 和数学语义分别演进。
inline constexpr char contractVersion[] = "6.1.0";
/// API 5 增加取模、间接索引、元数据读取与归约阶段；旧静态插件须重新编译。
inline constexpr char operatorApiVersion[] = "5.0.0";

/// 只声明语义身份；来源选择和工厂由 SourceBinding 单独维护。
struct Registration {
  SemanticIdentity semantic;
};

/// 版本化来源格式与节点标签；operation 缺省时匹配该格式/类型的任意 operation。
struct SourceSelector {
  std::string format;
  std::string type;
  std::optional<std::string> operation;
};

/// 同步读取借用的来源节点，返回新算子的独占所有权；不能保留来源引用。
using OperatorFactory = std::function<std::unique_ptr<Operator>(const SourceNode &)>;

/// 一对多规范化结果；附加张量只包含新值，不覆盖原值。
struct SourceExpansion {
  std::vector<std::unique_ptr<Operator>> nodes;
  std::vector<Tensor> tensors;
  std::vector<std::string> notes;
};
using SourceNormalizer = std::function<SourceExpansion(const SourceNode &)>;

/// 来源适配贡献；semantic 必须指向已登记的语义，而非复制领域行为。
struct SourceBinding {
  SemanticIdentity semantic;
  std::vector<SourceSelector> selectors;
  OperatorFactory factory;
  SourceNormalizer normalizer;
};

/// 调用方拥有的语义与来源目录；显式填充后冻结，不探测或执行设备。
/// 查询引用借用自本实例，装配期间 vector 扩容及实例销毁会使相关引用失效。
/// 冻结不使工厂捕获的外部状态自动线程安全，不提供并发装配同步。
class Registry {
public:
  /// 登记唯一完整语义；冻结、身份不完整或重复时抛出 Error。
  void add(Registration registration);
  /// 绑定已登记语义；目标缺失、空工厂/selector 或通配歧义时抛出 Error。
  /// 冻结后禁止修改，同一次失败不会留下部分绑定。
  void bind(SourceBinding binding);
  /// 保存调用方提供的非空内容身份；不自行计算摘要，重复冻结时抛出 Error。
  void freeze(std::string identity);

  /// 返回装配是否结束；冻结只禁止目录变更，不证明工厂数值正确。
  bool frozen() const noexcept { return mFrozen; }
  /// 借用调用方提供的身份；冻结前为空，引用不独立拥有字符串。
  const std::string &id() const noexcept { return mId; }
  /// 借用语义声明表；不包含来源 selector 或 factory。
  const std::vector<Registration> &registrations() const noexcept { return mRegistrations; }
  /// 借用独立来源绑定表；冻结后目录不再扩容，但工厂外部状态仍由提供者负责。
  const std::vector<SourceBinding> &sourceBindings() const noexcept { return mSourceBindings; }
  /// 在冻结目录中唯一匹配来源，返回新算子的独占所有权。
  /// 未冻结、未知来源、空结果或返回语义不符时抛出 Error；工厂异常原样传播。
  std::unique_ptr<Operator> create(const SourceNode &source) const;
  /// 执行来源绑定自己的规范化，不按算子名分派。
  SourceExpansion expand(const SourceNode &source) const;

private:
  bool mFrozen = false;
  std::string mId;
  std::vector<Registration> mRegistrations;
  std::vector<SourceBinding> mSourceBindings;
};

} // namespace opfusion
