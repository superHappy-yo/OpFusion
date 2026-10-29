#pragma once

#include "opfusion/graph/DAG.h"
#include "opfusion/graph/Tensor.h"
#include "opfusion/registry/Registry.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opfusion::frontend {

/// 原样保存的来源存储提示；单位未建立，不是物理表示、Buffer 或分配契约。
struct TensorOrigin {
  /// 缺省为空，不据此推断连续存储；数值单位不假定为元素或字节。
  std::vector<std::int64_t> strides;
  /// 缺省为 nullopt；即使有值也不是已验证的物理字节对齐。
  std::optional<std::uint64_t> alignment;
  /// 以下 nullopt 表示来源未声明，与已知 false 区分，不决定值的生命周期。
  std::optional<bool> virtualTensor;
  std::optional<bool> runtimePassByValue;
};

/// 规范节点到来源位置的映射；ONNX 允许多个规范节点对应同一来源节点。
struct NodeOrigin {
  OperatorId id = 0;
  /// 从 0 开始的来源数组位置，不作为跨导入稳定身份。
  std::size_t index = 0;
  std::string name;
  std::vector<std::string> inputs, outputs, normalizations;
};

/// 来源或阶段限制诊断；code 标识诊断类别，message 保留上下文，不是设备验收结果。
struct Diagnostic {
  std::string code;
  std::string message;
};

struct ParameterOrigin {
  TensorId id = 0;
  std::string name, contentDigest, externalLocation;
  std::uint64_t offset = 0, length = 0;
};

/// 一次成功导入的图与自有来源记录；失败抛出 Error，不交付部分结果。
/// 只建立逻辑模型，模型完整接口和执行条件是否已知须查询图与 diagnostics。
struct ImportResult {
  /// 与调用方共享已发布的只读 DAG，不借用导入文本或原 Registry。
  std::shared_ptr<const DAG> graph;
  /// 包括未被节点引用的声明张量；key 为 UID 而非来源数组位置。
  std::map<TensorId, TensorOrigin> tensorOrigins;
  /// 保存每个规范节点的原始来源位置、端口名称和规范化说明。
  std::vector<NodeOrigin> nodeOrigins;
  std::vector<Diagnostic> diagnostics;
  /// 具体来源子集的身份；JsonGraph 当前固定为 benchmark-json/1。
  std::string profile;
  std::string sourceDigest, modelContentDigest, producer, producerVersion;
  std::optional<std::int64_t> irVersion;
  std::map<std::string, std::int64_t> opsets;
  std::vector<ParameterOrigin> parameters;
};

/// 已实现的来源适配，不因文件扩展名自动接受任意 JSON 或 ONNX。
/// 非法枚举值在导入入口以 model_format_unsupported 拒绝。
enum class ModelFormat { Auto, JsonGraph, Onnx };

/// 拥有冻结 Registry 的值快照，格式读取器保留在 frontend 内部。
/// 不执行设备工作；复制目录不会延长工厂所借用的外部状态寿命，并发安全仍取决于工厂。
class ModelImporter {
public:
  /// 按值保存快照，传左值时复制；原 Registry 可在构造后销毁。
  /// 未冻结时抛出 model_registry_unfrozen；不隐式登记语义/绑定或冻结目录。
  explicit ModelImporter(Registry registry);
  /// 同步借用文件路径并读取指定格式；不保存路径引用；Auto 仅按 .onnx/.json 选择已实现的适配器。
  /// 文件/格式/schema/公共图校验失败抛出 Error，不返回部分图。
  ImportResult importFile(const std::string &path, ModelFormat format = ModelFormat::Auto) const;
  /// 同步借用文本和诊断来源名，成功返回新图与映射；不保存传入引用。
  /// 格式、schema 或公共图校验失败抛出 Error，不返回部分图。
  ImportResult importText(const std::string &text, ModelFormat format = ModelFormat::JsonGraph,
                          const std::string &sourceName = "<memory>") const;

private:
  const Registry mRegistry;
};

} // namespace opfusion::frontend
