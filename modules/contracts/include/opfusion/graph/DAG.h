#pragma once

#include "opfusion/graph/Operator.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

/// 按生产/消费端口派生的逻辑依赖，不代表设备复制或独立 Buffer。
/// 同值被多个输入端口使用时保留多条边；端口列表仍是唯一关系真值。
struct Edge {
  EdgeId id = 0;
  OperatorId producer = 0;
  std::size_t outputPort = 0;
  OperatorId consumer = 0;
  std::size_t inputPort = 0;
  TensorId tensor = 0;
};

/// 拥有逻辑张量及算子的不可复制、不可移动图，发布前完成结构校验和轴收敛。
/// 节点 ID/张量 ID 唯一、每个结果仅一个生产端口，节点依赖必须无环。
/// 不含设备、执行缓冲或调度；只读访问不使任意外部算子实现自动线程安全。
class DAG {
 public:
  /// 取得 tensors 的值和 nodes 的独占所有权，派生端口边及稳定拓扑顺序。
  /// 显式模型接口须完整且无重复；nullopt 保留接口未知，空列表表示显式无端口。
  /// 推导只读输入签名，再与声明输出核对；失败抛出 Error，不发布部分图。
  DAG(std::string id, std::string name, std::vector<Tensor> tensors,
      std::vector<std::unique_ptr<Operator>> nodes,
      std::optional<std::vector<TensorId>> modelOutputs = std::nullopt,
      std::string registryId = {},
      std::optional<std::vector<TensorId>> modelInputs = std::nullopt);
  DAG(const DAG&) = delete;
  DAG& operator=(const DAG&) = delete;
  /// 以下只读引用借用本图拥有的记录，限本图存续期；不会重新装配 Registry。
  const std::string& id() const noexcept { return mId; }
  /// 借用图的诊断名称，不以名称推断语义或布局。
  const std::string& name() const noexcept { return mName; }
  /// 借用装配环境身份；空值表示调用方未提供，不代表自动生成内容摘要。
  const std::string& registryId() const noexcept { return mRegistryId; }
  /// 借用唯一逻辑值表，不包含运行时分配。
  const std::vector<Tensor>& tensors() const noexcept { return mTensors; }
  /// 借用只读节点所有权表；调用方不得释放或修改节点。
  const std::vector<std::unique_ptr<const Operator>>& nodes() const noexcept { return mNodes; }
  /// 借用从端口列表派生的依赖边，保留重复操作数的独立使用。
  const std::vector<Edge>& edges() const noexcept { return mEdges; }
  /// 按 ID 借用张量；缺失抛出 tensor_missing。
  const Tensor& tensor(TensorId id) const;
  /// 按 ID 借用节点；缺失抛出 node_missing。
  const Operator& node(OperatorId id) const;
  /// 借用拓扑序；同时就绪节点按 ID 升序选取，不表示运行调度。
  const std::vector<OperatorId>& topologicalOrder() const noexcept { return mOrder; }
  /// 借用无内部生产者的边界值，可能包含参数或角色尚未知的值。
  const std::vector<TensorId>& externalInputs() const noexcept { return mExternal; }
  /// 借用无内部消费者的结果；不等同于完整模型输出声明。
  const std::vector<TensorId>& terminalOutputs() const noexcept { return mTerminal; }
  /// 借用显式模型输出；允许直接引用模型输入，nullopt 不猜测完整接口。
  const std::optional<std::vector<TensorId>>& modelOutputs() const noexcept {
    return mModelOutputs;
  }
  /// 借用显式运行时输入，不将固定参数伪装成模型输入。
  const std::optional<std::vector<TensorId>>& modelInputs() const noexcept { return mModelInputs; }
  /// 重验输出声明、input-only 推导、数值约束和索引关系；失败附节点上下文。
  /// 不回写已发布图，也不执行算子数值计算。
  void validate() const;
 private:
  // 仅构造期间单调补全轴知识；新增记录替换构造中张量，不改 shape/dtype。
  void resolveAxes();
  bool mergeTensorAxes(TensorId id, const std::optional<LogicalAxes>& axes);
  std::string mId, mName, mRegistryId;
  std::vector<Tensor> mTensors;
  std::vector<std::unique_ptr<const Operator>> mNodes;
  std::map<TensorId, std::size_t> mTensorIndex;
  std::map<OperatorId, std::size_t> mNodeIndex;
  std::vector<Edge> mEdges;
  std::vector<OperatorId> mOrder;
  std::vector<TensorId> mExternal, mTerminal;
  std::optional<std::vector<TensorId>> mModelOutputs, mModelInputs;
};

}  // namespace opfusion
