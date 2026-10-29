#include "opfusion/graph/DAG.h"

#include "opfusion/common/Error.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <utility>

// 构造期间取得逻辑记录所有权并派生关系；发布后只保留 const 算子。
namespace opfusion {
DAG::DAG(std::string id, std::string name, std::vector<Tensor> tensors,
         std::vector<std::unique_ptr<Operator>> nodes,
         std::optional<std::vector<TensorId>> modelOutputs, std::string registryId,
         std::optional<std::vector<TensorId>> modelInputs)
    : mId(std::move(id)), mName(std::move(name)), mRegistryId(std::move(registryId)),
      mTensors(std::move(tensors)), mModelOutputs(std::move(modelOutputs)),
      mModelInputs(std::move(modelInputs)) {
  require(!mId.empty(), "id_invalid", "DAG identity cannot be empty");
  for (std::size_t i = 0; i < mTensors.size(); ++i) {
    require(mTensorIndex.emplace(mTensors[i].id(), i).second, "duplicate_tensor",
            "duplicate tensor ID");
  }
  for (auto &node : nodes) {
    require(node != nullptr, "node_invalid", "null operator");
    require(mNodeIndex.emplace(node->id(), mNodes.size()).second, "duplicate_node",
            "duplicate operator ID");
    mNodes.push_back(std::move(node));
  }
  // 生产者和边均从端口列表派生，不给同一逻辑值维护第二套可变关系真值。
  std::map<TensorId, std::pair<OperatorId, std::size_t>> producers;
  std::map<TensorId, std::size_t> uses;
  for (const auto &node : mNodes) {
    for (std::size_t p = 0; p < node->outputs().size(); ++p) {
      auto value = node->outputs()[p];
      const auto &t = tensor(value);
      require(t.kind() == ValueKind::Unknown || t.kind() == ValueKind::OperatorResult,
              "value_role_mismatch", "an input or parameter cannot have an operator producer");
      require(producers.emplace(value, std::make_pair(node->id(), p)).second, "multiple_producers",
              "tensor has more than one producer port");
    }
  }
  const auto setKind = [&](TensorId value, ValueKind kind) {
    auto &t = mTensors.at(mTensorIndex.at(value));
    t = Tensor(t.id(), t.name(), t.shape(), t.dtype(), t.signature().axes, kind, t.parameter(),
               t.axesOrigin());
  };
  for (const auto &[value, producer] : producers) {
    (void)producer;
    setKind(value, ValueKind::OperatorResult);
  }
  if (mModelInputs) {
    std::set<TensorId> seen;
    for (auto value : *mModelInputs) {
      const auto &t = tensor(value);
      require(seen.insert(value).second && !producers.count(value) &&
                  (t.kind() == ValueKind::Unknown || t.kind() == ValueKind::Input),
              "input_invalid",
              "explicit model inputs must be unique, external, non-parameter values");
      setKind(value, ValueKind::Input);
    }
    for (const auto &t : mTensors) {
      require(t.kind() != ValueKind::Input || seen.count(t.id()), "input_invalid",
              "input role is missing from the explicit model interface");
    }
  }
  for (const auto &t : mTensors) {
    require(t.kind() != ValueKind::OperatorResult || producers.count(t.id()), "value_role_mismatch",
            "operator result has no producer port");
  }
  std::set<TensorId> external;
  std::map<OperatorId, std::size_t> degree;
  std::map<OperatorId, std::vector<OperatorId>> successors;
  for (const auto &node : mNodes) {
    degree[node->id()] = 0;
  }
  for (const auto &node : mNodes) {
    for (std::size_t p = 0; p < node->inputs().size(); ++p) {
      auto value = node->inputs()[p];
      (void)tensor(value);
      ++uses[value];
      auto from = producers.find(value);
      if (from == producers.end()) {
        external.insert(value);
      } else {
        mEdges.push_back({static_cast<EdgeId>(mEdges.size() + 1), from->second.first,
                          from->second.second, node->id(), p, value});
        // 重复使用占独立端口，因此入度和后继项也按使用次数成对计数。
        ++degree[node->id()];
        successors[from->second.first].push_back(node->id());
      }
    }
  }
  if (mModelInputs) {
    external.insert(mModelInputs->begin(), mModelInputs->end());
  }
  if (mModelOutputs) {
    for (auto value : *mModelOutputs) {
      (void)tensor(value);
      if (!producers.count(value)) {
        external.insert(value);
      }
    }
  }
  mExternal.assign(external.begin(), external.end());
  if (mModelInputs) {
    for (auto value : mExternal) {
      require(
          tensor(value).kind() == ValueKind::Input || tensor(value).kind() == ValueKind::Parameter,
          "input_invalid", "external value is neither an explicit input nor a declared parameter");
    }
  }
  for (const auto &[value, producer] : producers) {
    (void)producer;
    if (!uses.count(value)) {
      mTerminal.push_back(value);
    }
  }
  std::priority_queue<OperatorId, std::vector<OperatorId>, std::greater<OperatorId>> ready;
  for (const auto &[node, count] : degree) {
    if (count == 0) {
      ready.push(node);
    }
  }
  while (!ready.empty()) {
    auto idValue = ready.top();
    ready.pop();
    mOrder.push_back(idValue);
    for (auto next : successors[idValue]) {
      if (--degree[next] == 0) {
        ready.push(next);
      }
    }
  }
  require(mOrder.size() == mNodes.size(), "graph_cycle", "operator graph contains a cycle");
  resolveAxes();
  validate();
}
const Tensor &DAG::tensor(TensorId id) const {
  auto it = mTensorIndex.find(id);
  require(it != mTensorIndex.end(), "tensor_missing", "unknown tensor ID " + std::to_string(id));
  return mTensors[it->second];
}
const Operator &DAG::node(OperatorId id) const {
  auto it = mNodeIndex.find(id);
  require(it != mNodeIndex.end(), "node_missing", "unknown operator ID " + std::to_string(id));
  return *mNodes[it->second];
}
bool DAG::mergeTensorAxes(TensorId id, const std::optional<LogicalAxes> &axes) {
  if (!axes) {
    return false;
  }
  auto &t = mTensors.at(mTensorIndex.at(id));
  validateAxes(axes, t.shape().size());
  const auto merged = mergeAxes(t.signature().axes, axes);
  if (merged == t.signature().axes) {
    return false;
  }
  const auto origin = t.axesOrigin() == AxisOrigin::Declared || t.axesOrigin() == AxisOrigin::Mixed
                          ? AxisOrigin::Mixed
                          : AxisOrigin::Derived;
  t = Tensor(t.id(), t.name(), t.shape(), t.dtype(), merged, t.kind(), t.parameter(), origin);
  return true;
}
void DAG::resolveAxes() {
  // 发布前只做单调补全，不检查算子名称/具体类，也不回读输出 shape 作推导答案。
  // 局部 inputs 是值副本：替换构造中的张量不会使本轮推导的借用失效。
  bool changed;
  do {
    changed = false;
    for (auto id : mOrder) {
      const auto &op = node(id);
      try {
        std::vector<TensorSignature> inputs;
        for (auto value : op.inputs()) {
          inputs.push_back(tensor(value).signature());
        }
        op.validate(inputs);
        const auto inferred = op.infer(inputs);
        op.validateNumerics(inferred);
        require(inferred.size() == op.outputs().size(), "port_invalid",
                "inferred output count mismatch");
        for (std::size_t p = 0; p < inferred.size(); ++p) {
          const auto &declared = tensor(op.outputs()[p]);
          require(inferred[p].shape == declared.shape() && inferred[p].dtype == declared.dtype(),
                  "shape_mismatch",
                  "declared output differs from inferred shape/dtype at port " + std::to_string(p));
          changed = mergeTensorAxes(op.outputs()[p], inferred[p].axes) || changed;
        }
        const auto semantics = op.indexSemantics(inputs);
        semantics.validate(inputs, inferred);
        for (const auto &piece : semantics.pieces) {
          for (const auto &access : piece.inputs) {
            const auto inputId = op.inputs().at(access.port);
            changed = mergeTensorAxes(inputId, access.axes) || changed;
            if (!access.preservesAxes) {
              continue;
            }
            for (std::size_t d = 0; d < access.coordinates.size(); ++d) {
              const auto &from = access.coordinates[d];
              if (!from.lookups.empty() || from.terms.size() != 1 ||
                  from.terms[0].coefficient <= 0 || piece.domain[from.terms[0].axis].reduction) {
                continue;
              }
              for (const auto &output : piece.outputs) {
                for (std::size_t e = 0; e < output.coordinates.size(); ++e) {
                  const auto &to = output.coordinates[e];
                  if (!to.lookups.empty() || to.terms.size() != 1 ||
                      to.terms[0].axis != from.terms[0].axis || to.terms[0].coefficient != 1) {
                    continue;
                  }
                  const auto outputId = op.outputs().at(output.port);
                  const auto &read = from.terms[0];
                  const auto &write = to.terms[0];
                  const bool affine = !read.modulus && !write.modulus && write.divisor == 1;
                  // 仅传播算子已承诺保留的、完全相同的取模坐标；不把相同元素数当成相同轴。
                  const bool modular = read.modulus && write.modulus && *read.modulus > 1 &&
                                       read.coefficient == 1 && read.divisor == write.divisor &&
                                       read.modulus == write.modulus && from.offset == to.offset &&
                                       from.offset == 0 &&
                                       *read.modulus == tensor(inputId).shape()[d] &&
                                       *write.modulus == tensor(outputId).shape()[e];
                  if (!affine && !modular) {
                    continue;
                  }
                  const auto inputAxes = tensor(inputId).signature().axes;
                  const auto outputAxes = tensor(outputId).signature().axes;
                  if (inputAxes) {
                    LogicalAxes axes(tensor(outputId).shape().size());
                    axes[e] = (*inputAxes)[d];
                    changed = mergeTensorAxes(outputId, axes) || changed;
                  }
                  if (outputAxes) {
                    LogicalAxes axes(tensor(inputId).shape().size());
                    axes[d] = (*outputAxes)[e];
                    changed = mergeTensorAxes(inputId, axes) || changed;
                  }
                }
              }
            }
          }
        }
      } catch (const Error &error) {
        throw Error(error.code(), "node " + std::to_string(id) + " (" + op.semantic().name +
                                      "): " + error.what());
      }
    }
  } while (changed);
}
void DAG::validate() const {
  if (mModelOutputs) {
    std::set<TensorId> seen;
    for (auto value : *mModelOutputs) {
      (void)tensor(value);
      require(seen.insert(value).second, "output_invalid", "duplicate model output");
    }
  }
  for (auto id : mOrder) {
    const auto &op = node(id);
    try {
      std::vector<TensorSignature> inputs;
      for (auto value : op.inputs()) {
        inputs.push_back(tensor(value).signature());
      }
      op.validate(inputs);
      const auto inferred = op.infer(inputs);
      op.validateNumerics(inferred);
      require(inferred.size() == op.outputs().size(), "port_invalid",
              "inferred output count mismatch");
      for (std::size_t p = 0; p < inferred.size(); ++p) {
        (void)elements(inferred[p].shape);
        validateAxes(inferred[p].axes, inferred[p].shape.size());
        const auto &declared = tensor(op.outputs()[p]).signature();
        require(inferred[p].shape == declared.shape && inferred[p].dtype == declared.dtype,
                "shape_mismatch",
                "declared output differs from inferred shape/dtype at port " + std::to_string(p));
        require(compatibleAxes(inferred[p].axes, declared.axes), "axis_mismatch",
                "declared output axes conflict with inference");
      }
      op.indexSemantics(inputs).validate(inputs, inferred);
    } catch (const Error &error) {
      throw Error(error.code(),
                  "node " + std::to_string(id) + " (" + op.semantic().name + "): " + error.what());
    }
  }
}
} // namespace opfusion
