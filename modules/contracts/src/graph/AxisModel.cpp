// 结构轴从静态签名与已验证访问关系推导，不按算子名称或 tensor 名称分派。
#include "opfusion/graph/AxisModel.h"

#include "opfusion/common/Error.h"
#include "opfusion/graph/DAG.h"

#include <algorithm>
#include <set>
#include <utility>
namespace opfusion {
namespace {
bool same(const IndexExpression &a, const IndexExpression &b) {
  if (a.offset != b.offset || a.terms.size() != b.terms.size() ||
      a.lookups.size() != b.lookups.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.terms.size(); ++i) {
    const auto &x = a.terms[i];
    const auto &y = b.terms[i];
    if (x.axis != y.axis || x.coefficient != y.coefficient || x.divisor != y.divisor ||
        x.modulus != y.modulus) {
      return false;
    }
  }
  for (std::size_t i = 0; i < a.lookups.size(); ++i) {
    const auto &x = a.lookups[i];
    const auto &y = b.lookups[i];
    if (x.port != y.port || x.extent != y.extent || x.coordinates.size() != y.coordinates.size()) {
      return false;
    }
    for (std::size_t j = 0; j < x.coordinates.size(); ++j) {
      if (!same(x.coordinates[j], y.coordinates[j])) {
        return false;
      }
    }
  }
  return true;
}
std::set<std::size_t> variables(const IndexExpression &expression) {
  std::set<std::size_t> result;
  for (const auto &t : expression.terms) {
    if (!t.modulus || *t.modulus != 1) {
      result.insert(t.axis);
    }
  }
  for (const auto &lookup : expression.lookups) {
    for (const auto &coord : lookup.coordinates) {
      const auto vars = variables(coord);
      result.insert(vars.begin(), vars.end());
    }
  }
  return result;
}
bool shares(const IndexExpression &a, const IndexExpression &b) {
  const auto x = variables(a), y = variables(b);
  for (auto v : x) {
    if (y.count(v)) {
      return true;
    }
  }
  return false;
}
// 对标准逻辑行主序商/余数坐标识别拆合区间；不可整除的混合重排仍保留完整表达式。
bool radix(const IndexExpression &expression, std::int64_t extent) {
  return expression.offset == 0 && expression.lookups.empty() && expression.terms.size() == 1 &&
         expression.terms[0].coefficient == 1 && expression.terms[0].modulus == extent;
}
std::vector<AxisSource> sources(const DAG &graph, const Operator &op, const IndexPiece &piece,
                                const IndexExpression &write, std::int64_t extent) {
  std::vector<AxisSource> result;
  for (std::size_t access = 0; access < piece.inputs.size(); ++access) {
    const auto &input = piece.inputs[access];
    const auto &tensor = graph.tensor(op.inputs().at(input.port));
    for (std::size_t d = 0; d < input.coordinates.size(); ++d) {
      const auto &read = input.coordinates[d];
      bool related = shares(read, write);
      if (related && radix(write, extent) && radix(read, tensor.shape()[d]) && extent > 1 &&
          tensor.shape()[d] > 1) {
        const auto outStride = write.terms[0].divisor, inStride = read.terms[0].divisor;
        const auto outEnd =
            checkedMultiply(static_cast<Bytes>(outStride), static_cast<Bytes>(extent));
        const auto inEnd =
            checkedMultiply(static_cast<Bytes>(inStride), static_cast<Bytes>(tensor.shape()[d]));
        related = static_cast<Bytes>(std::max(outStride, inStride)) < std::min(outEnd, inEnd);
      }
      if (related) {
        result.push_back({tensor.id(), input.port, access, d, read});
      }
    }
  }
  return result;
}
AxisStructure structure(const DAG &graph, const IndexPiece &piece, const IndexExpression &write,
                        std::int64_t extent, const std::vector<AxisSource> &refs) {
  if (!piece.shapeReads.empty()) {
    return AxisStructure::Metadata;
  }
  if (piece.inputs.empty()) {
    return AxisStructure::Generated;
  }
  if (extent == 1) {
    return AxisStructure::Singleton;
  }
  for (const auto &ref : refs) {
    if (!ref.read.lookups.empty()) {
      return AxisStructure::Indirect;
    }
  }
  if (refs.empty()) {
    return AxisStructure::Reindexed;
  }
  bool identical = true;
  for (const auto &ref : refs) {
    identical &= same(ref.read, write) && graph.tensor(ref.tensor).shape()[ref.dimension] == extent;
  }
  if (identical) {
    return AxisStructure::Direct;
  }
  if (radix(write, extent)) {
    const auto stride = write.terms[0].divisor;
    const auto end = checkedMultiply(stride, extent);
    if (refs.size() == 1) {
      const auto &ref = refs[0];
      const auto inputExtent = graph.tensor(ref.tensor).shape()[ref.dimension];
      if (radix(ref.read, inputExtent)) {
        const auto inputStride = ref.read.terms[0].divisor;
        const auto inputEnd = checkedMultiply(inputStride, inputExtent);
        if (stride >= inputStride && stride % inputStride == 0 && end <= inputEnd &&
            inputExtent > extent) {
          return AxisStructure::Split;
        }
      }
    } else {
      Bytes product = 1;
      bool merged = true;
      std::optional<TensorId> source;
      for (const auto &ref : refs) {
        const auto inputExtent = graph.tensor(ref.tensor).shape()[ref.dimension];
        if (!radix(ref.read, inputExtent) || (source && *source != ref.tensor)) {
          merged = false;
          break;
        }
        source = ref.tensor;
        const auto inputStride = ref.read.terms[0].divisor;
        merged &= inputStride >= stride && inputStride % stride == 0 &&
                  checkedMultiply(inputStride, inputExtent) <= end;
        product = checkedMultiply(product, inputExtent);
      }
      if (merged && product == static_cast<Bytes>(extent)) {
        return AxisStructure::Merged;
      }
    }
  }
  return AxisStructure::Reindexed;
}
std::vector<TensorSignature> inputSignatures(const DAG &graph, const Operator &op) {
  std::vector<TensorSignature> inputs;
  for (auto id : op.inputs()) {
    inputs.push_back(graph.tensor(id).signature());
  }
  return inputs;
}
TensorAxisModel makeModel(const DAG &graph, const Tensor &tensor, const Operator *producer,
                          std::size_t outputPort, const IndexSemantics *semantics) {
  TensorAxisModel model;
  model.tensor = tensor.id();
  model.graphId = graph.id();
  model.registryId = graph.registryId();
  model.status = SemanticStatus::Known;
  model.origin = tensor.shape().empty() ? AxisModelOrigin::Scalar
                 : producer             ? AxisModelOrigin::OperatorRelation
                                        : AxisModelOrigin::Signature;
  if (tensor.signature().axes) {
    model.semanticOrigin = tensor.axesOrigin();
  }
  if (producer) {
    model.producer = producer->id();
    model.outputPort = outputPort;
    model.inputs = producer->inputs();
    model.status = semantics->status;
    model.reason = semantics->reason;
  }
  for (std::size_t d = 0; d < tensor.shape().size(); ++d) {
    CoordinateAxis axis;
    axis.position = d;
    axis.extent = tensor.shape()[d];
    if (tensor.signature().axes) {
      const auto &named = (*tensor.signature().axes)[d];
      if (named.kind != AxisKind::Unknown || !named.label.empty()) {
        axis.semantic = named;
      }
    }
    model.axes.push_back(std::move(axis));
  }
  if (producer && semantics->status == SemanticStatus::Known) {
    for (std::size_t p = 0; p < semantics->pieces.size(); ++p) {
      const auto &piece = semantics->pieces[p];
      const auto written = std::find_if(piece.outputs.begin(), piece.outputs.end(),
                                        [&](const auto &out) { return out.port == outputPort; });
      if (written == piece.outputs.end()) {
        continue;
      }
      model.pieces.push_back(p);
      for (auto &axis : model.axes) {
        AxisDerivation derivation;
        derivation.piece = p;
        derivation.write = written->coordinates[axis.position];
        derivation.sources = sources(graph, *producer, piece, derivation.write, axis.extent);
        derivation.shapeReads = piece.shapeReads;
        derivation.structure =
            structure(graph, piece, derivation.write, axis.extent, derivation.sources);
        axis.derivations.push_back(std::move(derivation));
      }
    }
  }
  return model;
}
bool sameSource(const AxisSource &a, const AxisSource &b) {
  return a.tensor == b.tensor && a.inputPort == b.inputPort && a.access == b.access &&
         a.dimension == b.dimension && same(a.read, b.read);
}
} // namespace
std::string toString(AxisStructure kind) {
  switch (kind) {
  case AxisStructure::Signature:
    return "signature";
  case AxisStructure::Generated:
    return "generated";
  case AxisStructure::Singleton:
    return "singleton";
  case AxisStructure::Direct:
    return "direct";
  case AxisStructure::Split:
    return "split";
  case AxisStructure::Merged:
    return "merged";
  case AxisStructure::Reindexed:
    return "reindexed";
  case AxisStructure::Indirect:
    return "indirect";
  case AxisStructure::Metadata:
    return "shape_metadata";
  }
  throw Error("axis_model_invalid", "invalid structural axis kind");
}
std::string toString(AxisModelOrigin origin) {
  switch (origin) {
  case AxisModelOrigin::Signature:
    return "source_signature";
  case AxisModelOrigin::OperatorRelation:
    return "operator_index_relation";
  case AxisModelOrigin::Scalar:
    return "scalar_no_axes";
  }
  throw Error("axis_model_invalid", "invalid axis model origin");
}
std::map<TensorId, TensorAxisModel> buildAxisModels(const DAG &graph) {
  std::map<TensorId, TensorAxisModel> result;
  for (const auto &t : graph.tensors()) {
    result.emplace(t.id(), makeModel(graph, t, nullptr, 0, nullptr));
  }
  for (const auto &node : graph.nodes()) {
    const auto inputs = inputSignatures(graph, *node);
    const auto output = node->infer(inputs);
    const auto semantics = node->indexSemantics(inputs);
    semantics.validate(inputs, output);
    for (std::size_t p = 0; p < node->outputs().size(); ++p) {
      const auto &t = graph.tensor(node->outputs()[p]);
      result.at(t.id()) = makeModel(graph, t, node.get(), p, &semantics);
    }
  }
  return result;
}
void TensorAxisModel::validate(const DAG &graph) const {
  require(graphId == graph.id() && registryId == graph.registryId(), "axis_model_context",
          "axis model belongs to a different graph or registry");
  const auto &value = graph.tensor(tensor);
  const Operator *node = nullptr;
  std::size_t port = 0;
  for (const auto &candidate : graph.nodes()) {
    for (std::size_t p = 0; p < candidate->outputs().size(); ++p) {
      if (candidate->outputs()[p] == tensor) {
        node = candidate.get();
        port = p;
      }
    }
  }
  IndexSemantics semantics;
  if (node) {
    const auto in = inputSignatures(graph, *node);
    semantics = node->indexSemantics(in);
    semantics.validate(in, node->infer(in));
  }
  const auto expected = makeModel(graph, value, node, port, node ? &semantics : nullptr);
  require(status == expected.status && reason == expected.reason && origin == expected.origin &&
              producer == expected.producer && outputPort == expected.outputPort &&
              inputs == expected.inputs && pieces == expected.pieces &&
              semanticOrigin == expected.semanticOrigin && axes.size() == expected.axes.size(),
          "axis_model_invalid", "axis model status, origin, ports or rank disagree with graph");
  for (std::size_t d = 0; d < axes.size(); ++d) {
    const auto &a = axes[d];
    const auto &b = expected.axes[d];
    require(a.position == b.position && a.extent == b.extent && a.semantic == b.semantic &&
                a.derivations.size() == b.derivations.size(),
            "axis_model_invalid", "axis dimension or semantic annotation differs from graph");
    for (std::size_t i = 0; i < a.derivations.size(); ++i) {
      const auto &x = a.derivations[i];
      const auto &y = b.derivations[i];
      require(x.piece == y.piece && x.structure == y.structure && same(x.write, y.write) &&
                  x.sources.size() == y.sources.size() &&
                  x.shapeReads.size() == y.shapeReads.size(),
              "axis_model_invalid", "axis derivation differs from producer relation");
      for (std::size_t j = 0; j < x.sources.size(); ++j) {
        require(sameSource(x.sources[j], y.sources[j]), "axis_model_invalid",
                "axis source coordinate or tensor was changed");
      }
      for (std::size_t j = 0; j < x.shapeReads.size(); ++j) {
        require(x.shapeReads[j].port == y.shapeReads[j].port &&
                    same(x.shapeReads[j].dimension, y.shapeReads[j].dimension),
                "axis_model_invalid", "shape metadata provenance was changed");
      }
    }
  }
}
} // namespace opfusion
