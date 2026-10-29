// 静态签名与属性校验，不读取声明输出，也不冒充数值 kernel。
#include "opfusion/operators/Gemm.h"

#include "SemanticChecks.h"
#include "StaticChecks.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <utility>
namespace opfusion::operators {
using namespace detail;
Gemm::Gemm(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
           GemmAttrs attrs, NumericalRequirements requirements)
    : Operator(id, semanticIdentity(), std::move(inputs), std::move(outputs),
               std::move(requirements)),
      mAttrs(std::move(attrs)) {
  validateAttributes();
}
SemanticIdentity Gemm::semanticIdentity() {
  return identity("Gemm");
}
void Gemm::validateAttributes() const {
  variablePorts(*this, 2, 3);
  require(std::isfinite(mAttrs.alpha) && std::isfinite(mAttrs.beta), "invalid_attribute",
          "Gemm scalar is not finite");
}
std::vector<TensorSignature> Gemm::infer(const std::vector<TensorSignature> &inputs) const {
  signatures(*this, inputs, inputs.size(), true);
  rank(inputs[0], 2);
  rank(inputs[1], 2);
  auto a = inputs[0].shape, b = inputs[1].shape;
  if (mAttrs.transposeA) {
    std::swap(a[0], a[1]);
  }
  if (mAttrs.transposeB) {
    std::swap(b[0], b[1]);
  }
  require(a[1] == b[0], "invalid_shape", "Gemm contracting dimensions mismatch");
  Shape shape{a[0], b[1]};
  if (inputs.size() == 3) {
    const auto &c = inputs[2].shape;
    require(c.size() <= 2, "invalid_shape", "Gemm bias rank invalid");
    for (std::size_t i = 0; i < c.size(); ++i) {
      require(c[c.size() - 1 - i] == 1 || c[c.size() - 1 - i] == shape[1 - i], "invalid_shape",
              "Gemm bias broadcast invalid");
    }
  }
  return {{shape, inputs[0].dtype}};
}
Traits Gemm::traits() const {
  return {true, false, true, true};
}
std::vector<Attribute> Gemm::attributes(const std::vector<TensorSignature> &inputs) const {
  (void)inputs;
  return {{"alpha", mAttrs.alpha},
          {"beta", mAttrs.beta},
          {"transA", mAttrs.transposeA},
          {"transB", mAttrs.transposeB}};
}
IndexSemantics Gemm::indexSemantics(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  const auto k = inputs[0].shape[mAttrs.transposeA ? 0 : 1];
  piece.domain.push_back({"k", 0, k, true});
  piece.reduction = ReductionKind::Sum;
  piece.reductionScale = mAttrs.alpha;
  piece.afterReductionScale = mAttrs.beta;
  std::vector<IndexExpression> a = mAttrs.transposeA
                                       ? std::vector<IndexExpression>{index(2), index(0)}
                                       : std::vector<IndexExpression>{index(0), index(2)};
  std::vector<IndexExpression> b = mAttrs.transposeB
                                       ? std::vector<IndexExpression>{index(1), index(2)}
                                       : std::vector<IndexExpression>{index(2), index(1)};
  piece.inputs = {{0, a, FillValue::None, inputs[0].axes, false},
                  {1, b, FillValue::None, inputs[1].axes, false}};
  if (inputs.size() == 3) {
    std::vector<IndexExpression> c;
    const auto offset = 2 - inputs[2].shape.size();
    for (std::size_t d = 0; d < inputs[2].shape.size(); ++d) {
      c.push_back(inputs[2].shape[d] == 1 ? IndexExpression{} : index(d + offset));
    }
    piece.inputs.push_back(
        {2, c, FillValue::None, inputs[2].axes, false, AccessPhase::AfterReduction});
  }
  piece.outputs = {{0, identityCoordinates(2)}};
  return known(std::move(piece));
}
BroadcastInfo Gemm::broadcasting(const std::vector<TensorSignature> &inputs) const {
  const auto output = infer(inputs).front();
  BroadcastInfo result;
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    PortBroadcast entry{port, {}};
    if (port == 2) {
      for (std::size_t d = 0; d < 2; ++d) {
        const auto offset = 2 - inputs[port].shape.size();
        if (d < offset || inputs[port].shape[d - offset] != output.shape[d]) {
          entry.axes.push_back(d);
        }
      }
    }
    result.applied = result.applied || !entry.axes.empty();
    result.inputs.push_back(std::move(entry));
  }
  return result;
}
} // namespace opfusion::operators
