// Gemm 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxGemmBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Gemm.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"alpha", "beta", "transA", "transB"});
  const auto ta = integer(s, "transA", 0), tb = integer(s, "transB", 0);
  require((ta == 0 || ta == 1) && (tb == 0 || tb == 1), "invalid_attribute",
          "invalid Gemm transpose flag");
  return std::make_unique<operators::Gemm>(
      id(s), s.inputs, s.outputs,
      operators::GemmAttrs{number(s, "alpha", 1), number(s, "beta", 1), ta == 1, tb == 1});
}

} // namespace
SourceBinding makeOnnxGemmBinding() {
  return binding(operators::Gemm::semanticIdentity(), "Gemm", create);
}
} // namespace opfusion::frontend::onnxGraph
