// MaxPool 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxMaxPoolBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/MaxPool.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s,
        {"kernel_shape", "strides", "dilations", "pads", "ceil_mode", "storage_order", "auto_pad"});
  require(!s.inputSignatures.empty(), "invalid_ports", "MaxPool input missing");
  const auto rank = s.inputSignatures[0].shape.size();
  require(rank >= 3 && rank <= 5, "invalid_shape", "MaxPool rank must be 3..5");
  operators::MaxPoolAttrs a;
  const auto n = rank - 2;
  a.spatialDimensions = n;
  a.dataAxes = spatialDataOrder(n);
  a.window = integers(s, "kernel_shape");
  a.strides = integers(s, "strides", Shape(n, 1));
  a.dilations = integers(s, "dilations", Shape(n, 1));
  a.pads = integers(s, "pads", Shape(2 * n, 0));
  const auto ceil = integer(s, "ceil_mode", 0);
  require(ceil == 0 || ceil == 1, "invalid_attribute", "invalid ceil_mode");
  a.ceilMode = ceil == 1;
  require(integer(s, "storage_order", 0) == 0 && text(s, "auto_pad", "NOTSET") == "NOTSET",
          "unsupported_attribute", "unsupported MaxPool storage order or auto_pad");
  return std::make_unique<operators::MaxPool>(id(s), s.inputs, s.outputs, a);
}

} // namespace
SourceBinding makeOnnxMaxPoolBinding() {
  return binding(operators::MaxPool::semanticIdentity(), "MaxPool", create);
}
} // namespace opfusion::frontend::onnxGraph
