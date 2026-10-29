// Conv 的 ONNX opset 17 绑定；格式政策只在本适配器中处理。
#include "OnnxConvBinding.h"

#include "OnnxFields.h"
#include "opfusion/operators/Add.h"
#include "opfusion/operators/Conv.h"
#include "opfusion/operators/Reshape.h"

#include <algorithm>
#include <cmath>
#include <limits>
namespace opfusion::frontend::onnxGraph {
namespace {
std::unique_ptr<Operator> create(const SourceNode &s) {
  known(s, {"strides", "dilations", "pads", "group", "auto_pad", "kernel_shape"});
  require(s.inputs.size() == 2, "invalid_ports", "Conv bias requires normalization");
  const auto n = s.inputSignatures.at(0).shape.size();
  require(n >= 3 && n <= 5, "invalid_shape", "Conv rank must be 3..5");
  operators::ConvAttrs a;
  a.spatialDimensions = n - 2;
  a.strides = integers(s, "strides", Shape(n - 2, 1));
  a.dilations = integers(s, "dilations", Shape(n - 2, 1));
  a.pads = integers(s, "pads", Shape(2 * (n - 2), 0));
  a.groups = integer(s, "group", 1);
  a.dataAxes = spatialDataOrder(n - 2);
  a.weightAxes = spatialWeightOrder(n - 2);
  require(text(s, "auto_pad", "NOTSET") == "NOTSET", "unsupported_attribute",
          "Conv auto_pad is not implemented");
  require(s.inputSignatures.at(1).shape.size() == n, "invalid_shape", "Conv weight rank mismatch");
  Shape kernel(s.inputSignatures.at(1).shape.begin() + 2, s.inputSignatures.at(1).shape.end());
  require(integers(s, "kernel_shape", kernel) == kernel, "invalid_attribute",
          "Conv kernel_shape disagrees with weights");
  a.kernelShape = kernel;
  return std::make_unique<operators::Conv>(id(s), s.inputs, s.outputs, a);
}
SourceExpansion normalize(const SourceNode &s) {
  SourceExpansion result;
  if (s.inputs.size() == 2) {
    result.nodes.push_back(create(s));
    return result;
  }
  require(s.inputs.size() == 3 && s.outputs.size() == 1, "invalid_ports",
          "Conv expects 2 or 3 inputs");
  require(s.inputSignatures.size() == 3 && !s.inputSignatures[1].shape.empty() &&
              s.inputSignatures[2].shape == Shape{s.inputSignatures[1].shape[0]} &&
              s.inputSignatures[2].dtype == s.inputSignatures[0].dtype,
          "invalid_shape", "Conv bias must be a channel vector of matching dtype");
  auto primary = s;
  primary.inputs.resize(2);
  primary.inputSignatures.resize(2);
  primary.outputs = {s.nextTensor};
  auto conv = create(primary);
  const auto out = conv->infer(primary.inputSignatures).front();
  result.nodes.push_back(std::move(conv));
  result.tensors.emplace_back(s.nextTensor, s.name + "/unbiased", out.shape, out.dtype,
                              std::nullopt, ValueKind::OperatorResult);
  Shape biasShape(out.shape.size(), 1);
  biasShape[1] = out.shape[1];
  const auto biasId = s.nextTensor + 1;
  result.tensors.emplace_back(biasId, s.name + "/bias-broadcast", biasShape, out.dtype,
                              std::nullopt, ValueKind::OperatorResult);
  result.nodes.push_back(std::make_unique<operators::Reshape>(
      id(s) + 1, std::vector<TensorId>{s.inputs[2]}, std::vector<TensorId>{biasId},
      operators::ReshapeAttrs{biasShape, false}));
  result.nodes.push_back(std::make_unique<operators::Add>(
      id(s) + 2, std::vector<TensorId>{s.nextTensor, biasId}, s.outputs));
  result.notes.push_back("Conv bias normalized once as Conv + Reshape + Add");
  return result;
}
} // namespace
SourceBinding makeOnnxConvBinding() {
  return binding(operators::Conv::semanticIdentity(), "Conv", create, normalize);
}
} // namespace opfusion::frontend::onnxGraph
