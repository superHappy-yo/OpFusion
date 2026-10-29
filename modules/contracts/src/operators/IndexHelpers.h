#pragma once
#include "StaticChecks.h"

#include <algorithm>
#include <functional>
#include <utility>
namespace opfusion::operators::detail {
// 逻辑行主序线性坐标不是物理 stride；只描述值序，不能推导 Buffer 别名或零拷贝。
inline std::vector<IndexExpression> delinearize(std::size_t flatAxis, const Shape &shape) {
  std::vector<IndexExpression> coordinates(shape.size());
  if (elements(shape) == 0) {
    return coordinates;
  }
  std::int64_t stride = 1;
  for (std::size_t d = shape.size(); d > 0; --d) {
    coordinates[d - 1] = {{{flatAxis, 1, stride, shape[d - 1]}}, 0};
    stride = multiply(stride, shape[d - 1]);
  }
  return coordinates;
}
inline IndexSemantics reshapeIndex(const TensorSignature &input, const TensorSignature &output) {
  require(elements(input.shape) == elements(output.shape), "invalid_shape",
          "reshape element count mismatch");
  IndexPiece piece;
  piece.domain = {{"flat", 0, shapeProduct(output.shape), false}};
  // 一维值扩成唯一非单例轴时，该轴身份可保留；拆分/合并多个轴时不猜测身份。
  const auto nonSingleton =
      std::count_if(output.shape.begin(), output.shape.end(), [](auto d) { return d > 1; });
  const bool preservesVector = input.shape.size() == 1 && input.shape[0] > 1 && nonSingleton == 1;
  piece.inputs = {{0, delinearize(0, input.shape), FillValue::None, input.axes, preservesVector}};
  piece.outputs = {{0, delinearize(0, output.shape)}};
  return known(std::move(piece));
}
// 只用于已知的重排/切片算子；不据此给任意算子发明求值规则。
inline std::vector<std::optional<Shape>>
foldCopyIntegers(const Operator &op, const std::vector<TensorSignature> &inputs,
                 const std::vector<std::optional<Shape>> &values) {
  const auto outputs = op.infer(inputs);
  const auto semantics = op.indexSemantics(inputs);
  semantics.validate(inputs, outputs);
  require(values.size() == inputs.size(), "invalid_ports",
          "static input values differ from input ports");
  std::vector<std::optional<Shape>> result(outputs.size());
  for (const auto &piece : semantics.pieces) {
    require(piece.reduction == ReductionKind::None && piece.inputs.size() == 1 &&
                piece.outputs.size() == 1 && piece.shapeReads.empty(),
            "index_invalid", "copy fold requires a single read/write per piece");
    const auto port = piece.inputs[0].port;
    if (!values[port]) {
      return result;
    }
    require(inputs[port].dtype == DataType::Int64 || inputs[port].dtype == DataType::Int32 ||
                inputs[port].dtype == DataType::Int8 || inputs[port].dtype == DataType::Bool,
            "invalid_dtype", "integer folding requires integer data");
    require(values[port]->size() == elements(inputs[port].shape), "invalid_shape",
            "static data length mismatch");
  }
  for (std::size_t p = 0; p < outputs.size(); ++p) {
    result[p] = Shape(elements(outputs[p].shape));
  }
  auto flat = [](const std::vector<IndexExpression> &coordinates, const Shape &shape,
                 const Shape &iteration) {
    Bytes index = 0;
    for (std::size_t d = 0; d < shape.size(); ++d) {
      const auto v = evaluateIndex(coordinates[d], iteration);
      require(v >= 0 && v < shape[d], "index_out_of_bounds", "static copy access outside tensor");
      index =
          checkedAdd(checkedMultiply(index, static_cast<Bytes>(shape[d])), static_cast<Bytes>(v));
    }
    return index;
  };
  for (const auto &piece : semantics.pieces) {
    Shape iter(piece.domain.size());
    std::function<void(std::size_t)> walk = [&](std::size_t d) {
      if (d < iter.size()) {
        for (iter[d] = piece.domain[d].lower; iter[d] < piece.domain[d].upper; ++iter[d]) {
          walk(d + 1);
        }
        return;
      }
      const auto &read = piece.inputs[0];
      const auto &write = piece.outputs[0];
      result[write.port]->at(flat(write.coordinates, outputs[write.port].shape, iter)) =
          values[read.port]->at(flat(read.coordinates, inputs[read.port].shape, iter));
    };
    walk(0);
  }
  return result;
}
} // namespace opfusion::operators::detail
