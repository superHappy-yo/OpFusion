#include "SemanticChecks.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace opfusion::operators::detail {

// 共用的纯签名/索引构造只接收显式借用参数；不读取 Registry、来源字段或输出声明。
SemanticIdentity identity(const char *name) {
  return {"opfusion", name};
}

void ports(const Operator &op, std::size_t count) {
  require(op.inputs().size() == count && op.outputs().size() == 1, "invalid_ports",
          op.semantic().key() + ": wrong input/output count");
}

void signatures(const Operator &op, const std::vector<TensorSignature> &inputs, std::size_t count,
                bool floatsOnly) {
  op.validate(inputs);
  require(inputs.size() == count, "invalid_ports", "wrong inference input count");
  for (const auto &input : inputs) {
    const bool floating = input.dtype == DataType::Float16 || input.dtype == DataType::Float32 ||
                          input.dtype == DataType::Float64;
    require(!floatsOnly || floating, "unsupported_dtype",
            "this semantic requires floating-point inputs");
    require(input.dtype == inputs.front().dtype, "invalid_dtype",
            "input dtypes must match; no implicit conversion");
  }
}

std::int64_t add(std::int64_t a, std::int64_t b) {
  require(a >= 0 && b >= 0 && a <= std::numeric_limits<std::int64_t>::max() - b, "shape_overflow",
          "dimension addition overflow");
  return a + b;
}

std::int64_t multiply(std::int64_t a, std::int64_t b) {
  require(a >= 0 && b >= 0 && (b == 0 || a <= std::numeric_limits<std::int64_t>::max() / b),
          "shape_overflow", "dimension multiplication overflow");
  return a * b;
}

void padding(const std::array<std::int64_t, 4> &values) {
  for (auto value : values) {
    require(value >= 0, "invalid_attribute", "padding must be nonnegative");
  }
}
void positive(const Shape &values) {
  require(!values.empty(), "invalid_attribute", "spatial arrays cannot be empty");
  for (auto value : values) {
    require(value > 0, "invalid_attribute", "window/stride/dilation must be positive");
  }
}
void padding(const Shape &values) {
  for (auto value : values) {
    require(value >= 0, "invalid_attribute", "padding must be nonnegative");
  }
}

std::int64_t spatial(std::int64_t input, std::int64_t kernel, std::int64_t dilation,
                     std::int64_t before, std::int64_t after, std::int64_t stride) {
  require(input > 0 && kernel > 0, "invalid_shape", "spatial dimensions must be positive");
  const auto effective = add(multiply(kernel - 1, dilation), 1);
  const auto padded = add(add(input, before), after);
  require(padded >= effective, "invalid_shape", "effective window exceeds padded input");
  return add((padded - effective) / stride, 1);
}

void rank(const TensorSignature &input, std::size_t expected) {
  require(input.shape.size() == expected, "invalid_shape", "unexpected input rank");
}

LogicalAxes nchw() {
  return logicalAxes(AxisOrder::NCHW);
}
LogicalAxes oihw() {
  return logicalAxes(AxisOrder::OIHW);
}
LogicalAxes matrixAxes() {
  return {{AxisKind::Row, {}}, {AxisKind::Column, {}}};
}

std::vector<TensorSignature> binary(const Operator &op,
                                    const std::vector<TensorSignature> &inputs) {
  signatures(op, inputs, 2);
  require(inputs[0].dtype != DataType::Bool, "unsupported_dtype",
          "arithmetic requires numeric data");
  const auto &lhs = inputs[0].shape;
  const auto &rhs = inputs[1].shape;
  Shape output(std::max(lhs.size(), rhs.size()), 1);
  std::optional<LogicalAxes> axes;
  for (std::size_t offset = 0; offset < output.size(); ++offset) {
    const auto a = offset < lhs.size() ? lhs[lhs.size() - 1 - offset] : 1;
    const auto b = offset < rhs.size() ? rhs[rhs.size() - 1 - offset] : 1;
    require(a == b || a == 1 || b == 1, "invalid_shape", "incompatible broadcast dimensions");
    // 不能使用 max(a,b)：已知空维 0 与广播维 1 的结果仍应为 0。
    output[output.size() - 1 - offset] = a == 1 ? b : a;
  }
  for (const auto &input : inputs) {
    if (input.axes) {
      LogicalAxes aligned(output.size());
      std::copy(input.axes->begin(), input.axes->end(), aligned.end() - input.axes->size());
      axes = mergeAxes(axes, aligned);
    }
  }
  return {{std::move(output), inputs[0].dtype, axes}};
}

std::vector<TensorSignature> unary(const Operator &op, const std::vector<TensorSignature> &inputs,
                                   bool floatsOnly) {
  signatures(op, inputs, 1, floatsOnly);
  return {inputs[0]};
}

IndexExpression index(std::size_t axis, std::int64_t coefficient, std::int64_t divisor,
                      std::int64_t offset) {
  return {{{axis, coefficient, divisor}}, offset};
}

std::vector<IterationAxis> iteration(const Shape &shape) {
  std::vector<IterationAxis> result;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result.push_back({"i" + std::to_string(i), 0, shape[i], false});
  }
  return result;
}

std::vector<IndexExpression> identityCoordinates(std::size_t rank) {
  std::vector<IndexExpression> result;
  for (std::size_t i = 0; i < rank; ++i) {
    result.push_back(index(i));
  }
  return result;
}

IndexSemantics known(IndexPiece piece) {
  return {SemanticStatus::Known, {}, {std::move(piece)}};
}

IndexSemantics pointwiseIndex(const std::vector<TensorSignature> &inputs,
                              const TensorSignature &output) {
  IndexPiece piece;
  piece.domain = iteration(output.shape);
  piece.outputs.push_back({0, identityCoordinates(output.shape.size())});
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    InputAccess access;
    access.port = port;
    access.preservesAxes = true;
    const auto offset = output.shape.size() - inputs[port].shape.size();
    if (output.axes) {
      access.axes = LogicalAxes(output.axes->begin() + offset, output.axes->end());
    }
    for (std::size_t axis = 0; axis < inputs[port].shape.size(); ++axis) {
      access.coordinates.push_back(inputs[port].shape[axis] == 1 ? IndexExpression{}
                                                                 : index(offset + axis));
    }
    // 单输入恒等访问仍保留单例维迭代器，避免丢失该维的轴来源。
    if (inputs.size() == 1) {
      access.coordinates = identityCoordinates(output.shape.size());
    }
    piece.inputs.push_back(std::move(access));
  }
  return known(std::move(piece));
}

} // namespace opfusion::operators::detail
