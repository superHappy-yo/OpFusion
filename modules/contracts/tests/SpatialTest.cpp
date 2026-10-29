// 以独立坐标循环验证 ND 推导和公开索引关系；只在测试中解释索引，不是执行后端。
#include "opfusion/common/Error.h"
#include "opfusion/operators/Conv.h"
#include "opfusion/operators/MatMul.h"
#include "opfusion/operators/MaxPool.h"
#include "opfusion/operators/Mul.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace opfusion;
using namespace opfusion::operators;
namespace {
int checks = 0;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) {
    throw std::runtime_error(message);
  }
}
template <class F> void rejects(F action) {
  ++checks;
  try {
    action();
  } catch (const Error &) {
    return;
  }
  throw std::runtime_error("expected domain rejection");
}
Shape coordinates(std::size_t flat, const Shape &shape) {
  Shape out(shape.size());
  for (std::size_t d = shape.size(); d > 0; --d) {
    out[d - 1] = flat % shape[d - 1];
    flat /= shape[d - 1];
  }
  return out;
}
std::size_t flat(const Shape &coord, const Shape &shape) {
  std::size_t n = 0;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    check(coord[d] >= 0 && coord[d] < shape[d], "coordinate outside tensor");
    n = n * shape[d] + coord[d];
  }
  return n;
}
Shape access(const std::vector<IndexExpression> &expr, const Shape &iter) {
  Shape out;
  for (const auto &e : expr) {
    out.push_back(evaluateIndex(e, iter));
  }
  return out;
}
std::vector<double> evaluate(const Operator &op, const std::vector<TensorSignature> &inputs,
                             const std::vector<std::vector<double>> &data) {
  const auto outputs = op.infer(inputs);
  auto sem = op.indexSemantics(inputs);
  sem.validate(inputs, outputs);
  check(outputs.size() == 1, "single test output");
  std::vector<double> result(elements(outputs[0].shape),
                             sem.pieces[0].reduction == ReductionKind::Maximum
                                 ? -std::numeric_limits<double>::infinity()
                                 : 0);
  for (const auto &piece : sem.pieces) {
    check(piece.reduction == ReductionKind::None || piece.reduction == ReductionKind::Sum ||
              piece.reduction == ReductionKind::Maximum,
          "spatial evaluator only accepts its tested reduction subset");
    Shape iter(piece.domain.size());
    std::function<void(std::size_t)> walk = [&](std::size_t d) {
      if (d < iter.size()) {
        for (iter[d] = piece.domain[d].lower; iter[d] < piece.domain[d].upper; ++iter[d]) {
          walk(d + 1);
        }
        return;
      }
      double value = piece.reduction == ReductionKind::Sum ? 1 : 0;
      for (const auto &input : piece.inputs) {
        const auto coord = access(input.coordinates, iter);
        bool outside = false;
        for (std::size_t k = 0; k < coord.size(); ++k) {
          outside |= coord[k] < 0 || coord[k] >= inputs[input.port].shape[k];
        }
        double v;
        if (outside) {
          check(input.outOfBounds != FillValue::None, "undeclared out of bounds");
          v = input.outOfBounds == FillValue::Zero ? 0 : -std::numeric_limits<double>::infinity();
        } else {
          v = data[input.port][flat(coord, inputs[input.port].shape)];
        }
        if (piece.reduction == ReductionKind::Sum) {
          value *= v;
        } else {
          value = v;
        }
      }
      const auto index = flat(access(piece.outputs[0].coordinates, iter), outputs[0].shape);
      if (piece.reduction == ReductionKind::Sum) {
        result[index] += value;
      } else if (piece.reduction == ReductionKind::Maximum) {
        result[index] = std::max(result[index], value);
      } else {
        result[index] = value;
      }
    };
    walk(0);
  }
  return result;
}
void spatial(std::size_t n) {
  ConvAttrs a;
  a.spatialDimensions = n;
  a.dataAxes = spatialDataOrder(n);
  a.weightAxes = spatialWeightOrder(n);
  a.groups = 2;
  a.strides = Shape(n, 1);
  a.strides[0] = 2;
  a.dilations = Shape(n, 1);
  a.dilations.back() = 2;
  a.pads = Shape(2 * n, 0);
  std::fill(a.pads.begin(), a.pads.begin() + n, 1);
  Shape xs{1, 4}, ws{6, 2};
  xs.insert(xs.end(), n, 5);
  ws.insert(ws.end(), n, 2);
  Shape expected = n == 1 ? Shape{1, 6, 2} : n == 2 ? Shape{1, 6, 3, 4} : Shape{1, 6, 3, 5, 4};
  Conv conv(1, {1, 2}, {3}, a);
  std::vector<TensorSignature> inputs{{xs, DataType::Float32}, {ws, DataType::Float32}};
  check(conv.infer(inputs)[0].shape == expected, "ND Conv output shape");
  std::vector<double> x(elements(xs)), w(elements(ws));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<int>(i % 13) - 7;
  }
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<int>(i % 5) - 2;
  }
  const auto actual = evaluate(conv, inputs, {x, w});
  Shape kernel(n, 2);
  for (std::size_t o = 0; o < actual.size(); ++o) {
    const auto out = coordinates(o, expected);
    double value = 0;
    for (std::int64_t c = 0; c < 2; ++c) {
      for (std::size_t k = 0; k < elements(kernel); ++k) {
        auto kc = coordinates(k, kernel);
        Shape xc{out[0], (out[1] / 3) * 2 + c}, wc{out[1], c};
        bool valid = true;
        for (std::size_t d = 0; d < n; ++d) {
          auto pos = out[d + 2] * a.strides[d] - 1 + kc[d] * a.dilations[d];
          xc.push_back(pos);
          wc.push_back(kc[d]);
          valid &= pos >= 0 && pos < 5;
        }
        if (valid) {
          value += x[flat(xc, xs)] * w[flat(wc, ws)];
        }
      }
    }
    check(actual[o] == value, "ND grouped Conv numerical mismatch");
  }
  MaxPoolAttrs p;
  p.spatialDimensions = n;
  p.dataAxes = spatialDataOrder(n);
  p.window = kernel;
  p.strides = a.strides;
  p.dilations = a.dilations;
  p.pads = a.pads;
  MaxPool pool(2, {1}, {2}, p);
  auto poolShape = expected;
  poolShape[1] = 4;
  check(pool.infer({inputs[0]})[0].shape == poolShape, "ND MaxPool output shape");
  for (auto &v : x) {
    v = -std::abs(v) - 1;
  }
  const auto pooled = evaluate(pool, {inputs[0]}, {x});
  for (std::size_t o = 0; o < pooled.size(); ++o) {
    const auto out = coordinates(o, poolShape);
    double value = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < elements(kernel); ++k) {
      const auto kc = coordinates(k, kernel);
      Shape coord{out[0], out[1]};
      bool valid = true;
      for (std::size_t d = 0; d < n; ++d) {
        const auto pos = out[d + 2] * p.strides[d] - 1 + kc[d] * p.dilations[d];
        coord.push_back(pos);
        valid &= pos >= 0 && pos < 5;
      }
      if (valid) {
        value = std::max(value, x[flat(coord, xs)]);
      }
    }
    check(pooled[o] == value, "ND MaxPool numerical mismatch");
  }
  auto invalid = a;
  invalid.strides.push_back(1);
  rejects([&] { Conv bad(1, {1, 2}, {3}, invalid); });
  invalid = a;
  invalid.pads.pop_back();
  rejects([&] { Conv bad(1, {1, 2}, {3}, invalid); });
  auto badInputs = inputs;
  badInputs[0].shape.push_back(1);
  rejects([&] { conv.infer(badInputs); });
  badInputs = inputs;
  badInputs[0].shape[1] = 3;
  rejects([&] { conv.infer(badInputs); });
  auto badPool = p;
  badPool.dilations.clear();
  rejects([&] { MaxPool bad(1, {1}, {2}, badPool); });
  auto semantics = conv.indexSemantics(inputs);
  semantics.pieces[0].inputs[1].coordinates[0].offset = 99;
  rejects([&] { semantics.validate(inputs, conv.infer(inputs)); });
  // 未分组卷积同样检查，不能仅靠分组实现通过测试。
  a.groups = 1;
  ws[1] = 4;
  Conv normal(1, {1, 2}, {3}, a);
  check(normal.infer({{xs, DataType::Float32}, {ws, DataType::Float32}})[0].shape == expected,
        "normal ND Conv");
  const auto ungrouped =
      evaluate(normal, {{xs, DataType::Float32}, {ws, DataType::Float32}},
               {std::vector<double>(elements(xs), 1), std::vector<double>(elements(ws), 1)});
  for (std::size_t o = 0; o < ungrouped.size(); ++o) {
    const auto out = coordinates(o, expected);
    double value = 0;
    for (std::size_t k = 0; k < elements(kernel); ++k) {
      const auto kc = coordinates(k, kernel);
      bool valid = true;
      for (std::size_t d = 0; d < n; ++d) {
        const auto pos = out[d + 2] * a.strides[d] - 1 + kc[d] * a.dilations[d];
        valid &= pos >= 0 && pos < 5;
      }
      if (valid) {
        value += 4;
      }
    }
    check(ungrouped[o] == value, "normal ND Conv numerical mismatch");
  }
  auto poolSemantics = pool.indexSemantics({inputs[0]});
  poolSemantics.pieces[0].inputs[0].outOfBounds = FillValue::None;
  rejects([&] { poolSemantics.validate({inputs[0]}, pool.infer({inputs[0]})); });
}
} // namespace
int main() {
  try {
    for (std::size_t n = 1; n <= 3; ++n) {
      spatial(n);
    }
    MaxPoolAttrs p;
    p.spatialDimensions = 1;
    p.dataAxes = AxisOrder::NCW;
    p.window = {3};
    p.strides = {2};
    p.dilations = {1};
    p.pads = {0, 0};
    p.ceilMode = true;
    MaxPool ceil(1, {1}, {2}, p);
    check(ceil.infer({{{1, 1, 6}, DataType::Float32}})[0].shape == Shape{1, 1, 3},
          "ceil pooling shape");
    check(evaluate(ceil, {{{1, 1, 6}, DataType::Float32}}, {{-6, -5, -4, -3, -2, -1}}) ==
              std::vector<double>{-4, -2, -1},
          "ceil pooling values");
    check(ceil.infer({{{1, 1, 2}, DataType::Float32}})[0].shape == Shape{1, 1, 1},
          "ceil first partial window");
    check(evaluate(ceil, {{{1, 1, 2}, DataType::Float32}}, {{-2, -1}}) == std::vector<double>{-1},
          "ceil partial window values");
    p.pads = {0, 2};
    p.window = {2};
    MaxPool trim(1, {1}, {2}, p);
    check(trim.infer({{{1, 1, 5}, DataType::Float32}})[0].shape == Shape{1, 1, 3},
          "ceil excludes padding-only window");
    MatMul mat(1, {1, 2}, {3});
    std::vector<TensorSignature> in{{{2, 2, 3}, DataType::Float32}, {{1, 3, 2}, DataType::Float32}};
    const auto result =
        evaluate(mat, in, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6}});
    check(result == std::vector<double>{22, 28, 49, 64, 76, 100, 103, 136},
          "batched MatMul values");
    check(mat.broadcasting(in).applied, "MatMul batch broadcast");
    Mul mul(1, {1, 2}, {3});
    check(
        !mul.broadcasting({{{1, 3, 4}, DataType::Float32}, {{1, 3, 4}, DataType::Float32}}).applied,
        "same-shape Mul does not broadcast");
    const auto broadcast =
        mul.broadcasting({{{2, 3, 4}, DataType::Float32}, {{3, 1}, DataType::Float32}});
    check(broadcast.applied && broadcast.inputs[1].axes == std::vector<std::size_t>{0, 2},
          "broadcast ports and axes");
    std::cout << checks << " spatial/index checks passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
