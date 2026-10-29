// 独立数值参考检验公开访问关系；测试解释器不是生产执行器。
#include "opfusion/common/Error.h"
#include "opfusion/graph/AxisModel.h"
#include "opfusion/graph/DAG.h"
#include "opfusion/operators/Constant.h"
#include "opfusion/operators/Flatten.h"
#include "opfusion/operators/Gather.h"
#include "opfusion/operators/Gemm.h"
#include "opfusion/operators/GlobalAveragePool.h"
#include "opfusion/operators/Reshape.h"
#include "opfusion/operators/ShapeOf.h"
#include "opfusion/operators/Softmax.h"
#include "opfusion/operators/Split.h"
#include "opfusion/operators/Transpose.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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
template <class F> void rejects(F f) {
  ++checks;
  try {
    f();
  } catch (const Error &) {
    return;
  }
  throw std::runtime_error("expected domain rejection");
}
std::size_t address(const std::vector<IndexExpression> &coordinates, const Shape &shape,
                    const Shape &iter, const std::vector<TensorSignature> &inputs,
                    const std::vector<std::optional<Shape>> &ints) {
  std::size_t result = 0;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    auto v = evaluateIndex(coordinates[d], iter, inputs, ints);
    check(v >= 0 && v < shape[d], "index outside tensor");
    result = result * shape[d] + v;
  }
  return result;
}
std::vector<std::vector<double>> interpret(const Operator &op,
                                           const std::vector<TensorSignature> &inputs,
                                           const std::vector<std::vector<double>> &values,
                                           const std::vector<std::optional<Shape>> &ints = {},
                                           const std::vector<std::vector<double>> &literals = {}) {
  const auto outputs = op.infer(inputs);
  const auto semantics = op.indexSemantics(inputs);
  semantics.validate(inputs, outputs);
  check(semantics.status == SemanticStatus::Known, "relationship must be known");
  std::vector<std::vector<double>> result;
  std::vector<std::vector<int>> writes;
  for (const auto &out : outputs) {
    result.emplace_back(elements(out.shape));
    writes.emplace_back(elements(out.shape), 0);
  }
  for (const auto &piece : semantics.pieces) {
    Shape iter(piece.domain.size());
    auto read = [&](const InputAccess &a) {
      return values.at(a.port).at(address(a.coordinates, inputs[a.port].shape, iter, inputs, ints));
    };
    std::function<void(std::size_t)> outer = [&](std::size_t d) {
      if (d < iter.size()) {
        if (piece.domain[d].reduction) {
          iter[d] = piece.domain[d].lower;
          outer(d + 1);
        } else {
          for (iter[d] = piece.domain[d].lower; iter[d] < piece.domain[d].upper; ++iter[d]) {
            outer(d + 1);
          }
        }
        return;
      }
      double value = 0;
      if (piece.reduction == ReductionKind::None) {
        if (!piece.shapeReads.empty()) {
          const auto &m = piece.shapeReads.front();
          value = inputs[m.port].shape.at(evaluateIndex(m.dimension, iter));
        } else if (!piece.inputs.empty()) {
          value = read(piece.inputs.back());
        }
      } else {
        std::vector<double> terms;
        std::function<void(std::size_t)> reduce = [&](std::size_t k) {
          if (k < iter.size()) {
            if (piece.domain[k].reduction) {
              for (iter[k] = piece.domain[k].lower; iter[k] < piece.domain[k].upper; ++iter[k]) {
                reduce(k + 1);
              }
            } else {
              reduce(k + 1);
            }
            return;
          }
          double product = 1;
          for (const auto &a : piece.inputs) {
            if (a.phase == AccessPhase::Iteration) {
              product *= read(a);
            }
          }
          terms.push_back(product);
        };
        reduce(0);
        if (piece.reduction == ReductionKind::NormalizedExponential) {
          const auto maximum = *std::max_element(terms.begin(), terms.end());
          double denominator = 0;
          for (double term : terms) {
            denominator += std::exp(term - maximum);
          }
          value = std::exp(read(piece.inputs[1]) - maximum) / denominator;
        } else {
          value = piece.reduction == ReductionKind::Maximum
                      ? -std::numeric_limits<double>::infinity()
                      : 0;
          for (double term : terms) {
            if (piece.reduction == ReductionKind::Maximum) {
              value = std::max(value, term);
            } else {
              value += term;
            }
          }
          if (piece.reduction == ReductionKind::Mean) {
            value /= terms.size();
          }
          value *= piece.reductionScale;
          for (const auto &a : piece.inputs) {
            if (a.phase == AccessPhase::AfterReduction) {
              value += piece.afterReductionScale * read(a);
            }
          }
        }
      }
      for (const auto &out : piece.outputs) {
        const auto at = address(out.coordinates, outputs[out.port].shape, iter, inputs, ints);
        if (piece.inputs.empty() && piece.shapeReads.empty()) {
          value = literals.at(out.port).at(at);
        }
        result[out.port][at] = value;
        ++writes[out.port][at];
      }
    };
    outer(0);
  }
  for (const auto &count : writes) {
    for (int n : count) {
      check(n == 1, "output not covered exactly once");
    }
  }
  return result;
}
void close(const std::vector<double> &actual, const std::vector<double> &expected) {
  check(actual.size() == expected.size(), "size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    check(std::abs(actual[i] - expected[i]) < 1e-12, "numeric relationship mismatch");
  }
}
TensorSignature fp(Shape shape) {
  return {std::move(shape), DataType::Float32};
}
void reshapeCases() {
  for (const auto &shape : std::vector<Shape>{{2, 3}, {1, 6, 1}, {3, 2}, {6}, {}}) {
    const auto size = elements(shape);
    Reshape op(1, {1}, {2}, ReshapeAttrs{{static_cast<std::int64_t>(size)}, false});
    std::vector<double> values(size);
    std::iota(values.begin(), values.end(), 1);
    close(interpret(op, {fp(shape)}, {values})[0], values);
  }
  Reshape n(1, {1}, {2}, ReshapeAttrs{{2, -1, 2}, false});
  close(interpret(n, {fp({3, 4})}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}})[0],
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Reshape empty(1, {1}, {2}, ReshapeAttrs{{0, 3}, true});
  check(interpret(empty, {fp({0, 3})}, {{}})[0].empty(), "empty reshape");
  Flatten flatten(1, {1}, {2}, FlattenAttrs{-1});
  close(interpret(flatten, {fp({2, 2, 2})}, {{1, 2, 3, 4, 5, 6, 7, 8}})[0],
        {1, 2, 3, 4, 5, 6, 7, 8});
  auto sem = n.indexSemantics({fp({3, 4})});
  sem.pieces[0].inputs[0].coordinates[0].terms[0].modulus = 0;
  rejects([&] { sem.validate({fp({3, 4})}, n.infer({fp({3, 4})})); });
  rejects([] { evaluateIndex({{{0, std::numeric_limits<std::int64_t>::max(), 1}}, 0}, {2}); });
}
void permutationAndSplit() {
  Transpose transpose(1, {1}, {2}, TransposeAttrs{{1, 0}});
  close(interpret(transpose, {fp({2, 3})}, {{1, 2, 3, 4, 5, 6}})[0], {1, 4, 2, 5, 3, 6});
  Transpose reverse(1, {1}, {2});
  close(interpret(reverse, {fp({2, 3})}, {{1, 2, 3, 4, 5, 6}})[0], {1, 4, 2, 5, 3, 6});
  const auto folded =
      transpose.evaluateIntegers({{{2, 3}, DataType::Int64}}, {Shape{1, 2, 3, 4, 5, 6}});
  check(folded[0] == Shape{1, 4, 2, 5, 3, 6}, "static Transpose fold");
  Split split(1, {1}, {2, 3}, SplitAttrs{-1, {1, 2}});
  const auto pieces = interpret(split, {fp({2, 3})}, {{1, 2, 3, 4, 5, 6}});
  close(pieces[0], {1, 4});
  close(pieces[1], {2, 3, 5, 6});
  Split equal(1, {1}, {2, 3}, SplitAttrs{0, {}});
  const auto even = interpret(equal, {fp({4})}, {{1, 2, 3, 4}});
  close(even[0], {1, 2});
  close(even[1], {3, 4});
  Split zero(1, {1}, {2, 3}, SplitAttrs{0, {0, 3}});
  const auto z = interpret(zero, {fp({3})}, {{1, 2, 3}});
  check(z[0].empty(), "empty split output");
  close(z[1], {1, 2, 3});
  const auto foldedSplit =
      split.evaluateIntegers({{{2, 3}, DataType::Int64}}, {Shape{1, 2, 3, 4, 5, 6}});
  check(foldedSplit[0] == Shape{1, 4} && foldedSplit[1] == Shape{2, 3, 5, 6}, "static Split fold");
}
void gatherCases() {
  Gather gather(1, {1, 2}, {3}, GatherAttrs{1});
  std::vector<TensorSignature> in{fp({2, 3}), {{2}, DataType::Int64}};
  std::vector<std::optional<Shape>> ints{std::nullopt, Shape{-1, 0}};
  close(interpret(gather, in, {{1, 2, 3, 4, 5, 6}, {-1, 0}}, ints)[0], {3, 1, 6, 4});
  Gather scalar(1, {1, 2}, {3}, GatherAttrs{-1});
  close(interpret(scalar, {fp({2, 3}), {{}, DataType::Int64}}, {{1, 2, 3, 4, 5, 6}, {1}},
                  {std::nullopt, Shape{1}})[0],
        {2, 5});
  const auto multi = interpret(gather, {fp({2, 3, 2}), {{2, 2}, DataType::Int64}},
                               {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 0, 1, -1}},
                               {std::nullopt, Shape{2, 0, 1, -1}});
  close(multi[0], {5, 6, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 11, 12});
  const auto sem = gather.indexSemantics(in);
  const auto expr = sem.pieces[0].inputs[1].coordinates[1];
  rejects([&] { evaluateIndex(expr, {0, 0}, in, {std::nullopt, Shape{3, 0}}); });
  rejects([&] { evaluateIndex(expr, {0, 0}, in, {std::nullopt, Shape{-4, 0}}); });
  auto bad = sem;
  bad.pieces[0].inputs[1].coordinates[1].lookups[0].port = 9;
  rejects([&] { bad.validate(in, gather.infer(in)); });
  in[1].dtype = DataType::Float32;
  rejects([&] { sem.validate(in, gather.infer({fp({2, 3}), {{2}, DataType::Int64}})); });
}
void reductionCases() {
  Softmax softmax(1, {1}, {2}, SoftmaxAttrs{-1});
  std::vector<double> data{1000, 1001, 999, -1000, -999, -1001};
  const auto actual = interpret(softmax, {fp({2, 3})}, {data})[0];
  std::vector<double> expected;
  for (std::size_t row = 0; row < 2; ++row) {
    double denominator = 0;
    const auto max = *std::max_element(data.begin() + row * 3, data.begin() + (row + 1) * 3);
    for (std::size_t j = 0; j < 3; ++j) {
      denominator += std::exp(data[row * 3 + j] - max);
    }
    for (std::size_t j = 0; j < 3; ++j) {
      expected.push_back(std::exp(data[row * 3 + j] - max) / denominator);
    }
  }
  close(actual, expected);
  auto sem = softmax.indexSemantics({fp({2, 3})});
  sem.pieces[0].domain.back().upper = 2;
  rejects([&] { sem.validate({fp({2, 3})}, softmax.infer({fp({2, 3})})); });
  sem = softmax.indexSemantics({fp({2, 3})});
  sem.pieces[0].inputs[1].coordinates[0] = {{{2, 1, 1}}, 0};
  rejects([&] { sem.validate({fp({2, 3})}, softmax.infer({fp({2, 3})})); });
  Softmax column(1, {1}, {2}, SoftmaxAttrs{0});
  const auto columnResult = interpret(column, {fp({2, 2})}, {{0, 1, 2, 3}})[0];
  const auto denominator = 1 + std::exp(2.0);
  close(columnResult, {1 / denominator, 1 / denominator, std::exp(2.0) / denominator,
                       std::exp(2.0) / denominator});
  GlobalAveragePool avg(1, {1}, {2});
  close(interpret(avg, {fp({1, 2, 2, 2})}, {{1, 2, 3, 4, 5, 6, 7, 8}})[0], {2.5, 6.5});
  Gemm gemm(1, {1, 2, 3}, {4}, GemmAttrs{2, 3, false, true});
  close(interpret(gemm, {fp({2, 3}), fp({2, 3}), fp({2})},
                  {{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {10, 20}})[0],
        {58, 124, 94, 214});
  check(gemm.broadcasting({fp({2, 3}), fp({2, 3}), fp({2})}).applied, "Gemm bias broadcasting");
  close(interpret(avg, {fp({1, 1, 2, 2, 2})}, {{1, 2, 3, 4, 5, 6, 7, 8}})[0], {4.5});
  Gemm transpose(1, {1, 2}, {3}, GemmAttrs{1, 1, true, false});
  close(interpret(transpose, {fp({3, 2}), fp({3, 2})}, {{1, 4, 2, 5, 3, 6}, {1, 4, 2, 5, 3, 6}})[0],
        {14, 32, 32, 77});
  Gemm zero(1, {1, 2, 3}, {4}, GemmAttrs{2, 3, false, false});
  close(interpret(zero, {fp({2, 0}), fp({0, 2}), fp({2})}, {{}, {}, {10, 20}})[0],
        {30, 60, 30, 60});
}
void vectorAxisPropagation() {
  auto graph = [](bool declaredConflict) {
    std::vector<Tensor> tensors;
    tensors.emplace_back(1, "bias", Shape{4}, DataType::Float32,
                         declaredConflict
                             ? std::optional<LogicalAxes>(LogicalAxes{{AxisKind::Row, {}}})
                             : std::nullopt);
    tensors.emplace_back(2, "broadcast-bias", Shape{1, 4, 1, 1}, DataType::Float32,
                         logicalAxes(AxisOrder::NCHW));
    std::vector<std::unique_ptr<Operator>> nodes;
    nodes.push_back(std::make_unique<Reshape>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2},
                                              ReshapeAttrs{{1, 4, 1, 1}, false}));
    return std::make_unique<DAG>("vector-axis", "vector-axis", std::move(tensors),
                                 std::move(nodes));
  };
  const auto derived = graph(false);
  check(derived->tensor(1).signature().axes ==
            std::optional<LogicalAxes>(LogicalAxes{{AxisKind::Channel, {}}}),
        "vector broadcast must inherit channel semantics");
  rejects([&] { graph(true); });
  std::vector<Tensor> tensors;
  tensors.emplace_back(1, "channel-vector", Shape{6}, DataType::Float32,
                       LogicalAxes{{AxisKind::Channel, {}}});
  tensors.emplace_back(2, "split-axis", Shape{2, 3}, DataType::Float32);
  std::vector<std::unique_ptr<Operator>> nodes;
  nodes.push_back(std::make_unique<Reshape>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2},
                                            ReshapeAttrs{{2, 3}, false}));
  DAG split("split", "split", std::move(tensors), std::move(nodes));
  check(!split.tensor(2).signature().axes, "split axis cannot inherit a guessed channel label");
}

class UnmodeledOperator final : public Operator {
public:
  UnmodeledOperator() : Operator(1, {"test", "unmodeled"}, {1}, {2}) {}
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override {
    return {inputs.at(0)};
  }
  Traits traits() const override { return {}; }
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &) const override {
    return {SemanticStatus::Unknown, "test provider has no index relation", {}};
  }

protected:
  void validateAttributes() const override {}
};
void unknownStructuralAxes() {
  std::vector<Tensor> tensors;
  tensors.emplace_back(1, "input", Shape{2, 3}, DataType::Float32);
  tensors.emplace_back(2, "output", Shape{2, 3}, DataType::Float32);
  std::vector<std::unique_ptr<Operator>> nodes;
  nodes.push_back(std::make_unique<UnmodeledOperator>());
  DAG graph("unknown", "unknown", std::move(tensors), std::move(nodes));
  const auto models = buildAxisModels(graph);
  models.at(2).validate(graph);
  check(models.at(2).status == SemanticStatus::Unknown && !models.at(2).reason.empty(),
        "static dimensions must not conceal missing provider relation");
  auto forged = models.at(2);
  forged.status = SemanticStatus::Known;
  forged.reason.clear();
  rejects([&] { forged.validate(graph); });
}
void structuralAxes() {
  auto reshapeGraph = [](Shape input, Shape output) {
    std::vector<Tensor> tensors;
    tensors.emplace_back(1, "input", input, DataType::Float32);
    tensors.emplace_back(2, "output", output, DataType::Float32);
    std::vector<std::unique_ptr<Operator>> nodes;
    nodes.push_back(std::make_unique<Reshape>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2},
                                              ReshapeAttrs{output, false}));
    return std::make_unique<DAG>("structural", "structural", std::move(tensors), std::move(nodes));
  };
  const auto graph = reshapeGraph({1, 256, 20, 20}, {1, 2, 128, 400});
  const auto models = buildAxisModels(*graph);
  for (const auto &item : models) {
    item.second.validate(*graph);
  }
  const auto &model = models.at(2);
  check(model.status == SemanticStatus::Known, "reshape coordinate model must be complete");
  check(model.axes[0].derivations[0].structure == AxisStructure::Singleton, "batch singleton");
  for (auto d : {1, 2}) {
    const auto &axis = model.axes[d];
    check(!axis.semantic, "split does not invent head/channel names");
    check(axis.derivations[0].structure == AxisStructure::Split, "channel split");
    check(axis.derivations[0].sources.size() == 1 && axis.derivations[0].sources[0].dimension == 1,
          "split must reference original channel");
  }
  const auto &merged = model.axes[3].derivations[0];
  check(merged.structure == AxisStructure::Merged && merged.sources.size() == 2 &&
            merged.sources[0].dimension == 2 && merged.sources[1].dimension == 3,
        "spatial merge must reference height and width");
  // Independently enumerate flattened coordinates and check the published split/merge expressions.
  const std::vector<TensorSignature> inputs{fp({1, 256, 20, 20})};
  for (std::int64_t flat : {0, 19, 20, 399, 400, 51199, 51200, 102399}) {
    const Shape expected{0, flat / 51200, (flat / 400) % 128, flat % 400};
    for (std::size_t d = 0; d < expected.size(); ++d) {
      check(evaluateIndex(model.axes[d].derivations[0].write, {flat}, inputs, {}) == expected[d],
            "structural coordinate differs from independent row-major reference");
    }
  }
  auto bad = model;
  bad.axes[1].extent = 3;
  rejects([&] { bad.validate(*graph); });
  bad = model;
  bad.axes[1].derivations[0].sources[0].tensor = 2;
  rejects([&] { bad.validate(*graph); });
  bad = model;
  bad.axes[1].derivations[0].write.offset = 1;
  rejects([&] { bad.validate(*graph); });
  bad = model;
  bad.axes[1].derivations[0].structure = AxisStructure::Direct;
  rejects([&] { bad.validate(*graph); });
  bad = model;
  bad.pieces = {9};
  rejects([&] { bad.validate(*graph); });
  bad = model;
  bad.registryId += "other";
  rejects([&] { bad.validate(*graph); });
  const auto mixed = reshapeGraph({6, 10}, {4, 15});
  const auto mixedModels = buildAxisModels(*mixed);
  check(mixedModels.at(2).axes[0].derivations[0].structure == AxisStructure::Reindexed,
        "crossing radix boundaries is not a pure split");
  const auto scalar = reshapeGraph({1}, {});
  const auto scalarModels = buildAxisModels(*scalar);
  check(scalarModels.at(2).origin == AxisModelOrigin::Scalar && scalarModels.at(2).axes.empty() &&
            scalarModels.at(2).status == SemanticStatus::Known,
        "scalar has no missing axes");
}

void metadataCases() {
  ShapeOf shape(1, {1}, {2}, ShapeOfAttrs{-2, std::nullopt});
  close(interpret(shape, {fp({2, 3, 4})}, {{}})[0], {3, 4});
  ShapeOf scalar(1, {1}, {2});
  check(interpret(scalar, {fp({})}, {{}})[0].empty(), "scalar metadata output");
  ConstantAttrs attrs;
  attrs.signature = {{3}, DataType::Int64};
  attrs.integers = Shape{5, 6, 7};
  Constant constant(1, {}, {2}, attrs);
  close(interpret(constant, {}, {}, {}, {{5, 6, 7}})[0], {5, 6, 7});
  auto constantSemantics = constant.indexSemantics({});
  check(constantSemantics.pieces[0].inputs.empty(), "constant cannot read model inputs");
  auto bad = shape.indexSemantics({fp({2, 3, 4})});
  bad.pieces[0].shapeReads[0].dimension.offset = 3;
  rejects([&] { bad.validate({fp({2, 3, 4})}, shape.infer({fp({2, 3, 4})})); });
}
} // namespace
int main() {
  try {
    reshapeCases();
    permutationAndSplit();
    gatherCases();
    reductionCases();
    metadataCases();
    vectorAxisPropagation();
    structuralAxes();
    unknownStructuralAxes();
    std::cout << checks << " complete index modeling checks passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
