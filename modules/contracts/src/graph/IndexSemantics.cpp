#include "opfusion/graph/IndexSemantics.h"

#include "opfusion/common/Error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
namespace opfusion {
namespace {
std::int64_t sum(std::int64_t a, std::int64_t b) {
  std::int64_t result = 0;
  require(!__builtin_add_overflow(a, b, &result), "index_overflow",
          "index expression addition overflow");
  return result;
}
std::int64_t scale(std::int64_t coefficient, std::int64_t value) {
  require(value >= 0, "index_invalid", "negative iterator value");
  std::int64_t result = 0;
  require(!__builtin_mul_overflow(coefficient, value, &result), "index_overflow",
          "index expression multiplication overflow");
  return result;
}
// 对有界迭代域给出取模的精确端点/跨周期包络，不把负系数的相关项当成独立证明。
std::pair<std::int64_t, std::int64_t> termRange(const IndexTerm &t, const IterationAxis &axis) {
  auto low = axis.lower / t.divisor, high = (axis.upper - 1) / t.divisor;
  if (t.modulus) {
    const auto m = *t.modulus;
    if (low / m == high / m) {
      low %= m;
      high %= m;
    } else {
      low = 0;
      high = m - 1;
    }
  }
  const auto a = scale(t.coefficient, low), b = scale(t.coefficient, high);
  return {std::min(a, b), std::max(a, b)};
}
void coordinates(const std::vector<IndexExpression> &, const TensorSignature &, const IndexPiece &,
                 const std::vector<TensorSignature> &, bool, bool, bool, unsigned);
std::pair<std::int64_t, std::int64_t> bounds(const IndexExpression &expression,
                                             const IndexPiece &piece,
                                             const std::vector<TensorSignature> &inputs,
                                             bool output, bool empty, unsigned depth) {
  require(depth <= 16, "index_invalid", "indirect index nesting exceeds supported depth");
  auto low = expression.offset, high = expression.offset;
  std::set<std::size_t> seen;
  for (const auto &t : expression.terms) {
    require(t.axis < piece.domain.size() && t.divisor > 0 && t.coefficient != 0 &&
                seen.insert(t.axis).second && (!t.modulus || *t.modulus > 0),
            "index_invalid", "invalid/repeated iterator term or modulus");
    const auto &axis = piece.domain[t.axis];
    require(!output || !axis.reduction, "index_invalid", "output depends on reduction");
    if (empty) {
      continue;
    }
    const auto range = termRange(t, axis);
    low = sum(low, range.first);
    high = sum(high, range.second);
  }
  require(!output || expression.lookups.empty(), "index_invalid",
          "indirect output writes are unsupported");
  for (const auto &lookup : expression.lookups) {
    require(lookup.port < inputs.size() && lookup.extent >= 0, "index_invalid",
            "invalid indirect source or extent");
    const auto &source = inputs[lookup.port];
    require(source.dtype == DataType::Int64 || source.dtype == DataType::Int32, "index_invalid",
            "indirect source must contain integer indices");
    coordinates(lookup.coordinates, source, piece, inputs, false, false, empty, depth + 1);
    if (empty) {
      continue;
    }
    require(lookup.extent > 0, "index_out_of_bounds",
            "nonempty indirect read has empty target axis");
    high = sum(high, lookup.extent - 1);
  }
  return {low, high};
}
void coordinates(const std::vector<IndexExpression> &expressions, const TensorSignature &tensor,
                 const IndexPiece &piece, const std::vector<TensorSignature> &inputs, bool padded,
                 bool output, bool empty, unsigned depth) {
  require(expressions.size() == tensor.shape.size(), "index_invalid", "access rank mismatch");
  for (std::size_t d = 0; d < expressions.size(); ++d) {
    const auto range = bounds(expressions[d], piece, inputs, output, empty, depth);
    if (!empty && !padded) {
      require(range.first >= 0 && range.second < tensor.shape[d], "index_out_of_bounds",
              "access relation exceeds logical tensor bounds");
    }
  }
}
std::int64_t evaluate(const IndexExpression &e, const Shape &iteration,
                      const std::vector<TensorSignature> &inputs,
                      const std::vector<std::optional<Shape>> &values, unsigned depth) {
  require(depth <= 16, "index_invalid", "indirect index nesting exceeds supported depth");
  auto result = e.offset;
  for (const auto &t : e.terms) {
    require(t.axis < iteration.size() && iteration[t.axis] >= 0 && t.divisor > 0 &&
                t.coefficient != 0 && (!t.modulus || *t.modulus > 0),
            "index_invalid", "invalid index evaluation term");
    auto v = iteration[t.axis] / t.divisor;
    if (t.modulus) {
      v %= *t.modulus;
    }
    result = sum(result, scale(t.coefficient, v));
  }
  for (const auto &lookup : e.lookups) {
    require(lookup.port < inputs.size() && lookup.port < values.size() &&
                values[lookup.port].has_value(),
            "index_value_unresolved", "integer input content required for indirect evaluation");
    const auto &sig = inputs[lookup.port];
    require(sig.dtype == DataType::Int64 || sig.dtype == DataType::Int32, "index_invalid",
            "indirect input must be integer");
    require(lookup.coordinates.size() == sig.shape.size() &&
                values[lookup.port]->size() == elements(sig.shape),
            "index_invalid", "indirect tensor rank/content mismatch");
    Bytes flat = 0;
    for (std::size_t d = 0; d < sig.shape.size(); ++d) {
      auto v = evaluate(lookup.coordinates[d], iteration, inputs, values, depth + 1);
      require(v >= 0 && v < sig.shape[d], "index_out_of_bounds",
              "indirect index tensor access outside bounds");
      flat = checkedAdd(checkedMultiply(flat, static_cast<Bytes>(sig.shape[d])),
                        static_cast<Bytes>(v));
    }
    auto v = values[lookup.port]->at(flat);
    require(lookup.extent > 0 && v >= -lookup.extent && v < lookup.extent, "index_out_of_bounds",
            "Gather index outside target axis");
    if (v < 0) {
      v += lookup.extent;
    }
    result = sum(result, v);
  }
  return result;
}
bool sameExpression(const IndexExpression &a, const IndexExpression &b) {
  if (a.offset != b.offset || a.terms.size() != b.terms.size() || !a.lookups.empty() ||
      !b.lookups.empty()) {
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
  return true;
}
} // namespace
std::string toString(ReductionKind kind) {
  switch (kind) {
  case ReductionKind::None:
    return "none";
  case ReductionKind::Sum:
    return "sum";
  case ReductionKind::Maximum:
    return "maximum";
  case ReductionKind::Mean:
    return "mean";
  case ReductionKind::NormalizedExponential:
    return "normalized_exponential";
  }
  throw Error("index_invalid", "invalid reduction kind");
}
std::int64_t evaluateIndex(const IndexExpression &e, const Shape &iteration,
                           const std::vector<TensorSignature> &inputs,
                           const std::vector<std::optional<Shape>> &values) {
  return evaluate(e, iteration, inputs, values, 0);
}
void IndexSemantics::validate(const std::vector<TensorSignature> &inputs,
                              const std::vector<TensorSignature> &outputs) const {
  switch (status) {
  case SemanticStatus::Unknown:
  case SemanticStatus::Unsupported:
    require(!reason.empty() && pieces.empty(), "index_invalid",
            "unknown semantics need a reason, not partial rules");
    return;
  case SemanticStatus::Known:
    break;
  default:
    throw Error("index_invalid", "invalid semantic status");
  }
  require(reason.empty() && !pieces.empty(), "index_invalid", "known semantics need access pieces");
  std::set<std::size_t> written;
  for (const auto &piece : pieces) {
    bool empty = false, emptyOutput = false, hasReduction = false;
    std::set<std::string> names;
    std::size_t reductions = 0;
    for (const auto &a : piece.domain) {
      require(!a.name.empty() && names.insert(a.name).second && a.lower >= 0 && a.upper >= a.lower,
              "index_invalid", "invalid iteration domain");
      empty |= a.lower == a.upper;
      emptyOutput |= !a.reduction && a.lower == a.upper;
      hasReduction |= a.reduction;
      reductions += a.reduction ? 1 : 0;
    }
    (void)toString(piece.reduction);
    require((piece.reduction == ReductionKind::None) == !hasReduction, "index_invalid",
            "reduction kind and domain disagree");
    require(std::isfinite(piece.reductionScale) && std::isfinite(piece.afterReductionScale),
            "index_invalid", "nonfinite reduction scaling");
    if (piece.reduction == ReductionKind::None) {
      require(piece.reductionScale == 1 && piece.afterReductionScale == 1, "index_invalid",
              "scaling without reduction");
    }
    require(!piece.outputs.empty(), "index_invalid", "piece has no output access");
    for (const auto &access : piece.inputs) {
      require(access.port < inputs.size(), "index_invalid", "input access port outside range");
      require(access.phase == AccessPhase::Iteration || access.phase == AccessPhase::AfterReduction,
              "index_invalid", "invalid access phase");
      require(access.phase != AccessPhase::AfterReduction || hasReduction, "index_invalid",
              "post-reduction input without reduction");
      switch (access.outOfBounds) {
      case FillValue::None:
      case FillValue::Zero:
      case FillValue::NegativeInfinity:
        break;
      default:
        throw Error("index_invalid", "invalid boundary fill");
      }
      validateAxes(access.axes, inputs[access.port].shape.size());
      require(compatibleAxes(access.axes, inputs[access.port].axes), "axis_mismatch",
              "input violates semantic axes");
      // 归约后的读取不随归约迭代器变化，空归约也仍需验证这部分输入。
      coordinates(access.coordinates, inputs[access.port], piece, inputs,
                  access.outOfBounds != FillValue::None,
                  access.phase == AccessPhase::AfterReduction,
                  access.phase == AccessPhase::AfterReduction ? emptyOutput : empty, 0);
    }
    for (const auto &metadata : piece.shapeReads) {
      require(metadata.port < inputs.size() && piece.reduction == ReductionKind::None,
              "index_invalid", "invalid shape metadata read");
      const auto range = bounds(metadata.dimension, piece, inputs, false, empty, 0);
      require(empty || (range.first >= 0 && static_cast<std::uint64_t>(range.second) <
                                                inputs[metadata.port].shape.size()),
              "index_out_of_bounds", "shape dimension access outside rank");
    }
    if (piece.reduction == ReductionKind::Mean) {
      for (const auto &a : piece.domain) {
        require(!a.reduction || a.upper > a.lower, "index_invalid", "mean of empty reduction");
      }
    }
    if (piece.reduction == ReductionKind::NormalizedExponential) {
      require(reductions == 1 && piece.inputs.size() == 2 && piece.outputs.size() == 1 &&
                  piece.reductionScale == 1 && piece.afterReductionScale == 1 &&
                  piece.shapeReads.empty(),
              "index_invalid", "normalized exponential requires numerator and one reduction axis");
      const auto &row = piece.inputs[0];
      const auto &numerator = piece.inputs[1];
      require(row.phase == AccessPhase::Iteration &&
                  numerator.phase == AccessPhase::AfterReduction && row.port == numerator.port &&
                  row.outOfBounds == FillValue::None && numerator.outOfBounds == FillValue::None,
              "index_invalid", "normalized exponential access roles invalid");
      const auto type = inputs[row.port].dtype;
      require(type == DataType::Float16 || type == DataType::Float32 || type == DataType::Float64,
              "index_invalid", "normalized exponential requires floating data");
      std::size_t changed = 0;
      for (std::size_t d = 0; d < row.coordinates.size(); ++d) {
        if (!sameExpression(row.coordinates[d], numerator.coordinates[d])) {
          ++changed;
          const auto &expr = row.coordinates[d];
          require(expr.offset == 0 && expr.lookups.empty() && expr.terms.size() == 1,
                  "index_invalid", "normalized exponential row access invalid");
          const auto &t = expr.terms[0];
          const auto &a = piece.domain[t.axis];
          require(a.reduction && a.lower == 0 && a.upper == inputs[row.port].shape[d] &&
                      t.coefficient == 1 && t.divisor == 1 && !t.modulus,
                  "index_invalid", "normalized exponential reduction must cover full axis");
        }
      }
      require(changed == 1, "index_invalid",
              "normalized exponential must reduce exactly one data axis");
    }
    for (const auto &access : piece.outputs) {
      require(access.port < outputs.size(), "index_invalid", "output port outside range");
      written.insert(access.port);
      coordinates(access.coordinates, outputs[access.port], piece, inputs, false, true, emptyOutput,
                  0);
    }
  }
  require(written.size() == outputs.size(), "index_invalid", "missing output access relation");
}
} // namespace opfusion
