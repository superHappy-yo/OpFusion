#pragma once

#include "opfusion/graph/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

/// 索引知识状态，不等同于设备支持或融合合法性结论。
enum class SemanticStatus { Known, Unknown, Unsupported };
/// 逻辑归约：Sum 初值零，Maximum 初值负无穷；Mean 要求归约非空。
/// NormalizedExponential 表示 exp(x_i-max(x))/sum(exp(x_j-max(x)))，
/// 第一个读取给出完整归约行，第二个 AfterReduction 读取给出分子 x_i。
enum class ReductionKind { None, Sum, Maximum, Mean, NormalizedExponential };
/// 归约后读取的值不参与归约乘积，例如 Gemm 的偏置或 Softmax 的分子。
enum class AccessPhase { Iteration, AfterReduction };
/// 输入越界时的数学填充值；None 要求访问保持在逻辑边界内。
enum class FillValue { None, Zero, NegativeInfinity };

/// 索引项 coefficient * floor(iterator[axis] / divisor)。
/// 迭代器非负，divisor 必须为正，coefficient 不得为零。
struct IndexTerm {
  std::size_t axis = 0;
  std::int64_t coefficient = 1;
  std::int64_t divisor = 1;
  /// 有值时为 floor(iterator/divisor) % modulus；modulus 必须为正。
  std::optional<std::int64_t> modulus = std::nullopt;
};
struct IndexLookup;
/// offset 加各项及受检间接值之和；同一坐标不能重复引用相同 iterator。
struct IndexExpression {
  std::vector<IndexTerm> terms;
  std::int64_t offset = 0;
  std::vector<IndexLookup> lookups = {};
};
/// 间接地址依赖一个整数输入值；必须验证 -extent <= value < extent，负值加 extent。
/// Known 表示关系已知，不代表运行时索引值已经验收；执行者必须保留该范围检查。
struct IndexLookup {
  std::size_t port = 0;
  std::vector<IndexExpression> coordinates;
  std::int64_t extent = 0;
};
/// 具名半开迭代域 [lower, upper)，允许空域；reduction 区分归约维。
struct IterationAxis {
  std::string name;
  std::int64_t lower = 0;
  std::int64_t upper = 0;
  bool reduction = false;
};
/// 按有序输入端口描述逻辑坐标及填充，不引用物理内存或张量数据。
struct InputAccess {
  std::size_t port = 0;
  std::vector<IndexExpression> coordinates;
  FillValue outOfBounds = FillValue::None;
  std::optional<LogicalAxes> axes;
  /// 声明单迭代器访问保留对应输出轴含义，供 DAG 发布前收敛轴知识。
  bool preservesAxes = false;
  AccessPhase phase = AccessPhase::Iteration;
};
/// 按输出端口描述写入坐标；坐标不得依赖归约迭代器。
struct OutputAccess {
  std::size_t port = 0;
  std::vector<IndexExpression> coordinates;
};
/// 元数据读取访问输入签名的 shape[dimension]，不读取输入张量元素。
struct ShapeAccess {
  std::size_t port = 0;
  IndexExpression dimension;
};

/// 一段迭代域及读写关系；Concat 用不同 piece 表达互斥输出区间。
struct IndexPiece {
  std::vector<IterationAxis> domain;
  std::vector<InputAccess> inputs;
  std::vector<OutputAccess> outputs;
  ReductionKind reduction = ReductionKind::None;
  std::vector<ShapeAccess> shapeReads = {};
  /// 普通归约结果的缩放及归约后加项的缩放；Softmax 必须均为 1。
  double reductionScale = 1.0, afterReductionScale = 1.0;
};

/// 值语义的访问关系，不是编译器 IR，也不是完整融合/执行支持证明。
/// Known 必须有 pieces 且无 reason；Unknown/Unsupported 必须有原因且无部分规则。
/// 输出范围只取非归约域，因此空归约不消除本应产生的输出。
struct IndexSemantics {
  SemanticStatus status = SemanticStatus::Unknown;
  std::string reason;
  std::vector<IndexPiece> pieces;
  /// 同步借用输入/输出签名，校验状态、端口、域、坐标边界和溢出。
  /// 调用方先保证静态签名合法；失败抛出 Error，不修改关系或签名。
  /// 检查结构与保守坐标范围，不证明任意多段关系的覆盖/互斥或数值等价。
  void validate(const std::vector<TensorSignature> &inputs,
                const std::vector<TensorSignature> &outputs) const;
};

/// 受检的纯整数地址求值；间接输入缺失/越界和算术溢出明确失败，不访问设备。
std::int64_t evaluateIndex(const IndexExpression &expression, const Shape &iteration,
                           const std::vector<TensorSignature> &inputs = {},
                           const std::vector<std::optional<Shape>> &integerInputs = {});
std::string toString(ReductionKind kind);

} // namespace opfusion
