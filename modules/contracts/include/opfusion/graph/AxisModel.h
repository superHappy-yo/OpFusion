#pragma once
#include "opfusion/graph/IndexSemantics.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace opfusion {
class DAG;
/// 轴坐标的结构来源，不是人为猜测的通道/头数等业务名称。
enum class AxisStructure {
  Signature,
  Generated,
  Singleton,
  Direct,
  Split,
  Merged,
  Reindexed,
  Indirect,
  Metadata
};
enum class AxisModelOrigin { Signature, OperatorRelation, Scalar };
std::string toString(AxisStructure kind);
std::string toString(AxisModelOrigin origin);
/// 精确关联到生产者索引 piece 的一个输入维度，read 保存其实际坐标表达式。
struct AxisSource {
  TensorId tensor = 0;
  std::size_t inputPort = 0, access = 0, dimension = 0;
  IndexExpression read;
};
/// 一段输出域上的轴来源；piece 引用同一生产者的 IndexSemantics.pieces。
struct AxisDerivation {
  std::size_t piece = 0;
  AxisStructure structure = AxisStructure::Reindexed;
  IndexExpression write;
  std::vector<AxisSource> sources;
  std::vector<ShapeAccess> shapeReads;
};
/// 完整的逻辑坐标轴。业务命名是可选注释，不决定坐标是否已建模。
struct CoordinateAxis {
  std::size_t position = 0;
  std::int64_t extent = 0;
  std::optional<Axis> semantic;
  std::vector<AxisDerivation> derivations;
};
/// 绑定不可变图/目录的结构轴视图；不修改 Tensor 的命名轴约束，不推断物理布局。
struct TensorAxisModel {
  TensorId tensor = 0;
  std::string graphId, registryId;
  SemanticStatus status = SemanticStatus::Unknown;
  std::string reason;
  AxisModelOrigin origin = AxisModelOrigin::Signature;
  std::optional<OperatorId> producer;
  std::optional<std::size_t> outputPort;
  std::vector<TensorId> inputs;
  std::vector<std::size_t> pieces;
  std::optional<AxisOrigin> semanticOrigin;
  std::vector<CoordinateAxis> axes;
  /// 核验范围、来源、结构分类和表达式确实来自该图的生产者关系；不能伪造 known。
  void validate(const DAG &graph) const;
};
/// 批量构造结构轴谱系，保留标量无轴、元数据、取模拆合和间接索引的区别。
/// 静态输入/参数的坐标来自已验证签名；生产者关系未知时保留该真实状态。
std::map<TensorId, TensorAxisModel> buildAxisModels(const DAG &graph);
} // namespace opfusion
