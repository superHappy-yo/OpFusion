#pragma once

// 给定候选的事件、缓冲与资源计划；只做纯校验，不执行分配、调度或设备工作。
#include "opfusion/model/Model.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

using TaskId = std::string;
using EventId = std::string;
using AllocationId = std::string;

/// 计划事件类别；Source/Sink 属于派生执行视图，不是原图逻辑算子。
enum class EventKind {
  Allocate, Release, TaskStart, TaskComplete, SourceReady, SinkComplete, TimingStart, TimingFinish
};
/// 显式事件依赖与可选微秒时间；预测时间不能替代必须存在的偏序依赖。
struct Event {
  EventId id;
  EventKind kind = EventKind::TaskStart;
  std::string subject;
  std::vector<EventId> dependsOn;
  std::optional<double> timeUs;
};
/// 执行任务的职责，不按算子名称或设备型号分类。
enum class TaskKind { Region, Handoff, BoundaryStage };
/// 阶段的实际输入/输出缓冲绑定；同一逻辑值仍可能对应不同分配。
struct StageBinding {
  std::string stage;
  std::string inputBinding;
  std::string outputBinding;
};
/// 一个区域、精确交接或边界适配的工作记录，不保存运行时句柄。
struct Task {
  TaskId id;
  TaskKind kind = TaskKind::Region;
  /// Candidate、Handoff 或边界 Stage 的身份，不是算子名称。
  std::string reference;
  /// 引用已完成交接以复用就绪副本；该任务仅建立依赖，不再执行原阶段。
  std::optional<TaskId> reuse;
  EventId start;
  EventId complete;
  /// 按候选边界端口稳定排列；Handoff/BoundaryStage 各有一个输入/输出。
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<StageBinding> stageBindings;
};
/// 每区域选择一个完整候选及其成本，不能拼接其他候选的耗时/工作区。
struct Selection {
  RegionId region;
  CandidateId candidate;
  CostId cost;
};
/// 唯一实际分配的计划声明；物理池容量按此身份计费，不按 TensorBinding 重复加总。
struct Allocation {
  AllocationId id;
  PoolId pool;
  std::optional<Bytes> sizeBytes;
  std::optional<Bytes> alignmentBytes;
  /// 已驻留分配自计划起点计费，不以首次消费者启动作为存活起点。
  bool preexisting = false;
  std::optional<EventId> allocate;
  /// 未声明释放表示持有至结束；输出就绪不等于所有读者/传输已完成。
  std::optional<EventId> release;
};
/// 逻辑值/候选表示到实际分配的映射；offsetBytes 为分配内字节偏移。
struct TensorBinding {
  std::string id;
  TensorId tensor = 0;
  RepresentationId representation;
  AllocationId allocation;
  Bytes offsetBytes = 0;
  /// 完整值就绪的事件，不表示存储可立即释放。
  EventId ready;
};
/// 任务工作区的分配切片；首版按完整任务保守预留，不暗示内部 tile 存活证明。
struct Workspace {
  TaskId task;
  PoolId pool;
  AllocationId allocation;
  Bytes offsetBytes = 0;
  Bytes sizeBytes = 0;
};
/// 输入驻留或输出目标与实际绑定/事件的对应关系。
struct BoundaryBinding {
  /// 分别引用 Residency.id 或 OutputTarget.id。
  std::string boundaryEntry;
  std::string tensorBinding;
  EventId event;
};
/// 派生执行视图的节点类别；Unknown 不表示有效 Source 或设备工作。
enum class TaskNodeKind { Unknown, Source, Region, Handoff, BoundaryStage, Sink };
/// 派生节点的具名身份；类别参与身份比较，默认 Unknown 会被校验拒绝。
struct TaskNode {
  TaskNodeKind kind = TaskNodeKind::Unknown;
  std::string id;
};
/// 派生 Source/Task/Sink 视图中的依赖；tensor 保留逻辑值身份。
struct TaskArc {
  TaskNode from;
  TaskNode to;
  TensorId tensor = 0;
};
/// 派生任务图，不改变原始 DAG，也不执行调度。
struct TaskGraph {
  std::vector<TaskNode> nodes;
  std::vector<TaskArc> arcs;
  /// 核验身份、引用与无环；端口/缓冲语义由 Plan 的其他阶段核验。
  Validation validate() const;
};

/// 人工给定的不可变模型校验计划，不是优化器、调度器或运行实例。
/// 查询引用借用自本实例；共享 Model 必须在计划存活期间保持不可变。
class Plan {
 public:
  /// 显式组装记录；不通过构造过程执行设备或自动补足未知条件。
  struct Builder {
    std::string id;
    std::shared_ptr<const Model> model;
    PartitionId partition;
    std::vector<Selection> selections;
    std::vector<Task> tasks;
    std::vector<Event> events;
    std::vector<Allocation> allocations;
    std::vector<TensorBinding> tensorBindings;
    std::vector<Workspace> workspaces;
    std::vector<BoundaryBinding> sources;
    std::vector<BoundaryBinding> sinks;
  };
  /// 按值接管记录；校验显式调用 validate，便于保留非法/未知输入的完整诊断。
  explicit Plan(Builder builder);
  Plan(const Plan&) = default;
  Plan& operator=(const Plan&) = delete;
  /// 计划身份。
  const std::string& id() const noexcept { return mData.id; }
  /// 借用共享只读模型指针；允许缺失以便 validate 返回 plan_model 诊断。
  const std::shared_ptr<const Model>& model() const noexcept { return mData.model; }
  /// 选择的完整划分身份。
  const PartitionId& partition() const noexcept { return mData.partition; }
  /// 借用完整候选/成本选择。
  const std::vector<Selection>& selections() const noexcept { return mData.selections; }
  /// 借用实际任务目录。
  const std::vector<Task>& tasks() const noexcept { return mData.tasks; }
  /// 借用偏序事件与给定时间表。
  const std::vector<Event>& events() const noexcept { return mData.events; }
  /// 借用唯一分配声明，不执行内存分配。
  const std::vector<Allocation>& allocations() const noexcept { return mData.allocations; }
  /// 借用逻辑值到物理分配的绑定。
  const std::vector<TensorBinding>& tensorBindings() const noexcept { return mData.tensorBindings; }
  /// 借用完整任务工作区预留。
  const std::vector<Workspace>& workspaces() const noexcept { return mData.workspaces; }
  /// 借用实际输入驻留绑定。
  const std::vector<BoundaryBinding>& sources() const noexcept { return mData.sources; }
  /// 借用必须交付的输出绑定。
  const std::vector<BoundaryBinding>& sinks() const noexcept { return mData.sinks; }
  /// 从当前记录派生 Source/Task/Sink 图；缺失必要引用时抛出 Error。
  TaskGraph taskGraph() const;
  /// 分阶段核验选择、事件、内存、资源与条件；Valid 不等于已经在设备运行。
  Validation validate() const;
 private:
  Builder mData;
};

}  // namespace opfusion
