#pragma once

// 一次 Plan 校验的私有索引、借用视图和阶段入口，不是公开执行上下文。
#include "opfusion/common/Error.h"
#include "opfusion/plan/Plan.h"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace opfusion::detail {

/// 仅在一次校验期间借用不可变 Plan 的记录；不得跨调用缓存这些指针。
struct PlanIndex {
  std::map<EventId, const Event*> events;
  std::map<TaskId, const Task*> tasks;
  std::map<AllocationId, const Allocation*> allocations;
  std::map<std::string, const TensorBinding*> bindings;
};

/// 由事件依赖构造的祖先关系；horizonUs 为给定时间表的最大有限微秒值。
struct EventOrder {
  std::map<EventId, std::set<EventId>> ancestors;
  double horizonUs = 0;
  /// 同一事件视为非严格先于；时间戳本身不建立偏序关系。
  bool precedes(const EventId& before, const EventId& after) const;
};

/// 同步借用本次计划、模型及派生索引/偏序；不拥有记录或修改冻结快照。
struct PlanValidationView {
  const Plan& plan;
  const Model& model;
  const PlanIndex& index;
  const EventOrder& order;
};

/// 某次实际请求的预测活动区间；规则下限不会再生成一份活动。
struct ResourceActivity {
  std::string owner;
  EventId startEvent;
  EventId endEvent;
  double startUs = 0;
  double endUs = 0;
  ResourceUse use;
};

/// 校验局部派生记录；保留所有绑定的读/传输区间，不写回计划或模型。
struct PlanUsage {
  std::map<RegionId, const Selection*> selections;
  std::map<CandidateId, const Task*> computeTasks;
  std::map<std::string, std::vector<std::pair<EventId, EventId>>> bindingUses;
  std::map<std::string, std::size_t> providers;
  std::vector<ResourceActivity> activities;
  std::map<TaskId, std::map<PoolId, std::optional<Bytes>>> workspaceRequirements;
  std::vector<const Task*> handoffTasks;
};

/// 从本次借用索引查询记录；未知身份抛出带种类的 Error，不默认插入。
template <class T>
const T& lookupPlanEntry(const std::map<std::string, const T*>& entries,
                         const std::string& id, const std::string& kind) {
  const auto found = entries.find(id);
  require(found != entries.end(), kind + "_missing", kind + " not found: " + id);
  return *found->second;
}

/// 建立本次索引，重复/空身份追加诊断；返回对象继续借用 plan。
PlanIndex indexPlan(const Plan& plan, Validation& validation);
/// 核验所依赖的模型与完整 IO 条件，不把未选候选的未知项全部传播。
void validatePlanModel(const Plan& plan, Validation& validation);
/// 核验完整候选/成本选择、覆盖与一致统计口径，记录所选项的借用。
void validateSelections(const Plan& plan, PlanUsage& usage, Validation& validation);
/// 建立事件祖先关系并检查时间；有环返回 false，调用方停止后续偏序核验。
bool buildEventOrder(const Plan& plan, const PlanIndex& index, EventOrder& order,
                      Validation& validation);
/// 借用事件并核验类别/拥有者；缺失引用抛出 Error。
const Event& eventFor(const PlanValidationView& view, const EventId& id, EventKind kind,
                       const std::string& subject, Validation& validation);
/// 沿用计划时间比较的相对容差，不用于放宽设备数值正确性。
bool nearPlanTime(double lhs, double rhs);
/// 核验使用前分配及使用后释放，first/last 均为具体事件引用。
void useAllocation(const PlanValidationView& view, const AllocationId& allocation,
                    const EventId& first, const EventId& last, const std::string& owner,
                    Validation& validation);
/// 记录完整使用区间并核验输入就绪；bool input 只区分真实输入/输出用途。
void useBinding(const PlanValidationView& view, PlanUsage& usage, const std::string& binding,
                 const EventId& first, const EventId& last, const std::string& owner,
                 bool input, Validation& validation);
/// 核验分配切片的字节边界；未知大小不视为零，溢出由 Error 报告。
void checkAllocationBounds(const Allocation& allocation, Bytes offset, std::optional<Bytes> bytes,
                            const std::string& owner, Validation& validation);
/// 核验实际分配的池、大小、对齐、物理/动态容量和分配事件。
void validateAllocations(const PlanValidationView& view, Validation& validation);
/// 核验逻辑值、表示、分配与就绪事件的绑定。
void validateTensorBindings(const PlanValidationView& view, Validation& validation);
/// 核验实际 Source 与 Boundary 驻留逐项匹配，记录供给者。
void validateSources(const PlanValidationView& view, PlanUsage& usage, Validation& validation);
/// 核验区域及适配任务，派生实际资源活动、工作区和全部绑定使用。
void validateTasks(const PlanValidationView& view, PlanUsage& usage, Validation& validation);
/// 核验所选区域交叉边的精确交接及消费者依赖。
void validateHandoffs(const PlanValidationView& view, const PlanUsage& usage,
                       Validation& validation);
/// 核验每个要求交付的 Sink 及其实际完成事件。
void validateSinks(const PlanValidationView& view, PlanUsage& usage, Validation& validation);
/// 核验完整任务工作区和既有绑定别名规则，不进行真实内存分配。
void validateWorkspaceAndAliases(const PlanValidationView& view, const PlanUsage& usage,
                                 Validation& validation);
/// 核验给定时间表容量、排他复用事件顺序和任务峰值的偏序保守上界，不生成调度。
void validateResources(const PlanValidationView& view, const PlanUsage& usage,
                        Validation& validation);
/// 拒绝没有实际拥有者的孤立事件与多重供给者。
void validateEventOwnership(const PlanValidationView& view, Validation& validation);

}  // namespace opfusion::detail
