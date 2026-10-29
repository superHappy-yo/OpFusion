// 负责逻辑区域、候选/交接、部署边界与给定 Plan 的纯契约回归，不执行搜索或设备工作。
// fixture 的所有 Evidence 都是 Simulation；结构一致必须精确保留未知原因，不能升级为生产准入。
#include "opfusion/model/Model.h"

#include "opfusion/common/Error.h"
#include "opfusion/common/Validation.h"
#include "opfusion/graph/Axis.h"
#include "opfusion/graph/DAG.h"
#include "opfusion/graph/IndexSemantics.h"
#include "opfusion/graph/Operator.h"
#include "opfusion/graph/Tensor.h"
#include "opfusion/plan/Plan.h"
#include "opfusion/platform/Boundary.h"
#include "opfusion/platform/CpuDevice.h"
#include "opfusion/platform/DeclaredDevice.h"
#include "opfusion/platform/Device.h"
#include "opfusion/platform/Platform.h"
#include "opfusion/platform/Resource.h"

// 同模块私有资源校验的组合反例；不把测试活动发布为真实 Plan 或设备证据。
#include "../src/plan/PlanValidation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace opfusion;
constexpr Bytes MiB = 1024 * 1024;
int checks = 0;

void check(bool condition, const std::string &message) {
  ++checks;
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string diagnostics(const Validation &result) {
  std::string details;
  for (const auto &item : result.issues) {
    details += "\n  " + item.code + " [" + item.object + "] " + item.message;
  }
  return details;
}

void valid(const Validation &result, const std::string &message) {
  check(result.valid(), message + diagnostics(result));
}

// 允许的未知原因是显式集合；任何新增未知或 Invalid 都必须暴露，不能泛接 Unresolved。
void unresolvedExactly(const Validation &result, std::initializer_list<std::string> expected,
                       const std::string &message) {
  std::set<std::string> actual;
  for (const auto &item : result.issues) {
    check(item.kind == IssueKind::Unresolved, message + diagnostics(result));
    actual.insert(item.code);
  }
  check(actual == std::set<std::string>(expected), message + diagnostics(result));
  check(result.status() == ValidationStatus::Unresolved, message + diagnostics(result));
}

void simulatedStructure(const Validation &result, const std::string &message) {
  unresolvedExactly(result, {"candidate_evidence_unknown", "handoff_evidence_unknown"}, message);
}

void issue(const Validation &result, IssueKind kind, const std::string &code) {
  check(std::any_of(
            result.issues.begin(), result.issues.end(),
            [&](const ValidationIssue &item) { return item.kind == kind && item.code == code; }),
        "Expected diagnostic " + code + diagnostics(result));
}

void error(const std::function<void()> &operation, const std::string &code) {
  try {
    operation();
  } catch (const Error &caught) {
    check(caught.code() == code, "Expected error " + code + ", got " + caught.code());
    return;
  }
  check(false, "Expected error " + code);
}

NumericalRequirements signatureRequirements(bool reduction) {
  NumericalRequirements requirements;
  requirements.allowedTypes = {DataType::Float16};
  requirements.computeType = DataType::Float16;
  if (reduction) {
    requirements.accumulationType = DataType::Float16;
  }
  return requirements;
}

// 仅按测试给定的输入签名返回声明结果；没有数值 kernel、真实索引关系或硬件证据。
// reduction 是 fixture 明示的数值约束，不按算子名称猜测，也不宣称真实 MatMul 实现。
class FakeOperator final : public Operator {
public:
  FakeOperator(OperatorId id, std::string name, std::vector<TensorId> inputs,
               std::vector<TensorId> outputs, std::vector<TensorSignature> expected,
               std::vector<TensorSignature> result, bool reduction = false)
      : Operator(id, {"synthetic", std::move(name)}, std::move(inputs), std::move(outputs),
                 signatureRequirements(reduction)),
        mExpected(std::move(expected)), mResult(std::move(result)), mReduction(reduction) {
    validateAttributes();
  }

  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override {
    require(inputs == mExpected, "fake_signature", "Synthetic semantic input mismatch");
    return mResult;
  }

  Traits traits() const override { return {true, !mReduction, mReduction, false}; }

  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override {
    (void)infer(inputs);
    return {SemanticStatus::Unknown, "Signature-only fixture has no supplied index relation", {}};
  }

protected:
  void validateAttributes() const override {
    require(inputs().size() == mExpected.size() && outputs().size() == mResult.size(), "fake_ports",
            "Synthetic signatures must match every ordered port");
    require(!mExpected.empty() && !mResult.empty(), "fake_signature",
            "Synthetic signature oracle must have explicit input and output signatures");
    validateNumerics(mExpected);
    validateNumerics(mResult);
  }

private:
  std::vector<TensorSignature> mExpected, mResult;
  bool mReduction;
};

std::shared_ptr<const DAG> graph(bool explicitOutputs = true) {
  const TensorSignature x{{512, 1024}, DataType::Float16};
  const TensorSignature w{{1024, 2048}, DataType::Float16};
  const TensorSignature y{{512, 2048}, DataType::Float16};
  std::vector<Tensor> tensors{
      {1, "X", x.shape, x.dtype, std::nullopt, ValueKind::Input},
      {2, "W", w.shape, w.dtype, std::nullopt, ValueKind::Parameter,
       ParameterRef{"synthetic:weights", "W", "synthetic:fixed-weight-content"}},
      {3, "Z", y.shape, y.dtype, std::nullopt, ValueKind::OperatorResult},
      {4, "Y", y.shape, y.dtype, std::nullopt, ValueKind::OperatorResult}};
  std::vector<std::unique_ptr<Operator>> operators;
  operators.push_back(std::make_unique<FakeOperator>(
      1, "MatMul", std::vector<TensorId>{1, 2}, std::vector<TensorId>{3},
      std::vector<TensorSignature>{x, w}, std::vector<TensorSignature>{y}, true));
  operators.push_back(std::make_unique<FakeOperator>(
      2, "ReLU", std::vector<TensorId>{3}, std::vector<TensorId>{4},
      std::vector<TensorSignature>{y}, std::vector<TensorSignature>{y}));
  return std::make_shared<DAG>(
      "g", "synthetic MatMul-ReLU", std::move(tensors), std::move(operators),
      explicitOutputs ? std::optional<std::vector<TensorId>>{{4}} : std::nullopt,
      "registry-synthetic-1", std::vector<TensorId>{1});
}

Representation dense(const Tensor &tensor, std::string id, std::string pool) {
  Representation result;
  result.id = std::move(id);
  result.tensor = tensor.id();
  result.dtype = tensor.dtype();
  result.layout = Layout::RowMajor;
  std::vector<Bytes> strides(tensor.shape().size());
  Bytes stride = *bits(tensor.dtype()) / 8;
  for (std::size_t i = tensor.shape().size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride = checkedMultiply(stride, static_cast<Bytes>(tensor.shape()[i - 1]));
  }
  result.stridesBytes = strides;
  result.paddingElementsByAxis = std::vector<Bytes>(tensor.shape().size(), 0);
  result.alignmentBytes = 64;
  result.packing = PackingKind::Dense;
  result.quantparams = Quantization{};
  result.actualBytes = tensor.logicalBytes();
  result.pool = std::move(pool);
  result.backendContext = BackendContext{"synthetic", "1", "test-abi", result.pool};
  return result;
}

std::string rep(TensorId tensor, const std::string &pool) {
  return "r-" + std::to_string(tensor) + "-" + pool;
}
std::string qid(const std::string &operation, const std::string &device) {
  return operation + "-" + device;
}
std::string poolOf(const std::string &device) {
  return device == "cpu" ? "host" : device;
}
std::string resourceOf(const std::string &device) {
  return "compute-" + device;
}

std::vector<std::unique_ptr<const Device>> cloneDevices(const Platform &platform) {
  std::vector<std::unique_ptr<const Device>> result;
  for (const auto &device : platform.devices()) {
    result.push_back(device->clone());
  }
  return result;
}

Evidence evidenceFor(std::string id) {
  Evidence result;
  result.id = "e-" + id;
  result.objectId = std::move(id);
  result.platformId = "p";
  result.backendVersion = "1";
  result.method = "synthetic contract case, not device validation";
  result.casesRef = "test:fixed-fp16-signatures";
  result.numericPolicyId = "exact-fp16";
  result.kind = EvidenceKind::Simulation;
  result.result = EvidenceResult::Passed;
  result.observationRef = "test:hand-checked-assumptions";
  return result;
}

Stage stage(std::string id, TensorId tensor, const std::string &from, const std::string &to,
            double durationUs) {
  Stage result;
  result.id = std::move(id);
  result.from = rep(tensor, from);
  result.to = rep(tensor, to);
  result.durationUs = durationUs;
  result.source = CostSource::Analytical;
  result.sourceRef = "synthetic:section7-assumptions";
  result.workspaceComplete = true;
  if (from == to) {
    result.kind = StageKind::Identity;
    result.transferBytes = 0;
  } else {
    result.kind = StageKind::Transfer;
    result.link = from + "->" + to;
    result.transferBytes = tensor == 1 ? MiB : 2 * MiB;
    const std::string device = from == "host" ? to : from;
    result.resources = {{"bus-" + device, 1}};
    result.executionDevice = device;
    result.implementation = "impl-" + device;
  }
  return result;
}

double movement(const std::string &from, const std::string &to, Bytes bytes) {
  if (from == to) {
    return 0;
  }
  if (from == "host" || to == "host") {
    const bool gpu = from == "gpu" || to == "gpu";
    return (gpu ? 100.0 : 400.0) +
           static_cast<double>(bytes) / (gpu ? 5.0 * MiB / 1000.0 : 2.5 * MiB / 1000.0);
  }
  return movement(from, "host", bytes) + movement("host", to, bytes);
}

Model::Builder modelData(bool explicitOutputs = true) {
  Model::Builder builder;
  builder.id = "model-1";
  builder.dag = graph(explicitOutputs);
  std::vector<std::unique_ptr<const Device>> devices;
  std::vector<MemoryPool> pools;
  std::vector<Resource> resources;
  for (const std::string device : {"cpu", "gpu", "npu"}) {
    DeviceDescription description;
    description.id = device;
    description.nodeId = "node-0";
    description.kind = device == "cpu"   ? DeviceKind::Cpu
                       : device == "gpu" ? DeviceKind::Gpu
                                         : DeviceKind::Accelerator;
    description.architecture = "synthetic-arch";
    description.instance = device + "-0";
    description.configuration = "fixed";
    description.backends = {"synthetic"};
    description.environments = {
        {"synthetic", "1", "test-abi", "test-runtime-1", false, std::nullopt}};
    DeviceResourceRules rules;
    rules.compute = std::vector<ResourceUse>{{resourceOf(device), 1}};
    rules.localConversion = rules.compute;
    if (device == "cpu") {
      description.cpuAllocation = CpuAllocation{1.0, std::vector<std::uint32_t>{0},
                                                NumaBinding::NotApplicable, std::nullopt};
      devices.push_back(std::make_unique<CpuDevice>(description, rules));
    } else {
      devices.push_back(std::make_unique<DeclaredDevice>(description, rules));
    }
    pools.push_back({poolOf(device), "physical-" + poolOf(device), 64 * MiB, {device}});
    resources.push_back({resourceOf(device),
                         ResourceKind::Compute,
                         ResourceScope::Device,
                         "slots",
                         1.0,
                         device,
                         std::nullopt,
                         {},
                         {}});
  }
  // 两个方向共享同一条总线库存；共享归属是 fixture 声明，不是探测结果。
  for (const std::string device : {"gpu", "npu"}) {
    resources.push_back({"bus-" + device,
                         ResourceKind::Link,
                         ResourceScope::Link,
                         "slots",
                         1.0,
                         std::nullopt,
                         std::nullopt,
                         {"cpu", device},
                         "synthetic:shared-bus-" + device});
  }
  std::vector<Link> links;
  for (const std::string device : {"gpu", "npu"}) {
    const double bandwidth = device == "gpu" ? 5.0 * MiB / 1000.0 : 2.5 * MiB / 1000.0;
    const double startup = device == "gpu" ? 100.0 : 400.0;
    links.push_back(
        {"host->" + device, "host", device, {{"bus-" + device, 1}}, bandwidth, startup});
    links.push_back(
        {device + "->host", device, "host", {{"bus-" + device, 1}}, bandwidth, startup});
  }
  builder.platform = std::make_shared<Platform>("p", "registry-synthetic-1", std::move(devices),
                                                pools, resources, links);
  builder.numericPolicy.id = "exact-fp16";
  builder.numericPolicy.requirements.allowedTypes = {DataType::Float16};
  builder.numericPolicy.requirements.computeType = DataType::Float16;
  builder.numericPolicy.requirements.accumulationType = DataType::Float16;
  builder.boundary.id = "boundary";
  builder.boundary.platformId = "p";
  builder.boundary.inputs = {{"X-host", 1, rep(1, "host"), "synthetic:X", 0.0},
                             {"W-host", 2, rep(2, "host"), "synthetic:W", 0.0},
                             {"W-gpu", 2, rep(2, "gpu"), "synthetic:W", 0.0},
                             {"W-npu", 2, rep(2, "npu"), "synthetic:W", 0.0}};
  builder.boundary.outputs = {{"Y-host", 4, rep(4, "host")}};
  builder.boundary.timing = TimingScope{"end-to-end",
                                        TimingStart::InputsAvailable,
                                        TimingFinish::OutputsDelivered,
                                        "metric-start",
                                        "metric-finish",
                                        true,
                                        true,
                                        false};
  for (const auto &pool : pools) {
    builder.boundary.availableBytesByPool[pool.id] = 64 * MiB;
  }
  for (const auto &resource : resources) {
    builder.boundary.availableResources[resource.id] = 1.0;
  }
  builder.regions.emplace_back("mat", *builder.dag, std::vector<OperatorId>{1});
  builder.regions.emplace_back("relu", *builder.dag, std::vector<OperatorId>{2});
  builder.partitions.emplace_back("singletons", *builder.dag, builder.regions);
  for (const auto &tensor : builder.dag->tensors()) {
    for (const std::string pool : {"host", "gpu", "npu"}) {
      builder.representations.push_back(dense(tensor, rep(tensor.id(), pool), pool));
    }
  }
  for (const std::string device : {"cpu", "gpu", "npu"}) {
    Implementation implementation;
    implementation.id = "impl-" + device;
    implementation.registryId = "registry-synthetic-1";
    implementation.backend = "synthetic";
    implementation.backendVersion = "1";
    implementation.provider = {"test-provider", "1", "synthetic-content-identity"};
    implementation.artifactRef = "synthetic:library-description";
    implementation.environment = {"test-abi", "test-runtime-1", std::nullopt};
    builder.implementations.push_back(implementation);
    for (const std::string operation : {"mat", "relu"}) {
      Candidate candidate;
      candidate.id = qid(operation, device);
      candidate.region = operation;
      candidate.device = device;
      candidate.implementation = implementation.id;
      candidate.config = {"cfg-" + candidate.id,
                          "synthetic.options",
                          "1",
                          {{"precision", "fp16"}},
                          {{resourceOf(device), 1}},
                          Arithmetic{DataType::Float16, DataType::Float16, ArithmeticMode::Strict}};
      candidate.numericPolicyId = builder.numericPolicy.id;
      const auto pool = poolOf(device);
      if (operation == "mat") {
        candidate.inputs = {{0, 1, rep(1, pool)}, {1, 2, rep(2, pool)}};
      } else {
        candidate.inputs = {{0, 3, rep(3, pool)}};
      }
      const TensorId output = operation == "mat" ? 3 : 4;
      candidate.outputs = {{0, output, rep(output, pool)}};
      candidate.support = Support::Supported;
      candidate.validation = ExecutionValidation::Verified;
      candidate.search = SearchResult::Found;
      candidate.evidenceRefs = {"e-" + candidate.id};
      builder.evidence.push_back(evidenceFor(candidate.id));
      builder.candidates.push_back(candidate);
      Cost cost;
      cost.id = "cost-" + candidate.id;
      cost.candidate = candidate.id;
      cost.platformId = "p";
      cost.boundaryId = "boundary";
      cost.executionConfigId = candidate.config.id;
      cost.sourceRef = "synthetic:section7-assumed-compute";
      cost.statistic.kind = StatisticKind::Mean;
      cost.measurementContext =
          "inputs already resident and ready -> complete region output; fixed synthetic config";
      cost.serviceTimeUs = operation == "mat" ? (device == "cpu"   ? 20000
                                                 : device == "gpu" ? 2000
                                                                   : 1500)
                                              : (device == "cpu"   ? 400
                                                 : device == "gpu" ? 100
                                                                   : 50);
      cost.workspaceComplete = true;
      cost.workspaceBytesByPool[pool] = device == "cpu"      ? 0
                                        : operation == "mat" ? (device == "gpu" ? 4096 : 8192)
                                                             : 1024;
      cost.readyUsByTensor[output] = cost.serviceTimeUs;
      builder.costs.push_back(cost);
    }
  }
  for (const std::string fromDevice : {"cpu", "gpu", "npu"}) {
    for (const std::string toDevice : {"cpu", "gpu", "npu"}) {
      Handoff handoff;
      handoff.id = "h-" + fromDevice + "-" + toDevice;
      handoff.edge = builder.dag->edges().front().id;
      handoff.producer = qid("mat", fromDevice);
      handoff.consumer = qid("relu", toDevice);
      handoff.platformId = "p";
      handoff.numericPolicyId = builder.numericPolicy.id;
      handoff.validation = ExecutionValidation::Verified;
      handoff.evidenceRefs = {"e-" + handoff.id};
      builder.evidence.push_back(evidenceFor(handoff.id));
      const auto from = poolOf(fromDevice), to = poolOf(toDevice);
      if (from != "host" && to != "host" && from != to) {
        handoff.stages.push_back(
            stage(handoff.id + "-a", 3, from, "host", movement(from, "host", 2 * MiB)));
        handoff.stages.push_back(
            stage(handoff.id + "-b", 3, "host", to, movement("host", to, 2 * MiB)));
      } else {
        handoff.stages.push_back(
            stage(handoff.id + "-a", 3, from, to, movement(from, to, 2 * MiB)));
      }
      builder.handoffs.push_back(handoff);
    }
  }
  for (const std::string device : {"gpu", "npu"}) {
    builder.boundaryStages.push_back(
        stage("upload-" + device, 1, "host", device, movement("host", device, MiB)));
    builder.boundaryStages.push_back(
        stage("download-" + device, 4, device, "host", movement(device, "host", 2 * MiB)));
  }
  return builder;
}

Plan::Builder planData(std::shared_ptr<const Model> model, const std::string &mat = "gpu",
                       const std::string &relu = "gpu") {
  Plan::Builder builder;
  builder.id = "plan-" + mat + "-" + relu;
  builder.model = std::move(model);
  builder.partition = "singletons";
  builder.selections = {{"mat", qid("mat", mat), "cost-" + qid("mat", mat)},
                        {"relu", qid("relu", relu), "cost-" + qid("relu", relu)}};
  const auto addEvent = [&](std::string id, EventKind kind, std::string subject,
                            std::vector<EventId> dependencies, double timeUs) {
    builder.events.push_back(
        {std::move(id), kind, std::move(subject), std::move(dependencies), timeUs});
  };
  const auto ensure = [&](TensorId tensor, const std::string &pool, const EventId &ready,
                          bool resident) {
    const auto bindingId = "b-" + std::to_string(tensor) + "-" + pool;
    if (std::any_of(builder.tensorBindings.begin(), builder.tensorBindings.end(),
                    [&](const TensorBinding &binding) { return binding.id == bindingId; })) {
      return bindingId;
    }
    const auto &representation = builder.model->representation(rep(tensor, pool));
    const auto allocationId = "a-" + bindingId;
    Allocation allocation;
    allocation.id = allocationId;
    allocation.pool = representation.pool;
    allocation.sizeBytes = representation.actualBytes;
    allocation.alignmentBytes = 64;
    allocation.preexisting = resident;
    if (!resident) {
      allocation.allocate = "alloc-" + allocationId;
      addEvent(*allocation.allocate, EventKind::Allocate, allocationId, {}, 0);
    }
    builder.allocations.push_back(allocation);
    builder.tensorBindings.push_back(
        {bindingId, tensor, representation.id, allocationId, 0, ready});
    return bindingId;
  };
  std::vector<EventId> sourceEvents;
  for (const auto &source : builder.model->boundary().inputs) {
    const auto sourceEvent = "source-" + source.id;
    addEvent(sourceEvent, EventKind::SourceReady, source.id, {}, *source.readyUs);
    sourceEvents.push_back(sourceEvent);
    // 后缀是 fixture 的副本身份；sharedPoolTest 可以令多个副本引用同一个物理池。
    const std::string pool = source.id == "X-host" || source.id == "W-host" ? "host"
                             : source.id == "W-gpu"                         ? "gpu"
                                                                            : "npu";
    builder.sources.push_back(
        {source.id, ensure(source.tensor, pool, sourceEvent, true), sourceEvent});
  }
  addEvent("metric-start", EventKind::TimingStart, "end-to-end", sourceEvents, 0);
  const auto allocationDependencies = [&](const std::vector<std::string> &bindings,
                                          std::vector<EventId> &dependencies) {
    for (const auto &bindingId : bindings) {
      const auto binding =
          std::find_if(builder.tensorBindings.begin(), builder.tensorBindings.end(),
                       [&](const TensorBinding &entry) { return entry.id == bindingId; });
      const auto allocation =
          std::find_if(builder.allocations.begin(), builder.allocations.end(),
                       [&](const Allocation &entry) { return entry.id == binding->allocation; });
      if (allocation->allocate && std::find(dependencies.begin(), dependencies.end(),
                                            *allocation->allocate) == dependencies.end()) {
        dependencies.push_back(*allocation->allocate);
      }
    }
  };
  const auto addTask = [&](Task task, double startUs, double serviceUs,
                           std::vector<EventId> dependencies) {
    allocationDependencies(task.inputs, dependencies);
    allocationDependencies(task.outputs, dependencies);
    for (const auto &binding : task.stageBindings) {
      allocationDependencies({binding.inputBinding, binding.outputBinding}, dependencies);
    }
    dependencies.push_back("metric-start");
    task.start = "start-" + task.id;
    task.complete = "end-" + task.id;
    addEvent(task.start, EventKind::TaskStart, task.id, std::move(dependencies), startUs);
    addEvent(task.complete, EventKind::TaskComplete, task.id, {task.start}, startUs + serviceUs);
    builder.tasks.push_back(std::move(task));
  };
  const auto matPool = poolOf(mat), reluPool = poolOf(relu);
  double now = 0;
  std::string x = "b-1-host", xReady = "source-X-host";
  if (matPool != "host") {
    const auto &upload = builder.model->boundaryStage("upload-" + matPool);
    x = ensure(1, matPool, "end-upload", false);
    Task task;
    task.id = "upload";
    task.kind = TaskKind::BoundaryStage;
    task.reference = upload.id;
    task.inputs = {"b-1-host"};
    task.outputs = {x};
    task.stageBindings = {{upload.id, task.inputs[0], x}};
    addTask(task, 0, *upload.durationUs, {xReady});
    now = *upload.durationUs;
    xReady = "end-upload";
  }
  const auto compute = [&](const std::string &operation, const std::string &device,
                           std::vector<std::string> inputs, std::vector<EventId> dependencies,
                           double startUs) {
    const auto candidate = qid(operation, device);
    const auto &cost = builder.model->cost("cost-" + candidate);
    const auto output =
        ensure(operation == "mat" ? 3 : 4, poolOf(device), "end-" + operation, false);
    Task task;
    task.id = operation;
    task.kind = TaskKind::Region;
    task.reference = candidate;
    task.inputs = std::move(inputs);
    task.outputs = {output};
    for (const auto &workspace : cost.workspaceBytesByPool) {
      if (workspace.second && *workspace.second > 0) {
        const auto allocationId = "workspace-" + operation;
        const auto allocated = "alloc-" + allocationId;
        Allocation allocation;
        allocation.id = allocationId;
        allocation.pool = workspace.first;
        allocation.sizeBytes = *workspace.second;
        allocation.alignmentBytes = 64;
        allocation.allocate = allocated;
        builder.allocations.push_back(allocation);
        addEvent(allocated, EventKind::Allocate, allocationId, {}, 0);
        dependencies.push_back(allocated);
        builder.workspaces.push_back(
            {operation, workspace.first, allocationId, 0, *workspace.second});
      }
    }
    addTask(task, startUs, *cost.serviceTimeUs, std::move(dependencies));
    return *cost.serviceTimeUs;
  };
  now += compute("mat", mat, {x, "b-2-" + matPool}, {xReady, "source-W-" + matPool}, now);
  const auto &handoff = builder.model->handoff("h-" + mat + "-" + relu);
  Task join;
  join.id = "join";
  join.kind = TaskKind::Handoff;
  join.reference = handoff.id;
  join.inputs = {"b-3-" + matPool};
  std::string previous = join.inputs[0];
  for (const auto &adapter : handoff.stages) {
    const std::string targetPool = adapter.to.substr(adapter.to.find_last_of('-') + 1);
    const auto target =
        ensure(3, targetPool, matPool == targetPool ? "end-mat" : "end-join", false);
    join.stageBindings.push_back({adapter.id, previous, target});
    previous = target;
  }
  join.outputs = {previous};
  addTask(join, now, *handoff.serviceTimeUs(), {"end-mat"});
  now += *handoff.serviceTimeUs();
  now += compute("relu", relu, {"b-3-" + reluPool}, {"end-join", "end-mat"}, now);
  std::string final = "b-4-" + reluPool, finalReady = "end-relu";
  if (reluPool != "host") {
    const auto &download = builder.model->boundaryStage("download-" + reluPool);
    final = ensure(4, "host", "end-download", false);
    Task task;
    task.id = "download";
    task.kind = TaskKind::BoundaryStage;
    task.reference = download.id;
    task.inputs = {"b-4-" + reluPool};
    task.outputs = {final};
    task.stageBindings = {{download.id, task.inputs[0], final}};
    addTask(task, now, *download.durationUs, {"end-relu"});
    now += *download.durationUs;
    finalReady = "end-download";
  }
  addEvent("sink-Y", EventKind::SinkComplete, "Y-host", {finalReady}, now);
  builder.sinks.push_back({"Y-host", final, "sink-Y"});
  addEvent("metric-finish", EventKind::TimingFinish, "end-to-end", {"sink-Y"}, now);
  return builder;
}

Event &event(Plan::Builder &builder, const std::string &id) {
  const auto found = std::find_if(builder.events.begin(), builder.events.end(),
                                  [&](const Event &entry) { return entry.id == id; });
  require(found != builder.events.end(), "fixture_event", "Unknown test event " + id);
  return *found;
}
Allocation &allocation(Plan::Builder &builder, const std::string &id) {
  const auto found = std::find_if(builder.allocations.begin(), builder.allocations.end(),
                                  [&](const Allocation &entry) { return entry.id == id; });
  require(found != builder.allocations.end(), "fixture_allocation",
          "Unknown test allocation " + id);
  return *found;
}
void dropDependency(Event &entry, const std::string &dependency) {
  entry.dependsOn.erase(std::remove(entry.dependsOn.begin(), entry.dependsOn.end(), dependency),
                        entry.dependsOn.end());
}
void removeTimingEvents(Plan::Builder &builder) {
  builder.events.erase(std::remove_if(builder.events.begin(), builder.events.end(),
                                      [](const Event &entry) {
                                        return entry.id == "metric-start" ||
                                               entry.id == "metric-finish";
                                      }),
                       builder.events.end());
  for (auto &entry : builder.events) {
    dropDependency(entry, "metric-start");
  }
}

void regionTests() {
  const TensorSignature signature{{1}, DataType::Float16};
  const auto make = [&](bool branch) {
    std::vector<Tensor> tensors{{1, "input", signature.shape, signature.dtype},
                                {2, "a", signature.shape, signature.dtype},
                                {3, "b", signature.shape, signature.dtype},
                                {4, "c", signature.shape, signature.dtype}};
    std::vector<std::unique_ptr<Operator>> nodes;
    for (OperatorId id = 1; id <= 3; ++id) {
      nodes.push_back(std::make_unique<FakeOperator>(
          id, "Pure", std::vector<TensorId>{id == 3 && branch ? 2 : id},
          std::vector<TensorId>{id + 1}, std::vector<TensorSignature>{signature},
          std::vector<TensorSignature>{signature}));
    }
    return std::make_shared<DAG>("regions", "regions", std::move(tensors), std::move(nodes),
                                 branch ? std::vector<TensorId>{3, 4} : std::vector<TensorId>{4},
                                 "registry-synthetic-1", std::vector<TensorId>{1});
  };
  const auto branch = make(true);
  Region fused("fused", *branch, {1, 2});
  // 区域内的中间值仍被外部分支读取，不能只保留区域末端输出。
  check(fused.inputs() == std::vector<TensorId>{1} &&
            fused.outputs() == std::vector<TensorId>({2, 3}),
        "Fusion preserves external branch output and model output");
  const auto chain = make(false);
  Region firstAndLast("a", *chain, {1, 3}), middle("b", *chain, {2});
  error([&] { Partition rejected("bad", *chain, {firstAndLast, middle}); }, "partition_cycle");
  error([&] { Partition rejected("bad", *branch, {fused, Region("overlap", *branch, {2, 3})}); },
        "partition_overlap");
  error([&] { Partition rejected("bad", *branch, {fused}); }, "partition_coverage");
}

void sharedPoolTest() {
  auto builder = modelData();
  builder.platform = std::make_shared<Platform>(
      "p", "registry-synthetic-1", cloneDevices(*builder.platform),
      std::vector<MemoryPool>{{"shared", "one-physical-pool", 64 * MiB, {"cpu", "gpu", "npu"}}},
      builder.platform->resources(), std::vector<Link>{});
  for (auto &representation : builder.representations) {
    representation.pool = "shared";
    representation.backendContext->context = "shared";
  }
  builder.boundary.availableBytesByPool.clear();
  builder.boundary.availableBytesByPool["shared"] = 64 * MiB;
  for (auto &cost : builder.costs) {
    const auto amount = cost.workspaceBytesByPool.begin()->second;
    cost.workspaceBytesByPool.clear();
    cost.workspaceBytesByPool["shared"] = amount;
  }
  for (auto &handoff : builder.handoffs) {
    Stage identity = handoff.stages.front();
    identity.kind = StageKind::Identity;
    identity.to = handoff.stages.back().to;
    identity.link.reset();
    identity.resources.clear();
    identity.durationUs = 0;
    identity.transferBytes = 0;
    handoff.stages = {identity};
  }
  for (auto &adapter : builder.boundaryStages) {
    adapter.kind = StageKind::Identity;
    adapter.link.reset();
    adapter.resources.clear();
    adapter.durationUs = 0;
    adapter.transferBytes = 0;
  }
  auto model = std::make_shared<Model>(builder);
  auto plan = planData(model);
  const auto alias = [&](const AllocationId &oldId, const AllocationId &actual) {
    for (auto &binding : plan.tensorBindings) {
      if (binding.allocation == oldId) {
        binding.allocation = actual;
      }
    }
    plan.allocations.erase(
        std::remove_if(plan.allocations.begin(), plan.allocations.end(),
                       [&](const Allocation &entry) { return entry.id == oldId; }),
        plan.allocations.end());
    const auto removedEvent = "alloc-" + oldId;
    plan.events.erase(std::remove_if(plan.events.begin(), plan.events.end(),
                                     [&](const Event &entry) { return entry.id == removedEvent; }),
                      plan.events.end());
    for (auto &entry : plan.events) {
      dropDependency(entry, removedEvent);
    }
  };
  alias("a-b-2-gpu", "a-b-2-host");
  alias("a-b-2-npu", "a-b-2-host");
  alias("a-b-1-gpu", "a-b-1-host");
  alias("a-b-4-host", "a-b-4-gpu");
  // Identity 输出必须复用真实 Allocation；同一 workspace 不能再被加一次峰值。
  Bytes total = 0;
  for (const auto &entry : plan.allocations) {
    total = checkedAdd(total, *entry.sizeBytes);
  }
  check(total == 9 * MiB + 5120, "Shared memory and workspaces hand-count to 9 MiB + 5120 bytes");
  builder.boundary.availableBytesByPool["shared"] = total;
  plan.model = std::make_shared<Model>(builder);
  simulatedStructure(Plan(plan).validate(), "Shared physical pool/allocation charged once");
  builder.boundary.availableBytesByPool["shared"] = total - 1;
  plan.model = std::make_shared<Model>(builder);
  issue(Plan(plan).validate(), IssueKind::Invalid, "pool_capacity");
}

void boundaryResourceTests(const std::shared_ptr<const Model> &model) {
  Boundary boundary = model->boundary();
  boundary.availableBytesByPool["gpu"] = 65 * MiB;
  issue(boundary.validate(model->platform()), IssueKind::Invalid, "available_exceeds_physical");
  boundary = model->boundary();
  boundary.availableResources["compute-gpu"] = std::nullopt;
  unresolvedExactly(boundary.validate(model->platform()), {"available_resource_unknown"},
                    "Unknown deployment resource limit is not the physical limit");
  auto builder = modelData();
  builder.boundary = boundary;
  auto plan = planData(model);
  plan.model = std::make_shared<Model>(builder);
  unresolvedExactly(
      Plan(plan).validate(),
      {"candidate_evidence_unknown", "handoff_evidence_unknown", "available_resource_unknown"},
      "Plan cannot infer missing deployment limits from Platform");
  boundary = model->boundary();
  boundary.availableResources["compute-gpu"] = 0.0;
  valid(boundary.validate(model->platform()),
        "Known zero dynamic capacity is a valid unavailable limit");
  builder = modelData();
  builder.boundary = boundary;
  plan.model = std::make_shared<Model>(builder);
  issue(Plan(plan).validate(), IssueKind::Invalid, "resource_available");
  issue(Plan(plan).validate(), IssueKind::Invalid, "resource_conflict");
  for (const double amount : {2.0, -1.0, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    boundary = model->boundary();
    boundary.availableResources["compute-gpu"] = amount;
    issue(boundary.validate(model->platform()), IssueKind::Invalid, "available_resource");
  }
  boundary = model->boundary();
  boundary.availableResources["absent-resource"] = 1.0;
  issue(boundary.validate(model->platform()), IssueKind::Invalid, "resource_missing");
}

void timingBoundaryTests(const std::shared_ptr<const Model> &model) {
  auto plan = planData(model);
  // 所有工作推迟 100 us，输入仍在 0 就绪；旧规则允许把这段等待从指标中截掉。
  for (auto &entry : plan.events) {
    if (entry.kind != EventKind::SourceReady && entry.kind != EventKind::Allocate) {
      *entry.timeUs += 100;
    }
  }
  issue(Plan(plan).validate(), IssueKind::Invalid, "timing_inputs_available_time");

  plan = planData(model);
  *event(plan, "metric-finish").timeUs += 100;
  issue(Plan(plan).validate(), IssueKind::Invalid, "timing_outputs_delivered_time");

  plan = planData(model);
  *event(plan, "sink-Y").timeUs += 100;
  *event(plan, "metric-finish").timeUs += 100;
  issue(Plan(plan).validate(), IssueKind::Invalid, "sink_completion_time");

  auto data = modelData();
  data.boundary.inputs[1].readyUs = 100;
  auto staggered = std::make_shared<Model>(data);
  plan = planData(staggered);
  for (auto &entry : plan.events) {
    if (entry.kind != EventKind::SourceReady && entry.kind != EventKind::Allocate) {
      *entry.timeUs += 100;
    }
  }
  simulatedStructure(Plan(plan).validate(),
                     "InputsAvailable equals the latest source, including nonzero readiness");

  data = modelData();
  data.boundary.outputs.push_back({"Y-gpu", 4, rep(4, "gpu")});
  auto twoTargets = std::make_shared<Model>(data);
  plan = planData(twoTargets);
  plan.events.push_back({"sink-Y-gpu",
                         EventKind::SinkComplete,
                         "Y-gpu",
                         {"end-relu"},
                         event(plan, "end-relu").timeUs});
  plan.sinks.push_back({"Y-gpu", "b-4-gpu", "sink-Y-gpu"});
  event(plan, "metric-finish").dependsOn.push_back("sink-Y-gpu");
  simulatedStructure(Plan(plan).validate(),
                     "OutputsDelivered waits for the latest of multiple required destinations");

  plan = planData(model);
  event(plan, "source-X-host").timeUs.reset();
  unresolvedExactly(
      Plan(plan).validate(),
      {"candidate_evidence_unknown", "handoff_evidence_unknown", "event_time_unknown"},
      "An unknown source time cannot be ignored when deriving timing boundaries");
}

void cumulativeResourceTests() {
  auto data = modelData();
  auto resources = data.platform->resources();
  for (auto &resource : resources) {
    if (resource.id == "compute-gpu") {
      resource.capacity = 3.0;
    }
  }
  data.platform =
      std::make_shared<Platform>("p", "registry-synthetic-1", cloneDevices(*data.platform),
                                 data.platform->pools(), resources, data.platform->links());
  data.boundary.availableResources["compute-gpu"] = 2.0;
  auto model = std::make_shared<Model>(data);
  Plan::Builder supplied;
  supplied.id = "private-resource-check";
  supplied.model = model;
  const Plan plan(std::move(supplied));
  const detail::PlanIndex index;
  const auto validate = [&](const std::vector<detail::ResourceActivity> &activities,
                            const detail::EventOrder &order) {
    detail::PlanUsage usage;
    usage.activities = activities;
    Validation result;
    detail::validateResources({plan, *model, index, order}, usage, result);
    return result;
  };

  // 三个任务的预测时间串行，但偏序没有保证；两两用量 2 不超限，合计 3 超限。
  std::vector<detail::ResourceActivity> activities{
      {"a", "start-a", "end-a", 0, 10, {"compute-gpu", 1}},
      {"b", "start-b", "end-b", 10, 20, {"compute-gpu", 1}},
      {"c", "start-c", "end-c", 20, 30, {"compute-gpu", 1}}};
  detail::EventOrder unordered;
  // EventOrder 接口保存传递闭包；各任务内部 start→complete 始终存在。
  unordered.ancestors["end-a"] = {"start-a"};
  unordered.ancestors["end-b"] = {"start-b"};
  unordered.ancestors["end-c"] = {"start-c"};
  unresolvedExactly(validate(activities, unordered), {"resource_capacity_order_unproven"},
                    "Pairwise fit and nonoverlapping timestamps do not prove cumulative capacity");

  auto ordered = unordered;
  ordered.ancestors["start-c"] = {"start-a", "end-a", "start-b", "end-b"};
  ordered.ancestors["end-c"] = {"start-a", "end-a", "start-b", "end-b", "start-c"};
  valid(validate(activities, ordered),
        "Two tasks may overlap when the third waits for both, without forced full serialization");

  activities[1].startUs = 0;
  activities[1].endUs = 10;
  activities[2].startUs = 0;
  activities[2].endUs = 10;
  issue(validate(activities, unordered), IssueKind::Invalid, "resource_conflict");

  for (auto &activity : activities) {
    activity.use.amount = 0.5;
  }
  valid(validate(activities, unordered), "Unordered requests are allowed when their total fits");
  for (std::size_t i = 0; i < activities.size(); ++i) {
    activities[i].use.amount = 0.75;
    activities[i].startUs = static_cast<double>(i) * 10;
    activities[i].endUs = activities[i].startUs + 10;
  }
  unresolvedExactly(validate(activities, unordered), {"resource_capacity_order_unproven"},
                    "Fractional requests respect the dynamic limit, not just physical capacity");

  activities = {{"a", "start-a", "end-a", 0, 10, {"compute-gpu", 1}},
                {"a", "start-a", "end-a", 10, 20, {"compute-gpu", 1}},
                {"b", "start-b", "end-b", 0, 20, {"compute-gpu", 0.75}}};
  valid(validate(activities, unordered),
        "Sequential stages within one task contribute their local peak rather than their sum");
}
} // namespace

int main() {
  try {
    static_assert(!std::is_copy_assignable<Model>::value,
                  "Published Model must not be overwritten");
    static_assert(!std::is_copy_assignable<Platform>::value,
                  "Published Platform must not be overwritten");
    auto data = modelData();
    auto model = std::make_shared<Model>(data);
    simulatedStructure(model->validate(),
                       "Synthetic model has no structural violation, but no device proof");
    check(std::all_of(
              model->evidence().begin(), model->evidence().end(),
              [](const Evidence &evidence) { return evidence.kind == EvidenceKind::Simulation; }),
          "All fixture evidence remains Simulation");
    for (const auto &candidate : model->candidates()) {
      unresolvedExactly(model->validateCandidate(candidate), {"candidate_evidence_unknown"},
                        "Each synthetic candidate lacks real execution evidence");
    }
    check(model->candidatesFor("mat").size() == 3 && model->candidatesFor("relu").size() == 3,
          "Complete candidate identities retain all three hypothetical device assignments");
    check(model->eligibleCandidatesFor("mat").empty() &&
              model->eligibleCandidatesFor("relu").empty(),
          "Simulation cannot create production-eligible candidates");
    check(model->deviceDomain("mat").empty() && model->deviceDomain("relu").empty(),
          "Device domain is derived only from eligible candidates, never Simulation labels");
    check(model->dag().tensor(1).logicalBytes() == MiB &&
              model->dag().tensor(2).logicalBytes() == 4 * MiB &&
              model->dag().tensor(3).logicalBytes() == 2 * MiB,
          "FP16 X/W/Z byte products are hand-calculated");
    check(model->representation(rep(3, "gpu")).requiredBytes(model->dag().tensor(3)) == 2 * MiB,
          "Dense byte strides reproduce FP16 size");
    auto representation = model->representation(rep(1, "host"));
    representation.actualBytes.reset();
    unresolvedExactly(representation.validate(model->dag().tensor(1), model->platform()),
                      {"actual_bytes_unknown"},
                      "Unknown physical bytes are not a zero-sized representation");
    representation.actualBytes = 0;
    issue(representation.validate(model->dag().tensor(1), model->platform()), IssueKind::Invalid,
          "representation_size");
    Tensor empty(99, "empty", {0, 100}, DataType::Float16);
    auto emptyRepresentation = dense(empty, "empty", "host");
    valid(emptyRepresentation.validate(empty, model->platform()),
          "Real empty tensor has known zero physical bytes");
    error([] { checkedMultiply(std::numeric_limits<Bytes>::max(), 2); }, "size_overflow");
    Tensor huge(100, "huge", {2}, DataType::Float16);
    auto hugeRepresentation = dense(huge, "huge", "host");
    hugeRepresentation.paddingElementsByAxis =
        std::vector<Bytes>{std::numeric_limits<Bytes>::max()};
    issue(hugeRepresentation.validate(huge, model->platform()), IssueKind::Invalid,
          "size_overflow");
    representation = model->representation(rep(1, "host"));
    representation.dtype = DataType::Unknown;
    check(!representation.requiredBytes(model->dag().tensor(1)),
          "Unknown dtype has unknown bytes, not zero");
    unresolvedExactly(representation.validate(model->dag().tensor(1), model->platform()),
                      {"representation_dtype_unknown"},
                      "Storage dtype cannot be guessed from logical bytes");
    representation = model->representation(rep(1, "host"));
    representation.stridesBytes = std::vector<Bytes>{2, 2};
    issue(representation.validate(model->dag().tensor(1), model->platform()), IssueKind::Invalid,
          "representation_overlap");
    representation = model->representation(rep(1, "host"));
    representation.layout.reset();
    unresolvedExactly(representation.validate(model->dag().tensor(1), model->platform()),
                      {"layout_unknown"},
                      "Byte strides do not supply a missing layout declaration");
    representation.layout = static_cast<Layout>(999);
    issue(representation.validate(model->dag().tensor(1), model->platform()), IssueKind::Invalid,
          "layout_invalid");
    representation.layout = Layout::NHWC;
    issue(representation.validate(model->dag().tensor(1), model->platform()), IssueKind::Invalid,
          "layout_rank");
    Tensor fourAxes(101, "four axes", {1, 2, 3, 4}, DataType::Float16);
    auto named = dense(fourAxes, "named-layout", "host");
    const std::vector<std::pair<Layout, AxisOrder>> namedOrders{{Layout::NCHW, AxisOrder::NCHW},
                                                                {Layout::NHWC, AxisOrder::NHWC},
                                                                {Layout::OIHW, AxisOrder::OIHW},
                                                                {Layout::HWIO, AxisOrder::HWIO}};
    for (const auto &order : namedOrders) {
      named.layout = order.first;
      unresolvedExactly(named.validate(fourAxes, model->platform()), {"layout_axes_unknown"},
                        "Rank alone does not establish a named physical axis order");
      Tensor declared(101, "declared axes", {1, 2, 3, 4}, DataType::Float16,
                      logicalAxes(order.second));
      valid(named.validate(declared, model->platform()),
            "Matching named axes/strides are a structural declaration, not device capability "
            "evidence");
    }
    named.layout = Layout::Opaque;
    unresolvedExactly(named.validate(fourAxes, model->platform()), {"layout_evidence_unknown"},
                      "Opaque layout needs an explicit format witness");
    named.packingValidationRef = "synthetic:opaque-layout-witness";
    valid(named.validate(fourAxes, model->platform()),
          "Opaque layout retains explicit witness and backend context");
    check(model->cost("cost-mat-gpu").workspaceBytesByPool.at("gpu") == 4096 &&
              model->cost("cost-mat-npu").workspaceBytesByPool.at("npu") == 8192,
          "Workspace follows the exact q, not the operator");
    check(model->handoff("h-npu-gpu").serviceTimeUs() == 1700,
          "NPU -> host -> GPU has 1200+500 us assumed service, no queueing");
    double best = std::numeric_limits<double>::infinity();
    for (const std::string mat : {"cpu", "gpu", "npu"}) {
      for (const std::string relu : {"cpu", "gpu", "npu"}) {
        auto plan = planData(model, mat, relu);
        const auto validation = Plan(plan).validate();
        simulatedStructure(validation,
                           "Nine serial mappings are structurally consistent: " + mat + "/" + relu);
        check(!validation.valid(), "No Simulation mapping is a complete Valid production plan");
        const double end = *event(plan, "sink-Y").timeUs;
        best = std::min(best, end);
        if (mat == "gpu" && relu == "gpu") {
          check(end == 2900, "0.30+2.00+0.10+0.50 ms = 2.90 ms with host Source/Sink");
        }
        if (mat == "npu" && relu == "npu") {
          check(end == 3550, "0.80+1.50+0.05+1.20 ms = 3.55 ms");
        }
        if (mat == "npu" && relu == "gpu") {
          check(end == 4600, "0.80+1.50+1.70+0.10+0.50 ms = 4.60 ms");
        }
      }
    }
    // 这里只核算给定的九组假设，没有运行优化器，更不是实机时延或最优性结论。
    check(best == 2900,
          "All nine hypothetical assignments have minimum 2.90 ms; not hardware evidence");
    auto plan = planData(model);
    const Plan frozen(plan);
    check(frozen.taskGraph().nodes.size() == 10,
          "Derived graph includes four Sources, five tasks and required host Sink");
    valid(frozen.taskGraph().validate(), "Derived Source/Sink task graph is acyclic");
    plan = planData(model, "gpu", "cpu");
    Task reused;
    reused.id = "reuse";
    reused.kind = TaskKind::Handoff;
    reused.reference = "h-gpu-cpu";
    reused.reuse = "join";
    reused.start = "start-reuse";
    reused.complete = "end-reuse";
    reused.inputs = {"b-3-gpu"};
    reused.outputs = {"b-3-host"};
    plan.tasks.push_back(reused);
    plan.events.push_back({reused.start, EventKind::TaskStart, reused.id, {"end-join"}, 2800.0});
    plan.events.push_back(
        {reused.complete, EventKind::TaskComplete, reused.id, {reused.start}, 2800.0});
    event(plan, "start-relu").dependsOn.push_back(reused.complete);
    simulatedStructure(Plan(plan).validate(),
                       "Completed replica supports a dependency-only join without a second copy");
    event(plan, "start-reuse").dependsOn = {"end-mat"};
    issue(Plan(plan).validate(), IssueKind::Invalid, "handoff_reuse_ready");
    auto withoutExplicitOutputs = std::make_shared<Model>(modelData(false));
    plan = planData(model);
    plan.model = withoutExplicitOutputs;
    unresolvedExactly(
        Plan(plan).validate(),
        {"candidate_evidence_unknown", "handoff_evidence_unknown", "model_outputs_unknown"},
        "Terminal-output fallback cannot prove complete model outputs");
    data.costs[2].serviceTimeUs = 999;
    check(model->cost("cost-mat-gpu").serviceTimeUs == 2000,
          "Editing a builder does not mutate a published snapshot");
    plan = planData(model);
    plan.selections[0].cost = "cost-relu-gpu";
    issue(Plan(plan).validate(), IssueKind::Invalid, "selection_cost");
    // 输出预分配及所有读者/传输的完成依赖不能由预测时间或输出就绪替代。
    plan = planData(model);
    dropDependency(event(plan, "start-mat"), "alloc-a-b-3-gpu");
    issue(Plan(plan).validate(), IssueKind::Invalid, "output_preallocation");
    plan = planData(model);
    plan.events.push_back({"free-Z", EventKind::Release, "a-b-3-gpu", {"end-mat"}, 2300.0});
    allocation(plan, "a-b-3-gpu").release = "free-Z";
    issue(Plan(plan).validate(), IssueKind::Invalid, "allocation_early_release");
    plan = planData(model);
    plan.events.push_back({"free-X", EventKind::Release, "a-b-1-host", {"source-X-host"}, 100.0});
    allocation(plan, "a-b-1-host").release = "free-X";
    issue(Plan(plan).validate(), IssueKind::Invalid, "allocation_early_release");
    plan = planData(model);
    dropDependency(event(plan, "start-relu"), "end-join");
    issue(Plan(plan).validate(), IssueKind::Invalid, "handoff_ready_dependency");
    plan = planData(model);
    event(plan, "start-relu").timeUs = 2200;
    event(plan, "end-relu").timeUs = 2300;
    issue(Plan(plan).validate(), IssueKind::Invalid, "resource_conflict");
    plan = planData(model);
    plan.workspaces.push_back(plan.workspaces.front());
    issue(Plan(plan).validate(), IssueKind::Invalid, "workspace_duplicate");
    plan = planData(model);
    plan.workspaces.front().sizeBytes = 4095;
    issue(Plan(plan).validate(), IssueKind::Invalid, "workspace_too_small");
    plan = planData(model);
    plan.workspaces.erase(plan.workspaces.begin());
    issue(Plan(plan).validate(), IssueKind::Invalid, "workspace_reservation_missing");
    plan = planData(model);
    event(plan, "start-mat").timeUs.reset();
    unresolvedExactly(
        Plan(plan).validate(),
        {"candidate_evidence_unknown", "handoff_evidence_unknown", "event_time_unknown"},
        "Missing schedule time cannot pass as feasible");
    auto unknown = modelData();
    unknown.costs[2].serviceTimeUs.reset();
    auto unknownModel = std::make_shared<Model>(unknown);
    check(Plan(planData(model)).model()->id() == "model-1",
          "Plan retains the frozen model identity");
    plan = planData(model);
    plan.model = unknownModel;
    unresolvedExactly(Plan(plan).validate(),
                      {"candidate_evidence_unknown", "handoff_evidence_unknown", "duration_unknown",
                       "task_service_unknown"},
                      "Unknown candidate duration is not zero");
    unknown = modelData();
    unknown.costs[2].workspaceComplete = false;
    unknownModel = std::make_shared<Model>(unknown);
    plan = planData(model);
    plan.model = unknownModel;
    unresolvedExactly(
        Plan(plan).validate(),
        {"candidate_evidence_unknown", "handoff_evidence_unknown", "workspace_unknown"},
        "Incomplete workspace map does not imply zero workspace");
    unknown = modelData();
    unknown.boundary.timing.reset();
    unknownModel = std::make_shared<Model>(unknown);
    plan = planData(model);
    plan.model = unknownModel;
    removeTimingEvents(plan);
    unresolvedExactly(
        Plan(plan).validate(),
        {"candidate_evidence_unknown", "handoff_evidence_unknown", "timing_scope_unknown"},
        "Missing timing scope is explicit, not an orphan-event counterexample");
    unknown = modelData();
    unknown.boundary.outputs.clear();
    unknownModel = std::make_shared<Model>(unknown);
    unresolvedExactly(
        unknownModel->validate(),
        {"candidate_evidence_unknown", "handoff_evidence_unknown", "output_target_unknown"},
        "Every logical model output needs an explicit delivery target");
    unknown = modelData();
    unknown.evidence[2].observationRef.clear();
    unknownModel = std::make_shared<Model>(unknown);
    plan = planData(model);
    plan.model = unknownModel;
    unresolvedExactly(
        Plan(plan).validate(),
        {"candidate_evidence_unknown", "handoff_evidence_unknown", "evidence_scope_unknown"},
        "Missing observation reference does not become passing evidence");
    unknown = modelData();
    unknown.candidates[2].support = Support::Unknown;
    unknown.candidates[2].validation = ExecutionValidation::Pending;
    unknown.candidates[2].search = SearchResult::BudgetExhausted;
    unknownModel = std::make_shared<Model>(unknown);
    plan = planData(model);
    plan.model = unknownModel;
    unresolvedExactly(Plan(plan).validate(),
                      {"candidate_evidence_unknown", "handoff_evidence_unknown", "support_unknown",
                       "candidate_pending", "selection_unverified"},
                      "Budget exhaustion and unknown support are not an infeasibility proof");
    // 未选探索项不能引入新的计划未知项；已选 Simulation 的证据缺口仍必须保留。
    unknown = modelData();
    auto extra = unknown.candidates[2];
    extra.id = "exploration";
    extra.support = Support::Unknown;
    extra.validation = ExecutionValidation::Pending;
    extra.evidenceRefs.clear();
    unknown.candidates.push_back(extra);
    unknownModel = std::make_shared<Model>(unknown);
    plan = planData(model);
    plan.model = unknownModel;
    simulatedStructure(Plan(plan).validate(),
                       "Unselected alternatives do not poison a specified selection");
    unknown = modelData();
    unknown.candidates.clear();
    unknown.costs.clear();
    unknown.handoffs.clear();
    unknown.evidence.clear();
    unknownModel = std::make_shared<Model>(unknown);
    unresolvedExactly(unknownModel->validate(), {"candidate_pool_empty"},
                      "Empty candidate pool is unresolved, not global infeasibility");
    check(unknownModel->eligibleCandidatesFor("mat").empty() &&
              unknownModel->deviceDomain("mat").empty(),
          "An empty catalog supplies no eligible device domain");
    auto bad = modelData();
    bad.costs[2].source = CostSource::Measured;
    error([&] { Model rejected(bad); }, "measurement_ref_required");
    bad = modelData();
    bad.costs[2].source = CostSource::Predicted;
    bad.costs[2].samplesUs = {2000};
    error([&] { Model rejected(bad); }, "predicted_samples");
    bad = modelData();
    bad.costs[2].executionConfigId = "another-q";
    error([&] { Model rejected(bad); }, "cost_conditions");
    auto handoff = model->handoff("h-npu-gpu");
    handoff.stages[1].from = rep(3, "npu");
    issue(model->validateHandoff(handoff), IssueKind::Invalid, "handoff_continuity");
    auto adapter = model->boundaryStage("upload-gpu");
    adapter.link = "gpu->host";
    issue(model->validateStage(adapter), IssueKind::Invalid, "stage_link_direction");
    adapter = model->boundaryStage("upload-gpu");
    adapter.transferBytes = 0;
    issue(model->validateStage(adapter), IssueKind::Invalid, "transfer_bytes");
    adapter = model->handoff("h-gpu-gpu").stages.front();
    adapter.to = rep(3, "npu");
    issue(model->validateStage(adapter), IssueKind::Invalid, "identity_representation");
    bad = modelData();
    auto quantized = bad.representations[7];
    quantized.id = "quantized-Z";
    quantized.tensor = 3;
    quantized.dtype = DataType::Int8;
    quantized.actualBytes = MiB;
    quantized.stridesBytes = std::vector<Bytes>{2048, 1};
    quantized.quantparams = Quantization{QuantizationKind::PerTensor, {0.1}, {0}, std::nullopt};
    bad.representations.push_back(quantized);
    unknownModel = std::make_shared<Model>(bad);
    adapter = stage("quant", 3, "gpu", "gpu", 10);
    adapter.from = rep(3, "gpu");
    adapter.to = quantized.id;
    adapter.kind = StageKind::Quantize;
    adapter.resources = {{"compute-gpu", 1}};
    adapter.implementation = "impl-gpu";
    adapter.executionDevice = "gpu";
    issue(unknownModel->validateStage(adapter), IssueKind::Invalid, "numeric_policy");
    // Compute 和本地转换必须走设备最低资源规则；非空元数据不等于预留执行资源。
    auto candidate = model->candidate("mat-gpu");
    candidate.config.resources.clear();
    issue(model->validateCandidate(candidate), IssueKind::Invalid, "execution_resource_missing");
    issue(model->validateCandidate(candidate), IssueKind::Invalid, "execution_compute_resource");
    adapter = stage("local-repack", 3, "gpu", "gpu", 10);
    adapter.kind = StageKind::Repack;
    adapter.resources = {{"compute-gpu", 1}};
    adapter.implementation = "impl-gpu";
    adapter.executionDevice = "gpu";
    valid(model->validateStage(adapter),
          "Declared local repack satisfies structural resource rules only");
    adapter.resources.clear();
    issue(model->validateStage(adapter), IssueKind::Invalid, "execution_resource_missing");
    issue(model->validateStage(adapter), IssueKind::Invalid, "execution_compute_resource");
    auto pools = model->platform().pools();
    pools[1].physicalIdentity = pools[0].physicalIdentity;
    error(
        [&] {
          Platform rejected("p", "registry-synthetic-1", cloneDevices(model->platform()), pools,
                            model->platform().resources(), model->platform().links());
        },
        "physical_pool_duplicate");
    boundaryResourceTests(model);
    timingBoundaryTests(model);
    cumulativeResourceTests();
    regionTests();
    sharedPoolTest();
    std::cout << "model tests: " << checks
              << " checks passed (synthetic only; no device execution)\n";
    return 0;
  } catch (const std::exception &caught) {
    std::cerr << "model tests failed: " << caught.what() << '\n';
    return 1;
  }
}
