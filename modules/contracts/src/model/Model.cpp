// 给定区域/候选/表示/成本的纯一致性校验；不搜索候选、不加载制品或运行设备。
#include "opfusion/model/Model.h"

#include "RepresentationChecks.h"
#include "opfusion/common/Error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace opfusion {
namespace {
template <class T>
const T &lookup(const std::vector<T> &values, const std::string &id, const std::string &kind) {
  const auto it =
      std::find_if(values.begin(), values.end(), [&](const T &x) { return x.id == id; });
  require(it != values.end(), kind + "_missing", kind + " not found: " + id);
  return *it;
}
template <class T>
void identities(const std::vector<T> &values, const std::string &kind, Validation &v) {
  std::set<std::string> seen;
  for (const auto &x : values)
    if (x.id.empty() || !seen.insert(x.id).second)
      v.invalid("identity_invalid", x.id, "Empty or duplicate " + kind + " identity");
}
bool contains(const std::vector<OperatorId> &ids, OperatorId id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}
bool hasTensor(const std::vector<TensorId> &ids, TensorId id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}
void uniqueTensor(std::vector<TensorId> &ids, TensorId id) {
  if (!hasTensor(ids, id))
    ids.push_back(id);
}
bool finiteNonnegative(double x) {
  return std::isfinite(x) && x >= 0.0;
}
bool near(double a, double b) {
  return std::abs(a - b) <= 1e-9 * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}
void throwInvalid(const Validation &v) {
  for (const auto &issue : v.issues)
    if (issue.kind == IssueKind::Invalid)
      throw Error(issue.code, issue.object + ": " + issue.message);
}
void duration(const std::optional<double> &value, const std::string &id, Validation &v) {
  if (!value)
    v.unresolved("duration_unknown", id, "Unknown service duration is not zero");
  else if (!finiteNonnegative(*value))
    v.invalid("duration_invalid", id, "Duration must be finite and nonnegative");
}
void provenance(CostSource source, const std::string &ref,
                const std::vector<std::string> &measurements, const std::string &id,
                Validation &v) {
  switch (source) {
  case CostSource::Analytical:
  case CostSource::Predicted:
  case CostSource::Measured:
    break;
  default:
    v.invalid("cost_source_invalid", id, "Invalid cost source");
  }
  if (ref.empty())
    v.unresolved("cost_source_unknown", id, "Cost provenance is required");
  if (source == CostSource::Measured && measurements.empty())
    v.invalid("measurement_ref_required", id, "Measured costs require observation references");
  if (source != CostSource::Measured && !measurements.empty())
    v.invalid("cost_source_mixed", id,
              "Predicted/analytical records cannot masquerade as measured records");
  for (const auto &m : measurements)
    if (m.empty())
      v.invalid("measurement_ref_required", id, "Empty observation reference");
}
void workspace(const Platform &p, bool complete,
               const std::map<PoolId, std::optional<Bytes>> &values, const std::string &id,
               Validation &v) {
  if (!complete)
    v.unresolved("workspace_unknown", id, "Workspace declaration is incomplete, not zero");
  for (const auto &entry : values) {
    try {
      const auto &pool = p.pool(entry.first);
      if (!entry.second)
        v.unresolved("workspace_unknown", id, "Unknown bytes in pool " + entry.first);
      else if (pool.physicalCapacityBytes && *entry.second > *pool.physicalCapacityBytes)
        v.invalid("workspace_capacity", id, "Workspace exceeds physical pool capacity");
    } catch (const Error &e) {
      v.invalid(e.code(), id, e.what());
    }
  }
}
void numericRepresentation(const Representation &r, const Tensor &t, const NumericPolicy &policy,
                           const std::string &id, Validation &v) {
  const auto &req = policy.requirements;
  if (t.dtype() == DataType::Unknown) {
    v.unresolved("logical_dtype_unknown", id,
                 "Logical dtype is unknown; a storage dtype cannot silently resolve it");
    return;
  }
  if (r.dtype == DataType::Unknown)
    return;
  if (!req.allowedTypes.empty() && std::find(req.allowedTypes.begin(), req.allowedTypes.end(),
                                             r.dtype) == req.allowedTypes.end())
    v.invalid("numeric_type", id, "Representation dtype is excluded by numeric policy");
  const bool quantized = r.quantparams && r.quantparams->kind != QuantizationKind::None;
  if (r.dtype != t.dtype() || quantized) {
    if (!req.allowApproximation || (quantized && !policy.allowQuantization))
      v.invalid("numeric_policy", id, "Precision/quantization changes require explicit permission");
    if (policy.validationRefs.empty())
      v.unresolved("numeric_evidence_unknown", id,
                   "Approximate numeric policy needs validation references");
  }
}
} // namespace

Region::Region(RegionId id, const DAG &graph, std::vector<OperatorId> members)
    : mId(std::move(id)), mGraphId(graph.id()), mMembers(std::move(members)) {
  require(!mId.empty() && !mMembers.empty(), "region_empty",
          "Region needs identity and nonempty members");
  std::set<OperatorId> seen;
  for (auto node : mMembers) {
    graph.node(node);
    require(seen.insert(node).second, "region_duplicate", "Repeated region member");
  }
  std::map<TensorId, OperatorId> producers;
  for (const auto &node : graph.nodes())
    for (auto t : node->outputs())
      producers.emplace(t, node->id());
  const auto &outputs = graph.modelOutputs() ? *graph.modelOutputs() : graph.terminalOutputs();
  // 端口顺序来自原图拓扑而非调用方成员排列；跨区域消费者需要的中间值必须保留。
  for (auto nodeId : graph.topologicalOrder()) {
    if (!seen.count(nodeId))
      continue;
    const auto &node = graph.node(nodeId);
    for (auto t : node.inputs()) {
      const auto producer = producers.find(t);
      if (producer == producers.end() || !seen.count(producer->second))
        uniqueTensor(mInputs, t);
    }
    for (auto t : node.outputs()) {
      bool external = hasTensor(outputs, t);
      for (const auto &edge : graph.edges())
        if (edge.tensor == t && !seen.count(edge.consumer))
          external = true;
      if (external)
        uniqueTensor(mOutputs, t);
    }
  }
}
Partition::Partition(PartitionId id, const DAG &graph, const std::vector<Region> &regions)
    : mId(std::move(id)), mGraphId(graph.id()) {
  require(!mId.empty(), "partition_identity", "Partition identity is required");
  std::map<OperatorId, RegionId> owner;
  std::map<RegionId, std::set<RegionId>> next;
  std::map<RegionId, std::size_t> indegree;
  for (const auto &region : regions) {
    require(region.graphId() == graph.id(), "partition_graph", "Region belongs to another graph");
    require(indegree.emplace(region.id(), 0).second, "partition_region_duplicate",
            "Duplicate region identity");
    mRegions.push_back(region.id());
    for (auto node : region.members()) {
      graph.node(node);
      require(owner.emplace(node, region.id()).second, "partition_overlap", "Regions overlap");
    }
  }
  require(owner.size() == graph.nodes().size(), "partition_coverage",
          "Partition must cover every operator exactly once");
  // 原图无环不保证区域商图无环；跨区边按区域去重后重新做拓扑核验。
  for (const auto &e : graph.edges()) {
    const auto &a = owner.at(e.producer);
    const auto &b = owner.at(e.consumer);
    if (a != b && next[a].insert(b).second)
      ++indegree[b];
  }
  std::vector<RegionId> ready;
  for (const auto &entry : indegree)
    if (entry.second == 0)
      ready.push_back(entry.first);
  std::size_t visited = 0;
  while (!ready.empty()) {
    const auto idValue = ready.back();
    ready.pop_back();
    ++visited;
    for (const auto &target : next[idValue])
      if (--indegree[target] == 0)
        ready.push_back(target);
  }
  require(visited == regions.size(), "partition_cycle",
          "Acyclic original graph has a cyclic region quotient");
}

bool operator==(const BackendContext &a, const BackendContext &b) {
  return a.backend == b.backend && a.version == b.version && a.abi == b.abi &&
         a.context == b.context;
}
bool operator==(const Quantization &a, const Quantization &b) {
  return a.kind == b.kind && a.scales == b.scales && a.zeroPoints == b.zeroPoints &&
         a.axis == b.axis;
}
bool sameRepresentation(const Representation &a, const Representation &b, bool location) {
  return a.tensor == b.tensor && a.dtype == b.dtype && a.layout == b.layout &&
         a.stridesBytes == b.stridesBytes && a.paddingElementsByAxis == b.paddingElementsByAxis &&
         a.alignmentBytes == b.alignmentBytes && a.packing == b.packing &&
         a.packingFormat == b.packingFormat && a.packingValidationRef == b.packingValidationRef &&
         a.quantparams == b.quantparams && a.actualBytes == b.actualBytes &&
         (!location || (a.pool == b.pool && a.backendContext == b.backendContext));
}
std::optional<Bytes> Representation::requiredBytes(const Tensor &value) const {
  if (!packing || *packing != PackingKind::Dense || !stridesBytes || !paddingElementsByAxis)
    return std::nullopt;
  const auto widthBits = bits(dtype);
  if (!widthBits)
    return std::nullopt;
  require(*widthBits % 8 == 0, "representation_dtype",
          "Dense storage requires byte-addressable dtype");
  const auto &shape = value.shape();
  require(stridesBytes->size() == shape.size() && paddingElementsByAxis->size() == shape.size(),
          "representation_rank", "Stride and padding ranks must equal tensor rank");
  std::vector<std::pair<Bytes, Bytes>> axes;
  bool empty = false;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    require(shape[i] >= 0, "shape_unresolved", "Static nonnegative dimensions required");
    const auto extent = checkedAdd(static_cast<Bytes>(shape[i]), (*paddingElementsByAxis)[i]);
    if (extent == 0)
      empty = true;
    axes.emplace_back((*stridesBytes)[i], extent);
  }
  if (empty)
    return Bytes{0};
  // 按物理步长累积跨度以拒绝不同元素别名；padding 会增大物理范围而非逻辑 shape。
  std::sort(axes.begin(), axes.end());
  Bytes span = *widthBits / 8;
  for (const auto &axis : axes) {
    if (axis.second <= 1)
      continue;
    require(axis.first >= span, "representation_overlap", "Dense strides alias distinct elements");
    span = checkedAdd(span, checkedMultiply(axis.second - 1, axis.first));
  }
  return span;
}
Validation Representation::validate(const Tensor &value, const Platform &p) const {
  Validation v;
  if (id.empty() || tensor != value.id())
    v.invalid("representation_identity", id, "Representation must bind its logical tensor");
  try {
    const auto &physical = p.pool(pool);
    if (physical.physicalIdentity.empty())
      v.unresolved("physical_pool_unknown", id, "Representation pool identity is unknown");
    if (!physical.physicalCapacityBytes)
      v.unresolved("physical_capacity_unknown", id, "Representation pool capacity is unknown");
    else if (actualBytes && *actualBytes > *physical.physicalCapacityBytes)
      v.invalid("representation_capacity", id, "Representation exceeds its physical pool");
    (void)bits(dtype);
  } catch (const Error &e) {
    v.invalid(e.code(), id, e.what());
  }
  if (dtype == DataType::Unknown)
    v.unresolved("representation_dtype_unknown", id, "Storage dtype is unknown");
  if (!layout)
    v.unresolved("layout_unknown", id, "Layout is required");
  else {
    switch (*layout) {
    case Layout::RowMajor:
    case Layout::ColumnMajor:
    case Layout::Strided:
      break;
    case Layout::NCHW:
    case Layout::NHWC:
    case Layout::OIHW:
    case Layout::HWIO:
      if (value.shape().size() != 4)
        v.invalid("layout_rank", id, "Named four-axis layout requires a rank-four tensor");
      break;
    case Layout::Opaque:
      if (packingValidationRef.empty())
        v.unresolved("layout_evidence_unknown", id,
                     "Opaque layout requires a format validation witness");
      break;
    default:
      v.invalid("layout_invalid", id, "Unrecognized Layout enum value");
      break;
    }
  }
  if (!stridesBytes)
    v.unresolved("strides_unknown", id, "Byte strides are unknown");
  if (!paddingElementsByAxis)
    v.unresolved("padding_unknown", id, "Element padding is unknown");
  if (!alignmentBytes)
    v.unresolved("alignment_unknown", id, "Alignment is unknown");
  else if (*alignmentBytes == 0 || (*alignmentBytes & (*alignmentBytes - 1)) != 0)
    v.invalid("alignment_invalid", id, "Alignment must be a positive power of two");
  if (!actualBytes)
    v.unresolved("actual_bytes_unknown", id, "Actual physical bytes are unknown, not zero");
  if (!packing)
    v.unresolved("packing_unknown", id, "Packing classification is unknown");
  else
    switch (*packing) {
    case PackingKind::Dense:
      if (!packingFormat.empty())
        v.invalid("packing_format", id, "Dense packing cannot name an opaque format");
      break;
    case PackingKind::Custom:
      if (packingFormat.empty() || packingValidationRef.empty())
        v.unresolved("packing_evidence_unknown", id,
                     "Custom packing needs format identity and validation reference");
      v.unresolved("packing_check_unavailable", id,
                   "No custom-format validation provider has been integrated");
      break;
    default:
      v.invalid("packing_invalid", id, "Invalid packing classification");
    }
  v.append(detail::validateStorageLayout(*this, value));
  if (!backendContext)
    v.unresolved("backend_context_unknown", id, "A backend memory context is required");
  else if (backendContext->backend.empty() || backendContext->version.empty() ||
           backendContext->abi.empty() || backendContext->context.empty())
    v.unresolved("backend_context_unknown", id,
                 "Backend, version, ABI and context must be explicit");
  try {
    if (stridesBytes && packing && *packing == PackingKind::Dense &&
        stridesBytes->size() != value.shape().size())
      v.invalid("representation_rank", id, "Stride rank mismatch");
    if (paddingElementsByAxis && paddingElementsByAxis->size() != value.shape().size())
      v.invalid("representation_rank", id, "Padding rank mismatch");
    // 即使 stride/padding 未声明，Dense 也至少需要保存全部逻辑元素。
    const auto minimum = logicalBytes(value.shape(), dtype);
    if (packing == PackingKind::Dense && actualBytes && minimum && *actualBytes < *minimum)
      v.invalid("representation_size", id, "Dense bytes cannot be smaller than logical storage");
    const auto required = requiredBytes(value);
    if (required && actualBytes && *actualBytes < *required)
      v.invalid("representation_size", id, "Actual bytes do not cover dtype/stride/padding span");
  } catch (const Error &e) {
    v.invalid(e.code(), id, e.what());
  }
  if (!quantparams)
    v.unresolved("quantparams_unknown", id, "No-quantization must also be explicit");
  else {
    const auto &q = *quantparams;
    if (q.kind != QuantizationKind::None && q.kind != QuantizationKind::PerTensor &&
        q.kind != QuantizationKind::PerAxis)
      v.invalid("quantparams_kind", id, "Invalid quantization kind");
    if (q.kind == QuantizationKind::None) {
      if (!q.scales.empty() || !q.zeroPoints.empty() || q.axis)
        v.invalid("quantparams_invalid", id,
                  "Unquantized representations cannot carry quantization parameters");
    } else {
      if (dtype != DataType::Int8 && dtype != DataType::Int32)
        v.invalid("quantparams_dtype", id,
                  "Quantization requires a supported integer storage dtype");
      std::size_t expected = 1;
      if (q.kind == QuantizationKind::PerAxis) {
        if (!q.axis || *q.axis >= value.shape().size())
          v.invalid("quantparams_axis", id, "Per-axis quantization needs a valid axis");
        else
          expected = static_cast<std::size_t>(value.shape()[*q.axis]);
      } else if (q.axis)
        v.invalid("quantparams_axis", id, "Per-tensor quantization has no axis");
      if (q.scales.size() != expected || q.zeroPoints.size() != expected || expected == 0)
        v.invalid("quantparams_count", id, "Scales/zero points must match the quantization domain");
      for (double scale : q.scales)
        if (!std::isfinite(scale) || scale <= 0.0)
          v.invalid("quantparams_scale", id, "Scales must be finite and positive");
      for (auto zero : q.zeroPoints)
        if ((dtype == DataType::Int8 && (zero < -128 || zero > 127)) ||
            (dtype == DataType::Int32 && (zero < std::numeric_limits<std::int32_t>::min() ||
                                          zero > std::numeric_limits<std::int32_t>::max())))
          v.invalid("quantparams_zero", id, "Zero point is outside storage dtype");
    }
  }
  return v;
}
std::optional<double> Handoff::serviceTimeUs() const {
  if (stages.empty())
    return std::nullopt;
  double sum = 0.0;
  for (const auto &stage : stages) {
    if (!stage.durationUs)
      return std::nullopt;
    require(finiteNonnegative(*stage.durationUs), "duration_invalid",
            "Invalid handoff stage duration");
    sum += *stage.durationUs;
    require(std::isfinite(sum), "duration_overflow", "Handoff duration overflow");
  }
  return sum;
}

Model::Model(Builder builder) : mData(std::move(builder)) {
  require(mData.dag && mData.platform, "model_context", "Model requires frozen DAG and Platform");
  throwInvalid(validate());
}
const Region &Model::region(const RegionId &id) const {
  const auto it = std::find_if(mData.regions.begin(), mData.regions.end(),
                               [&](const Region &r) { return r.id() == id; });
  require(it != mData.regions.end(), "region_missing", "Region not found: " + id);
  return *it;
}
const Partition &Model::partition(const PartitionId &id) const {
  const auto it = std::find_if(mData.partitions.begin(), mData.partitions.end(),
                               [&](const Partition &p) { return p.id() == id; });
  require(it != mData.partitions.end(), "partition_missing", "Partition not found: " + id);
  return *it;
}
const Representation &Model::representation(const RepresentationId &id) const {
  return lookup(mData.representations, id, "representation");
}
const Implementation &Model::implementation(const ImplementationId &id) const {
  return lookup(mData.implementations, id, "implementation");
}
const Candidate &Model::candidate(const CandidateId &id) const {
  return lookup(mData.candidates, id, "candidate");
}
const Cost &Model::cost(const CostId &id) const {
  return lookup(mData.costs, id, "cost");
}
const Handoff &Model::handoff(const HandoffId &id) const {
  return lookup(mData.handoffs, id, "handoff");
}
const Stage &Model::boundaryStage(const std::string &id) const {
  return lookup(mData.boundaryStages, id, "boundary_stage");
}
const Evidence &Model::evidence(const std::string &id) const {
  return lookup(mData.evidence, id, "evidence");
}
std::vector<const Candidate *> Model::candidatesFor(const RegionId &id) const {
  region(id);
  std::vector<const Candidate *> result;
  for (const auto &c : mData.candidates)
    if (c.region == id)
      result.push_back(&c);
  return result;
}
std::vector<const Candidate *> Model::eligibleCandidatesFor(const RegionId &id) const {
  // 登记池不等于生产设备域；状态、资源和证据条件必须同时完整成立。
  std::vector<const Candidate *> result;
  for (const auto *c : candidatesFor(id))
    if (c->support == Support::Supported && c->validation == ExecutionValidation::Verified &&
        validateCandidate(*c).valid())
      result.push_back(c);
  return result;
}
std::vector<DeviceId> Model::deviceDomain(const RegionId &id) const {
  std::set<DeviceId> result;
  for (const auto *c : eligibleCandidatesFor(id))
    result.insert(c->device);
  return {result.begin(), result.end()};
}

Validation Model::validateCandidate(const Candidate &c) const {
  Validation v;
  try {
    const auto &r = region(c.region);
    const auto &impl = implementation(c.implementation);
    v.append(validateExecution(c.implementation, c.device, c.id));
    v.append(validateArithmetic(c));
    if (dag().registryId().empty() || platform().registryId().empty())
      v.unresolved("registry_unknown", c.id,
                   "Graph and platform registry identities must be frozen");
    else if (dag().registryId() != platform().registryId())
      v.invalid("registry_mismatch", c.id, "Graph and platform use different registry snapshots");
    if (numericPolicy().id.empty())
      v.unresolved("numeric_policy_unknown", c.id, "Numeric policy identity is unknown");
    if (c.numericPolicyId != numericPolicy().id)
      v.invalid("candidate_numeric_policy", c.id, "Candidate numeric policy mismatch");
    if (c.config.id.empty() || c.config.nameSpace.empty() || c.config.version.empty())
      v.unresolved("execution_config_unknown", c.id,
                   "Execution configuration identity/schema is required");
    for (const auto &option : c.config.options)
      if (option.first.empty() || option.second.empty())
        v.invalid("execution_option", c.id, "Options must have explicit keys and values");
    // 下限由设备协议提供，实际用量仍来自该完整候选；不自动补资源。
    v.append(validateExecutionResourceUses(platform(), c.device, DeviceWork::Compute,
                                           c.config.resources, c.id));
    const auto ports = [&](const std::vector<PortRepresentation> &bindings,
                           const std::vector<TensorId> &expected) {
      if (bindings.size() != expected.size())
        v.invalid("candidate_ports", c.id, "Boundary ports must match derived region IO exactly");
      std::set<std::size_t> seen;
      for (const auto &binding : bindings) {
        if (binding.port >= expected.size() || !seen.insert(binding.port).second ||
            (binding.port < expected.size() && expected[binding.port] != binding.tensor))
          v.invalid("candidate_port", c.id, "Wrong tensor or repeated boundary port");
        const auto &tensor = dag().tensor(binding.tensor);
        const auto &rep = representation(binding.representation);
        if (rep.tensor != binding.tensor)
          v.invalid("candidate_representation", c.id, "Port representation binds another tensor");
        if (!platform().accessible(rep.pool, c.device))
          v.invalid("candidate_pool_access", c.id, "Device cannot access port pool");
        if (rep.backendContext &&
            (rep.backendContext->backend != impl.backend ||
             rep.backendContext->version != impl.backendVersion ||
             (!impl.environment.abi.empty() && rep.backendContext->abi != impl.environment.abi)))
          v.invalid("candidate_context", c.id,
                    "Port backend version/ABI does not match implementation");
        v.append(rep.validate(tensor, platform()));
        numericRepresentation(rep, tensor, numericPolicy(), c.id, v);
        if (tensor.kind() == ValueKind::Parameter &&
            (!tensor.parameter() || tensor.parameter()->contentDigest.empty()))
          v.unresolved("parameter_content_unknown", c.id,
                       "Parameter contents must have an immutable reference and digest");
        for (auto nodeId : r.members()) {
          const auto &node = dag().node(nodeId);
          if (!hasTensor(node.inputs(), binding.tensor) &&
              !hasTensor(node.outputs(), binding.tensor))
            continue;
          const auto &req = node.requirements();
          if (!req.allowedTypes.empty() && rep.dtype != DataType::Unknown &&
              std::find(req.allowedTypes.begin(), req.allowedTypes.end(), rep.dtype) ==
                  req.allowedTypes.end())
            v.invalid("operator_numeric_type", c.id,
                      "Representation violates an operator numeric requirement");
          const bool quantized = rep.quantparams && rep.quantparams->kind != QuantizationKind::None;
          if (((rep.dtype != DataType::Unknown && rep.dtype != tensor.dtype()) || quantized) &&
              !req.allowApproximation)
            v.invalid("operator_numeric_policy", c.id,
                      "Precision or same-dtype quantization cannot relax an operator requirement");
        }
      }
    };
    ports(c.inputs, r.inputs());
    ports(c.outputs, r.outputs());
    switch (c.support) {
    case Support::Supported:
    case Support::Unsupported:
      break;
    case Support::Unknown:
      v.unresolved("support_unknown", c.id, "Unknown support is not unsupported");
      break;
    default:
      v.invalid("support_invalid", c.id, "Invalid support state");
    }
    switch (c.search) {
    case SearchResult::Found:
    case SearchResult::BudgetExhausted:
    case SearchResult::NotSearched:
      break;
    default:
      v.invalid("search_invalid", c.id, "Invalid search state");
    }
    switch (c.validation) {
    case ExecutionValidation::Pending:
      v.unresolved("candidate_pending", c.id, "Execution validation is pending");
      break;
    case ExecutionValidation::Verified:
    case ExecutionValidation::Failed:
      break;
    default:
      v.invalid("execution_validation_invalid", c.id, "Invalid execution validation state");
    }
    // Verified 标签不能代替适用证据；Simulation/Analytical 不会通过此生产准入门槛。
    if (c.validation == ExecutionValidation::Verified) {
      bool passed = false;
      for (const auto &ref : c.evidenceRefs) {
        const auto &e = evidence(ref);
        if (e.objectId != c.id || e.platformId != platform().id() ||
            e.numericPolicyId != c.numericPolicyId || e.backendVersion != impl.backendVersion)
          v.invalid("candidate_evidence_conditions", c.id,
                    "Evidence belongs to different candidate/conditions");
        if (e.method.empty() || e.casesRef.empty() || e.observationRef.empty())
          v.unresolved("evidence_scope_unknown", c.id,
                       "Passing evidence needs method, input scope and observation reference");
        else if (e.result == EvidenceResult::Passed && e.kind == EvidenceKind::RealDevice)
          passed = true;
      }
      if (!passed)
        v.unresolved("candidate_evidence_unknown", c.id,
                     "Verified execution needs passing real-device evidence in the stated scope");
    }
    if (needsNumericAcceptance(c)) {
      bool accepted = false;
      for (const auto &ref : numericPolicy().validationRefs) {
        const auto &a = numericAcceptance(ref);
        if (std::find(a.candidates.begin(), a.candidates.end(), c.id) != a.candidates.end() &&
            validateNumericAcceptance(a).valid())
          accepted = true;
      }
      if (!accepted)
        v.unresolved("numeric_evidence_unknown", c.id,
                     "No approved whole-graph numerical combination contains this candidate");
    }
  } catch (const Error &e) {
    v.invalid(e.code(), c.id, e.what());
  }
  return v;
}
Validation Model::validateCost(const Cost &c) const {
  Validation v;
  try {
    const auto &q = candidate(c.candidate);
    if (c.platformId != platform().id() || c.boundaryId != boundary().id ||
        c.executionConfigId != q.config.id)
      v.invalid("cost_conditions", c.id,
                "Cost must bind the exact candidate configuration, platform and boundary");
    duration(c.serviceTimeUs, c.id, v);
    provenance(c.source, c.sourceRef, c.measurementRefs, c.id, v);
    if (c.measurementContext.empty())
      v.unresolved("measurement_context_unknown", c.id,
                   "Service-ready-to-region-complete context must be explicit");
    switch (c.statistic.kind) {
    case StatisticKind::Unknown:
      v.unresolved("statistic_unknown", c.id, "Target statistic must be explicit");
      break;
    case StatisticKind::Mean:
    case StatisticKind::Median:
    case StatisticKind::Quantile:
      break;
    default:
      v.invalid("statistic_invalid", c.id, "Invalid statistic kind");
    }
    if (c.statistic.kind == StatisticKind::Quantile) {
      if (!c.statistic.quantile)
        v.unresolved("quantile_unknown", c.id, "Quantile parameter is missing");
      else if (!std::isfinite(*c.statistic.quantile) || *c.statistic.quantile < 0 ||
               *c.statistic.quantile > 1)
        v.invalid("quantile_invalid", c.id, "Quantile must be in [0,1]");
    } else if (c.statistic.quantile)
      v.invalid("statistic_invalid", c.id, "Non-quantile statistic has a quantile parameter");
    if (c.source != CostSource::Measured && !c.samplesUs.empty())
      v.invalid("predicted_samples", c.id, "Only measured costs may carry observed samples");
    for (double sample : c.samplesUs)
      if (!finiteNonnegative(sample))
        v.invalid("sample_invalid", c.id, "Observed sample must be finite and nonnegative");
    workspace(platform(), c.workspaceComplete, c.workspaceBytesByPool, c.id, v);
    for (const auto &entry : c.workspaceBytesByPool)
      if (!platform().accessible(entry.first, q.device))
        v.invalid("workspace_pool_access", c.id, "Candidate device cannot access workspace pool");
    v.append(validateResourceUses(platform(), c.scopedResources, c.id, q.device));
    for (const auto &use : c.scopedResources)
      for (const auto &configUse : q.config.resources)
        if (use.resource == configUse.resource)
          v.invalid("resource_double_count", c.id,
                    "One resource must not be repeated in configuration and scoped costs");
    // 首版只有完整区域输出可见，不能用较早的 tile 就绪偏移引入跨区流式执行。
    const auto &outputs = region(q.region).outputs();
    for (auto t : outputs) {
      const auto offset = c.readyUsByTensor.find(t);
      if (offset == c.readyUsByTensor.end() || !offset->second)
        v.unresolved("output_ready_unknown", c.id, "Every full-region output needs a ready offset");
      else if (!finiteNonnegative(*offset->second) ||
               (c.serviceTimeUs && !near(*offset->second, *c.serviceTimeUs)))
        v.invalid("output_ready", c.id,
                  "First version only supports full-region-complete output visibility");
    }
    for (const auto &offset : c.readyUsByTensor)
      if (!hasTensor(outputs, offset.first))
        v.invalid("output_ready_tensor", c.id, "Ready offset is not a boundary output");
  } catch (const Error &e) {
    v.invalid(e.code(), c.id, e.what());
  }
  return v;
}
Validation Model::validateStage(const Stage &s) const {
  Validation v;
  try {
    const auto &from = representation(s.from);
    const auto &to = representation(s.to);
    if (s.id.empty())
      v.invalid("stage_identity", s.id, "Stage identity is required");
    switch (s.kind) {
    case StageKind::Identity:
    case StageKind::Transfer:
    case StageKind::Convert:
    case StageKind::Repack:
    case StageKind::Quantize:
    case StageKind::Dequantize:
    case StageKind::Synchronize:
      break;
    default:
      v.invalid("stage_kind", s.id, "Invalid adapter stage kind");
    }
    if (s.kind != StageKind::Identity) {
      if (!s.implementation || !s.executionDevice)
        v.unresolved("stage_execution_unknown", s.id,
                     "Adapter work needs an execution device, provider and executable artifact");
      else {
        v.append(validateExecution(*s.implementation, *s.executionDevice, s.id));
        if (s.kind != StageKind::Transfer &&
            (!platform().accessible(from.pool, *s.executionDevice) ||
             !platform().accessible(to.pool, *s.executionDevice)))
          v.invalid("stage_device_access", s.id,
                    "Local adapter device cannot access its endpoint pools");
        for (const auto &entry : s.workspaceBytesByPool)
          if (!platform().accessible(entry.first, *s.executionDevice))
            v.invalid("stage_workspace_access", s.id,
                      "Adapter device cannot access its workspace pool");
      }
    }
    if (from.tensor != to.tensor)
      v.invalid("stage_tensor", s.id, "A stage must preserve the logical value identity");
    v.append(from.validate(dag().tensor(from.tensor), platform()));
    v.append(to.validate(dag().tensor(to.tensor), platform()));
    numericRepresentation(from, dag().tensor(from.tensor), numericPolicy(), s.id, v);
    numericRepresentation(to, dag().tensor(to.tensor), numericPolicy(), s.id, v);
    duration(s.durationUs, s.id, v);
    provenance(s.source, s.sourceRef, s.measurementRefs, s.id, v);
    workspace(platform(), s.workspaceComplete, s.workspaceBytesByPool, s.id, v);
    const bool localCompute = s.kind == StageKind::Convert || s.kind == StageKind::Repack ||
                              s.kind == StageKind::Quantize || s.kind == StageKind::Dequantize;
    if (localCompute && s.executionDevice) {
      v.append(validateExecutionResourceUses(platform(), *s.executionDevice,
                                             DeviceWork::LocalConversion, s.resources, s.id));
    } else {
      // 复制仍检查明确 Link 资源；Identity/Synchronize 不凭类别强塞 Compute。
      v.append(validateResourceUses(platform(), s.resources, s.id, s.executionDevice));
    }
    if (needsNumericAcceptance(s))
      for (const auto &node : dag().nodes())
        if ((hasTensor(node->inputs(), from.tensor) || hasTensor(node->outputs(), from.tensor)) &&
            !node->requirements().allowApproximation)
          v.invalid("stage_operator_numeric_policy", s.id,
                    "An adapter cannot relax a touching operator's strict numerical requirements");
    if (!s.transferBytes)
      v.unresolved("transfer_bytes_unknown", s.id,
                   "Even a no-copy stage must explicitly declare zero bytes");
    const bool identical = sameRepresentation(from, to);
    if (s.kind == StageKind::Identity || s.kind == StageKind::Synchronize) {
      if (!identical)
        v.invalid("identity_representation", s.id,
                  "No-copy handoff requires identical physical representation and context");
      if (s.link || (s.transferBytes && *s.transferBytes != 0))
        v.invalid("identity_transfer", s.id, "Identity/synchronization does not move bytes");
      if (s.kind == StageKind::Identity && s.durationUs && *s.durationUs != 0.0)
        v.invalid("identity_duration", s.id,
                  "Nonzero synchronization must be an explicit synchronization stage");
    } else {
      if (s.resources.empty())
        v.unresolved("stage_resources_unknown", s.id,
                     "Transfers/conversions need explicit resources");
      if (s.kind == StageKind::Transfer) {
        if (!s.link)
          v.unresolved("stage_link_unknown", s.id, "Transfer needs an explicit directed link");
        else {
          const auto &link = platform().link(*s.link);
          if (link.sourcePool != from.pool || link.targetPool != to.pool)
            v.invalid("stage_link_direction", s.id,
                      "Directed link endpoints do not match stage pools");
          for (const auto &required : link.resources) {
            const auto use =
                std::find_if(s.resources.begin(), s.resources.end(), [&](const ResourceUse &u) {
                  return u.resource == required.resource && u.amount >= required.amount;
                });
            if (use == s.resources.end())
              v.invalid("stage_link_resource", s.id, "Stage omits a shared link resource");
          }
        }
        if (!sameRepresentation(from, to, false))
          v.invalid("transfer_conversion", s.id,
                    "A raw transfer cannot silently transform storage representation");
        if (s.transferBytes && from.actualBytes && *s.transferBytes != *from.actualBytes)
          v.invalid("transfer_bytes", s.id,
                    "Raw transfer bytes must equal the actual representation bytes");
        if (s.transferBytes && *s.transferBytes > 0 && s.durationUs && *s.durationUs == 0)
          v.invalid("transfer_duration", s.id,
                    "A nonempty copy cannot be assigned zero time by device/type identity");
      } else {
        if (from.pool != to.pool || s.link)
          v.invalid("conversion_location", s.id,
                    "Conversions are local; movement must be a separate stage");
        if (s.transferBytes && *s.transferBytes != 0)
          v.invalid("conversion_transfer", s.id,
                    "Local conversion is not separately charged as a copy");
        const bool quantChange = !(from.quantparams == to.quantparams);
        const bool typeChange = from.dtype != to.dtype;
        if ((quantChange || typeChange) && !numericPolicy().requirements.allowApproximation)
          v.invalid("stage_numeric_policy", s.id, "A stage cannot relax the numeric policy");
        if (quantChange && !numericPolicy().allowQuantization)
          v.invalid("stage_quantization_policy", s.id, "Quantization change was not allowed");
        if (s.kind == StageKind::Repack && (quantChange || typeChange))
          v.invalid("repack_numeric", s.id, "Repacking cannot change numeric semantics");
        if (s.kind == StageKind::Quantize &&
            (!to.quantparams || to.quantparams->kind == QuantizationKind::None))
          v.invalid("quantize_target", s.id,
                    "Quantize target must declare quantization parameters");
        if (s.kind == StageKind::Dequantize &&
            (!from.quantparams || from.quantparams->kind == QuantizationKind::None ||
             !to.quantparams || to.quantparams->kind != QuantizationKind::None))
          v.invalid("dequantize_endpoints", s.id,
                    "Dequantize needs a quantized source and unquantized target");
      }
    }
  } catch (const Error &e) {
    v.invalid(e.code(), s.id, e.what());
  }
  return v;
}
Validation Model::validateHandoff(const Handoff &h) const {
  Validation v;
  try {
    const auto &producer = candidate(h.producer);
    const auto &consumer = candidate(h.consumer);
    v.append(validateCandidate(producer));
    v.append(validateCandidate(consumer));
    switch (h.validation) {
    case ExecutionValidation::Pending:
    case ExecutionValidation::Verified:
    case ExecutionValidation::Failed:
      break;
    default:
      v.invalid("execution_validation_invalid", h.id, "Invalid handoff validation state");
    }
    const auto edge = std::find_if(dag().edges().begin(), dag().edges().end(),
                                   [&](const Edge &e) { return e.id == h.edge; });
    require(edge != dag().edges().end(), "handoff_edge", "Handoff edge does not exist");
    if (!contains(region(producer.region).members(), edge->producer) ||
        !contains(region(consumer.region).members(), edge->consumer) ||
        producer.region == consumer.region)
      v.invalid("handoff_candidates", h.id,
                "Handoff candidates must implement the two sides of this crossing edge");
    if (h.platformId != platform().id() || h.numericPolicyId != numericPolicy().id)
      v.invalid("handoff_conditions", h.id, "Handoff conditions mismatch");
    const auto out =
        std::find_if(producer.outputs.begin(), producer.outputs.end(),
                     [&](const PortRepresentation &p) { return p.tensor == edge->tensor; });
    const auto in =
        std::find_if(consumer.inputs.begin(), consumer.inputs.end(),
                     [&](const PortRepresentation &p) { return p.tensor == edge->tensor; });
    require(out != producer.outputs.end() && in != consumer.inputs.end(), "handoff_ports",
            "Edge must bind both candidate boundary ports");
    // 同设备/同表示也必须显式声明阶段与就绪依赖，不能按身份默认零成本交接。
    if (h.stages.empty())
      v.invalid("handoff_stages", h.id,
                "Identity still requires an explicit stage and readiness dependency");
    else {
      if (h.stages.front().from != out->representation || h.stages.back().to != in->representation)
        v.invalid("handoff_endpoints", h.id,
                  "Stage endpoints do not match the selected candidate representations");
      std::set<std::string> stages;
      for (std::size_t i = 0; i < h.stages.size(); ++i) {
        const auto &s = h.stages[i];
        if (!stages.insert(s.id).second)
          v.invalid("stage_duplicate", h.id, "Repeated stage identity");
        if (i && h.stages[i - 1].to != s.from)
          v.invalid("handoff_continuity", h.id, "Consecutive stage representations do not connect");
        if (representation(s.from).tensor != edge->tensor ||
            representation(s.to).tensor != edge->tensor)
          v.invalid("handoff_tensor", h.id, "Handoff stage transforms another edge value");
        v.append(validateStage(s));
      }
    }
    h.serviceTimeUs();
    if (h.validation == ExecutionValidation::Pending)
      v.unresolved("handoff_pending", h.id, "Handoff verification is pending");
    if (h.validation == ExecutionValidation::Verified) {
      bool passed = false;
      for (const auto &ref : h.evidenceRefs) {
        const auto &e = evidence(ref);
        if (e.objectId != h.id || e.platformId != h.platformId ||
            e.numericPolicyId != h.numericPolicyId)
          v.invalid("handoff_evidence_conditions", h.id,
                    "Evidence does not bind these handoff conditions");
        if (e.method.empty() || e.casesRef.empty() || e.observationRef.empty() ||
            e.backendVersion.empty())
          v.unresolved(
              "evidence_scope_unknown", h.id,
              "Passing evidence still needs method, conditions, scope and raw record reference");
        else if (e.result == EvidenceResult::Passed && e.kind == EvidenceKind::RealDevice)
          passed = true;
      }
      if (!passed)
        v.unresolved("handoff_evidence_unknown", h.id,
                     "Verified handoff requires passing scoped evidence");
    }
  } catch (const Error &e) {
    v.invalid(e.code(), h.id, e.what());
  }
  return v;
}
Validation Model::validate() const {
  Validation v;
  if (!mData.dag || !mData.platform) {
    v.invalid("model_context", mData.id, "Frozen graph/platform are required");
    return v;
  }
  if (id().empty())
    v.invalid("model_identity", id(), "Snapshot identity is required");
  try {
    dag().validate();
  } catch (const Error &e) {
    v.invalid(e.code(), id(), e.what());
  }
  v.append(platform().validate());
  v.append(boundary().validate(platform()));
  if (dag().registryId().empty() || platform().registryId().empty())
    v.unresolved("registry_unknown", id(), "Graph/platform registry identities must be explicit");
  else if (dag().registryId() != platform().registryId())
    v.invalid("registry_mismatch", id(), "Graph and platform use different registry snapshots");
  if (!dag().modelOutputs())
    v.unresolved("model_outputs_unknown", id(),
                 "Terminal-output fallback is not proof of complete model outputs");
  if (!dag().modelInputs())
    v.unresolved("model_inputs_unknown", id(),
                 "External dependencies do not establish ordered runtime inputs versus parameters");
  if (numericPolicy().id.empty())
    v.unresolved("numeric_policy_unknown", id(), "Explicit numeric policy identity is required");
  try {
    numericPolicy().requirements.validate();
  } catch (const Error &e) {
    v.invalid(e.code(), id(), e.what());
  }
  identities(mData.representations, "representation", v);
  identities(mData.implementations, "implementation", v);
  identities(mData.candidates, "candidate", v);
  identities(mData.costs, "cost", v);
  identities(mData.handoffs, "handoff", v);
  identities(mData.boundaryStages, "boundary stage", v);
  identities(mData.evidence, "evidence", v);
  identities(mData.numericAcceptances, "numeric acceptance", v);
  for (const auto &a : mData.numericAcceptances)
    v.append(validateNumericAcceptance(a));
  std::set<std::string> acceptedRefs;
  for (const auto &ref : numericPolicy().validationRefs) {
    if (!acceptedRefs.insert(ref).second)
      v.invalid("numeric_acceptance_duplicate", id(), "Repeated numerical acceptance reference");
    try {
      (void)numericAcceptance(ref);
    } catch (const Error &e) {
      v.invalid(e.code(), id(), e.what());
    }
  }
  std::set<RegionId> regionsSeen;
  for (const auto &r : mData.regions) {
    if (!regionsSeen.insert(r.id()).second)
      v.invalid("region_duplicate", r.id(), "Repeated region identity");
    if (r.graphId() != dag().id()) {
      v.invalid("region_graph", r.id(), "Wrong graph");
      continue;
    }
    try {
      const Region expected(r.id(), dag(), r.members());
      if (expected.inputs() != r.inputs() || expected.outputs() != r.outputs())
        v.invalid("region_boundary", r.id(), "Region boundary is stale");
    } catch (const Error &e) {
      v.invalid(e.code(), r.id(), e.what());
    }
    if (candidatesFor(r.id()).empty())
      v.unresolved("candidate_pool_empty", r.id(),
                   "No registered candidate; this is not a proof of global infeasibility");
  }
  std::set<PartitionId> partitionsSeen;
  for (const auto &p : mData.partitions) {
    if (!partitionsSeen.insert(p.id()).second)
      v.invalid("partition_duplicate", p.id(), "Repeated partition identity");
    try {
      if (p.graphId() != dag().id())
        v.invalid("partition_graph", p.id(), "Wrong graph");
      std::vector<Region> rs;
      for (const auto &r : p.regions())
        rs.push_back(region(r));
      Partition checked(p.id(), dag(), rs);
    } catch (const Error &e) {
      v.invalid(e.code(), p.id(), e.what());
    }
  }
  for (const auto &r : mData.representations) {
    try {
      v.append(r.validate(dag().tensor(r.tensor), platform()));
    } catch (const Error &e) {
      v.invalid(e.code(), r.id, e.what());
    }
  }
  for (const auto &impl : mData.implementations)
    v.append(validateImplementation(impl));
  for (const auto &e : mData.evidence) {
    switch (e.kind) {
    case EvidenceKind::RealDevice:
    case EvidenceKind::Simulation:
    case EvidenceKind::Analytical:
      break;
    default:
      v.invalid("evidence_kind", e.id, "Invalid evidence kind");
    }
    switch (e.result) {
    case EvidenceResult::Passed:
    case EvidenceResult::Failed:
    case EvidenceResult::Unknown:
      break;
    default:
      v.invalid("evidence_result", e.id, "Invalid evidence result");
    }
    if (e.objectId.empty() || e.platformId.empty() || e.backendVersion.empty() ||
        e.method.empty() || e.casesRef.empty() || e.numericPolicyId.empty() ||
        e.observationRef.empty())
      v.unresolved("evidence_scope_unknown", e.id,
                   "Evidence needs object, conditions, method, test scope, numeric policy and raw "
                   "record reference");
    if (e.platformId != platform().id() || e.numericPolicyId != numericPolicy().id)
      v.invalid("evidence_conditions", e.id, "Evidence conditions do not match this snapshot");
    const bool subject = std::any_of(mData.candidates.begin(), mData.candidates.end(),
                                     [&](const Candidate &c) { return c.id == e.objectId; }) ||
                         std::any_of(mData.handoffs.begin(), mData.handoffs.end(),
                                     [&](const Handoff &h) { return h.id == e.objectId; });
    if (!subject)
      v.invalid("evidence_subject", e.id,
                "Evidence must refer to a candidate or handoff in the snapshot");
  }
  for (const auto &c : mData.candidates)
    v.append(validateCandidate(c));
  for (const auto &c : mData.costs)
    v.append(validateCost(c));
  for (const auto &h : mData.handoffs)
    v.append(validateHandoff(h));
  for (const auto &s : mData.boundaryStages) {
    v.append(validateStage(s));
    try {
      const auto t = representation(s.from).tensor;
      const bool external = hasTensor(dag().externalInputs(), t);
      const bool output = dag().modelOutputs() && hasTensor(*dag().modelOutputs(), t);
      if (!external && !output) {
        if (!dag().modelOutputs())
          v.unresolved("boundary_stage_scope_unknown", s.id,
                       "Explicit model outputs are needed to verify output-stage scope");
        else
          v.invalid("boundary_stage_tensor", s.id,
                    "Boundary stages are only for external inputs or model outputs");
      }
    } catch (const Error &e) {
      v.invalid(e.code(), s.id, e.what());
    }
  }
  std::set<TensorId> resident, requested;
  for (const auto &input : boundary().inputs) {
    try {
      dag().tensor(input.tensor);
      if (!hasTensor(dag().externalInputs(), input.tensor))
        v.invalid("boundary_input_tensor", input.id,
                  "Initial residency must be an external value, not a computed result");
      const auto &r = representation(input.representation);
      if (r.tensor != input.tensor)
        v.invalid("boundary_input_representation", input.id, "Wrong logical value");
      numericRepresentation(r, dag().tensor(input.tensor), numericPolicy(), input.id, v);
      resident.insert(input.tensor);
    } catch (const Error &e) {
      v.invalid(e.code(), input.id, e.what());
    }
  }
  for (auto t : dag().externalInputs())
    if (!resident.count(t))
      v.unresolved("input_residency_unknown", std::to_string(t),
                   "Every input/parameter needs actual residency");
  for (const auto &output : boundary().outputs) {
    try {
      dag().tensor(output.tensor);
      if (dag().modelOutputs() && !hasTensor(*dag().modelOutputs(), output.tensor))
        v.invalid("boundary_output_tensor", output.id, "Target is not a declared model output");
      const auto &r = representation(output.representation);
      if (r.tensor != output.tensor)
        v.invalid("boundary_output_representation", output.id, "Wrong logical value");
      numericRepresentation(r, dag().tensor(output.tensor), numericPolicy(), output.id, v);
      requested.insert(output.tensor);
    } catch (const Error &e) {
      v.invalid(e.code(), output.id, e.what());
    }
  }
  if (dag().modelOutputs())
    for (auto t : *dag().modelOutputs())
      if (!requested.count(t))
        v.unresolved("output_target_unknown", std::to_string(t),
                     "Every model output needs an explicit target");
  return v;
}

} // namespace opfusion
