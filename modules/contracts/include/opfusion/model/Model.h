#pragma once

// 区域、候选、表示、条件成本及证据的不可变模型，不负责搜索或设备执行。
#include "opfusion/graph/DAG.h"
#include "opfusion/platform/Boundary.h"
#include "opfusion/platform/Platform.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opfusion {

using RegionId = std::string;
using PartitionId = std::string;
using RepresentationId = std::string;
using ImplementationId = std::string;
using CandidateId = std::string;
using CostId = std::string;
using HandoffId = std::string;

/// 原图节点的非空区域；接口从完整图推导，保留所有外部需要的输出。
class Region {
 public:
  /// 同步借用 graph 计算边界；未知/重复成员抛出 Error，不保留图引用。
  Region(RegionId id, const DAG& graph, std::vector<OperatorId> members);
  /// 区域身份。
  const RegionId& id() const noexcept { return mId; }
  /// 所属不可变图的身份，不能跨图拼接区域。
  const std::string& graphId() const noexcept { return mGraphId; }
  /// 借用保存的成员表；不是设备或复制任务。
  const std::vector<OperatorId>& members() const noexcept { return mMembers; }
  /// 借用从图推导的区域输入值身份。
  const std::vector<TensorId>& inputs() const noexcept { return mInputs; }
  /// 借用完整区域输出；包含区域外消费者和模型交付所需值。
  const std::vector<TensorId>& outputs() const noexcept { return mOutputs; }
 private:
  RegionId mId;
  std::string mGraphId;
  std::vector<OperatorId> mMembers;
  std::vector<TensorId> mInputs, mOutputs;
};

/// 覆盖原图且区域互斥的完整划分；商图必须无环，不执行划分搜索。
class Partition {
 public:
  /// 同步借用图/区域核验覆盖、互斥和商图；违例抛出 Error。
  Partition(PartitionId id, const DAG& graph, const std::vector<Region>& regions);
  /// 划分身份。
  const PartitionId& id() const noexcept { return mId; }
  /// 所属原图身份。
  const std::string& graphId() const noexcept { return mGraphId; }
  /// 借用已核验的区域身份顺序，不持有外部可变区域引用。
  const std::vector<RegionId>& regions() const noexcept { return mRegions; }
 private:
  PartitionId mId;
  std::string mGraphId;
  std::vector<RegionId> mRegions;
};

/// 物理表示依赖的后端版本、ABI 和上下文身份，不保存 SDK 活对象。
struct BackendContext {
  std::string backend;
  std::string version;
  std::string abi;
  std::string context;
};
/// 精确比较声明的全部上下文字段。
bool operator==(const BackendContext& a, const BackendContext& b);
/// 量化类型；None 与量化信息未知不是同一种状态。
enum class QuantizationKind { None, PerTensor, PerAxis };
/// 显式量化参数；axis 为逻辑轴索引，参数数量须与该轴匹配。
struct Quantization {
  QuantizationKind kind = QuantizationKind::None;
  std::vector<double> scales;
  std::vector<std::int64_t> zeroPoints;
  std::optional<std::size_t> axis;
};
/// 精确比较声明的量化参数，不进行近似数值验收。
bool operator==(const Quantization& a, const Quantization& b);
/// 连续布局与提供者自定义打包分别处理，后者必须带格式和校验依据。
enum class PackingKind { Dense, Custom };

/// 候选的物理表示，不是逻辑 Tensor，也不是已经分配的运行时缓冲。
struct Representation {
  RepresentationId id;
  TensorId tensor = 0;
  DataType dtype = DataType::Unknown;
  std::optional<Layout> layout;
  std::optional<std::vector<Bytes>> stridesBytes;
  /// 单位为元素，不能与 stridesBytes 或实际字节占用混用。
  std::optional<std::vector<Bytes>> paddingElementsByAxis;
  std::optional<Bytes> alignmentBytes;
  std::optional<PackingKind> packing;
  /// 仅 Custom 使用的开放格式身份；不是任意私有字段袋。
  std::string packingFormat;
  std::string packingValidationRef;
  std::optional<Quantization> quantparams;
  std::optional<Bytes> actualBytes;
  PoolId pool;
  std::optional<BackendContext> backendContext;
  /// 计算给定逻辑值在声明布局中的最低字节数；未知保持 nullopt，溢出抛出 Error。
  std::optional<Bytes> requiredBytes(const Tensor& tensorValue) const;
  /// 同步借用逻辑值和平台核验大小、布局、量化与池；不分配内存。
  Validation validate(const Tensor& tensorValue, const Platform& platform) const;
};
/// 比较表示声明；includeLocation=false 仅忽略位置/后端上下文，用于纯复制检查。
bool sameRepresentation(const Representation& a, const Representation& b,
                         bool includeLocation = true);

/// 具名数值政策；局部精度变化不能代替精确组合的整图验收。
struct NumericPolicy {
  std::string id;
  NumericalRequirements requirements;
  bool allowQuantization = false;
  /// 引用 NumericAcceptance，而非任意非空字符串见证。
  std::vector<std::string> validationRefs;
};
/// 提供者的独立版本和内容身份，不与后端版本混为一项。
struct ProviderIdentity { std::string id; std::string version; std::string digest; };
/// 制品所要求的 ABI、运行时与可选驱动版本。
struct EnvironmentRequirements {
  std::string abi;
  std::string runtimeVersion;
  std::optional<std::string> driverVersion;
};
/// 已登记实现的声明；引用制品不证明其已加载、执行或数值通过。
struct Implementation {
  ImplementationId id;
  std::string registryId;
  std::string backend;
  std::string backendVersion;
  ProviderIdentity provider;
  std::optional<std::string> compilerVersion;
  std::optional<std::string> programRef;
  std::optional<std::string> artifactRef;
  std::optional<std::string> structure;
  EnvironmentRequirements environment;
};
/// 精确政策与降低精度的执行模式分开，未知 dtype 仍不能视为默认精度。
enum class ArithmeticMode { Strict, ReducedPrecision };
/// 候选计算及累加精度，不从输出 dtype 反推真实执行模式。
struct Arithmetic {
  DataType computeType = DataType::Unknown;
  std::optional<DataType> accumulationType;
  ArithmeticMode mode = ArithmeticMode::Strict;
};
/// 版本化执行配置与实际资源请求；设备规则下限不能代替此完整记录。
struct ExecutionConfig {
  std::string id;
  std::string nameSpace;
  std::string version;
  std::map<std::string, std::string> options;
  std::vector<ResourceUse> resources;
  std::optional<Arithmetic> arithmetic;
};
/// 区域边界端口绑定；保留逻辑值与物理表示的独立身份。
struct PortRepresentation {
  std::size_t port = 0;
  TensorId tensor = 0;
  RepresentationId representation;
};
/// 未知支持与明确不支持分开。
enum class Support { Supported, Unsupported, Unknown };
/// 编译、执行验证和搜索结果是不同维度。
enum class ExecutionValidation { Pending, Verified, Failed };
/// 搜索耗尽/尚未搜索不构成不存在候选的证明。
enum class SearchResult { Found, BudgetExhausted, NotSearched };
/// 证据来源不能互相升级，模拟/分析不构成真实设备验证。
enum class EvidenceKind { RealDevice, Simulation, Analytical };
/// 证据的声明结果，不是校验器对原始观测真实性的背书。
enum class EvidenceResult { Passed, Failed, Unknown };
/// 绑定对象、平台、数值政策和输入范围的证据引用；不保存原始测量数据。
struct Evidence {
  std::string id;
  std::string objectId;
  std::string platformId;
  std::string backendVersion;
  std::string method;
  std::string casesRef;
  std::string numericPolicyId;
  EvidenceKind kind = EvidenceKind::Simulation;
  EvidenceResult result = EvidenceResult::Unknown;
  std::string observationRef;
};
/// 一个完整可选项：设备、实现、接口和配置不可从不同候选中拼接。
struct Candidate {
  CandidateId id;
  RegionId region;
  DeviceId device;
  ImplementationId implementation;
  ExecutionConfig config;
  std::string numericPolicyId;
  std::vector<PortRepresentation> inputs;
  std::vector<PortRepresentation> outputs;
  Support support = Support::Unknown;
  ExecutionValidation validation = ExecutionValidation::Pending;
  SearchResult search = SearchResult::NotSearched;
  std::vector<std::string> evidenceRefs;
};
/// 成本来源与目标统计量独立，预测值不能伪装成测量样本。
enum class CostSource { Analytical, Predicted, Measured };
/// 未指定统计口径不能默认解释为均值。
enum class StatisticKind { Unknown, Mean, Median, Quantile };
/// Quantile 必须显式指定 [0,1] 分位，其他统计量不能携带分位参数。
struct Statistic { StatisticKind kind = StatisticKind::Unknown; std::optional<double> quantile; };
/// 绑定精确候选与部署条件的成本；微秒和字节分别记账，缺失不是零。
struct Cost {
  CostId id;
  CandidateId candidate;
  std::string platformId;
  std::string boundaryId;
  std::string executionConfigId;
  CostSource source = CostSource::Analytical;
  std::string sourceRef;
  std::optional<double> serviceTimeUs;
  Statistic statistic;
  std::string measurementContext;
  std::vector<double> samplesUs;
  std::vector<std::string> measurementRefs;
  bool workspaceComplete = false;
  std::map<PoolId, std::optional<Bytes>> workspaceBytesByPool;
  /// 与配置请求不能重复；Thread/Block 局部资源不当作整任务并发活动。
  std::vector<ResourceUse> scopedResources;
  /// 首版仅接受完整区域完成时输出就绪，不表达跨区域流式消费。
  std::map<TensorId, std::optional<double>> readyUsByTensor;
  std::optional<std::string> uncertainty;
};
/// 适配工作类别；本地转换与池间复制必须拆分声明。
enum class StageKind { Identity, Transfer, Convert, Repack, Quantize, Dequantize, Synchronize };
/// 一个实际适配阶段；成本、工作区、实现和执行设备均显式绑定。
struct Stage {
  std::string id;
  StageKind kind = StageKind::Identity;
  RepresentationId from;
  RepresentationId to;
  std::optional<Bytes> transferBytes;
  std::optional<LinkId> link;
  std::vector<ResourceUse> resources;
  std::optional<double> durationUs;
  CostSource source = CostSource::Analytical;
  std::string sourceRef;
  std::vector<std::string> measurementRefs;
  bool workspaceComplete = false;
  std::map<PoolId, std::optional<Bytes>> workspaceBytesByPool;
  /// 非 Identity 工作通过 Implementation 引用提供者/制品，不保存执行句柄。
  std::optional<ImplementationId> implementation;
  std::optional<DeviceId> executionDevice;
};
/// 原始边两端精确候选的交接见证；阶段串行服务时间不包含中途排队。
struct Handoff {
  HandoffId id;
  EdgeId edge = 0;
  CandidateId producer;
  CandidateId consumer;
  std::string platformId;
  std::string numericPolicyId;
  ExecutionValidation validation = ExecutionValidation::Pending;
  std::vector<std::string> evidenceRefs;
  std::vector<Stage> stages;
  /// 累加已知阶段微秒数；任一未知保持 nullopt，不把跨设备身份当零成本。
  std::optional<double> serviceTimeUs() const;
};
/// 精确划分/候选/交接/边界适配组合的整图数值验收声明。
struct NumericAcceptance {
  std::string id;
  std::string graphId;
  PartitionId partition;
  std::string platformId;
  std::string boundaryId;
  std::string numericPolicyId;
  std::vector<CandidateId> candidates;
  std::vector<HandoffId> handoffs;
  std::vector<std::string> boundaryStages;
  EvidenceKind kind = EvidenceKind::Simulation;
  EvidenceResult result = EvidenceResult::Unknown;
  std::string method;
  std::string casesRef;
  std::string observationRef;
  std::optional<double> absoluteError;
  std::optional<double> relativeError;
};

/// 拥有候选等值记录并共享只读图/平台的快照；不预测、搜索或执行。
/// 查询引用借用自本实例；变更应发布新模型，不能覆盖被 Plan 引用的记录。
class Model {
 public:
  /// 组装输入；构造 Model 时按值接管，外部再改 builder 不回写快照。
  struct Builder {
    std::string id;
    std::shared_ptr<const DAG> dag;
    std::shared_ptr<const Platform> platform;
    Boundary boundary;
    NumericPolicy numericPolicy;
    std::vector<Region> regions;
    std::vector<Partition> partitions;
    std::vector<Representation> representations;
    std::vector<Implementation> implementations;
    std::vector<Candidate> candidates;
    std::vector<Cost> costs;
    std::vector<Handoff> handoffs;
    std::vector<Stage> boundaryStages;
    std::vector<Evidence> evidence;
    std::vector<NumericAcceptance> numericAcceptances;
  };
  /// 接管记录并拒绝已知非法输入；未知条件保留，不升级成可执行性。
  explicit Model(Builder builder);
  Model(const Model&) = default;
  Model& operator=(const Model&) = delete;
  /// 模型身份。
  const std::string& id() const noexcept { return mData.id; }
  /// 借用共享只读语义图。
  const DAG& dag() const noexcept { return *mData.dag; }
  /// 借用共享只读平台快照。
  const Platform& platform() const noexcept { return *mData.platform; }
  /// 借用本次部署边界。
  const Boundary& boundary() const noexcept { return mData.boundary; }
  /// 借用具名数值政策。
  const NumericPolicy& numericPolicy() const noexcept { return mData.numericPolicy; }
  /// 借用区域目录。
  const std::vector<Region>& regions() const noexcept { return mData.regions; }
  /// 借用完整划分目录。
  const std::vector<Partition>& partitions() const noexcept { return mData.partitions; }
  /// 借用物理表示目录，不包含实际运行缓冲。
  const std::vector<Representation>& representations() const noexcept {
    return mData.representations;
  }
  /// 借用实现声明目录，不加载制品。
  const std::vector<Implementation>& implementations() const noexcept {
    return mData.implementations;
  }
  /// 借用全部候选，包含尚未获准执行的候选。
  const std::vector<Candidate>& candidates() const noexcept { return mData.candidates; }
  /// 借用条件成本目录。
  const std::vector<Cost>& costs() const noexcept { return mData.costs; }
  /// 借用精确候选间交接目录。
  const std::vector<Handoff>& handoffs() const noexcept { return mData.handoffs; }
  /// 借用输入/交付适配阶段。
  const std::vector<Stage>& boundaryStages() const noexcept { return mData.boundaryStages; }
  /// 借用证据声明，不访问原始观测。
  const std::vector<Evidence>& evidence() const noexcept { return mData.evidence; }
  /// 借用组合数值验收记录。
  const std::vector<NumericAcceptance>& numericAcceptances() const noexcept {
    return mData.numericAcceptances;
  }
  /// 按身份借用区域，缺失抛出 Error。
  const Region& region(const RegionId& id) const;
  /// 按身份借用完整划分，缺失抛出 Error。
  const Partition& partition(const PartitionId& id) const;
  /// 按身份借用表示，缺失抛出 Error。
  const Representation& representation(const RepresentationId& id) const;
  /// 按身份借用实现，缺失抛出 Error。
  const Implementation& implementation(const ImplementationId& id) const;
  /// 按身份借用候选，缺失抛出 Error。
  const Candidate& candidate(const CandidateId& id) const;
  /// 按身份借用成本，缺失抛出 Error。
  const Cost& cost(const CostId& id) const;
  /// 按身份借用交接，缺失抛出 Error。
  const Handoff& handoff(const HandoffId& id) const;
  /// 按身份借用边界阶段，缺失抛出 Error。
  const Stage& boundaryStage(const std::string& id) const;
  /// 按身份借用证据声明，缺失抛出 Error。
  const Evidence& evidence(const std::string& id) const;
  /// 按身份借用数值验收声明，缺失抛出 Error。
  const NumericAcceptance& numericAcceptance(const std::string& id) const;
  /// 返回指定区域的全部候选借用指针，生命周期受本模型约束。
  std::vector<const Candidate*> candidatesFor(const RegionId& region) const;
  /// 仅返回 Supported/Verified 且候选校验为 Valid 的条目，不执行验证任务。
  std::vector<const Candidate*> eligibleCandidatesFor(const RegionId& region) const;
  /// 从完整合格候选推导设备域；搜索耗尽不能证明物理设备不支持。
  std::vector<DeviceId> deviceDomain(const RegionId& region) const;
  /// 核对全部记录；未使用候选的未知项不会自动构成所选 Plan 的非法条件。
  Validation validate() const;
  /// 核验精确候选接口、设备资源、环境与证据，不自动补候选或资源请求。
  Validation validateCandidate(const Candidate& candidate) const;
  /// 核验成本绑定、来源、工作区和输出就绪偏移，未知成本不是零。
  Validation validateCost(const Cost& cost) const;
  /// 核验适配阶段；本地计算转换使用设备规则，原始传输保留 Link 约束。
  Validation validateStage(const Stage& stage) const;
  /// 核验原始边、两端候选、阶段链及交接证据。
  Validation validateHandoff(const Handoff& handoff) const;
  /// 核验实现/制品/环境身份，仅检查声明，不加载或执行制品。
  Validation validateImplementation(const Implementation& implementation) const;
  /// 核验给定设备的后端/ABI/运行时/驱动条件，不探测环境。
  Validation validateExecution(const ImplementationId& implementation, const DeviceId& device,
                               const std::string& object) const;
  /// 核验整图组合范围、来源和数值容差，不计算真实输出误差。
  Validation validateNumericAcceptance(const NumericAcceptance& acceptance) const;
  /// 判断候选精度声明是否需要额外组合验收。
  bool needsNumericAcceptance(const Candidate& candidate) const;
  /// 判断阶段表示变换是否需要额外组合验收。
  bool needsNumericAcceptance(const Stage& stage) const;
 private:
  Validation validateArithmetic(const Candidate& candidate) const;
  Builder mData;
};

}  // namespace opfusion
