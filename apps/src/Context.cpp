// 部署 JSON 只在 apps 组装公共对象；声明不等于能力、分配或执行证据。
#include "Context.h"

#include "opfusion/common/Error.h"
#include "opfusion/model/Model.h"
#include "opfusion/platform/Boundary.h"
#include "opfusion/platform/CpuDevice.h"
#include "opfusion/platform/DeclaredDevice.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <type_traits>
namespace opfusion::app {
namespace {
using nlohmann::json;
void fields(const json &obj, std::initializer_list<const char *> keys) {
  require(obj.is_object(), "context_invalid", "context records must be objects");
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    require(std::any_of(keys.begin(), keys.end(), [&](const char *key) { return it.key() == key; }),
            "context_invalid", "unknown context field: " + it.key());
  }
}
std::string text(const json &obj, const char *key) {
  require(obj.contains(key) && obj.at(key).is_string(), "context_invalid",
          std::string("missing string: ") + key);
  auto value = obj.at(key).get<std::string>();
  require(!value.empty(), "context_invalid", std::string("empty identity: ") + key);
  return value;
}
template <class T> std::optional<T> optional(const json &obj, const char *key) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    return std::nullopt;
  }
  if constexpr (std::is_unsigned_v<T>) {
    require(obj.at(key).is_number_unsigned() ||
                (obj.at(key).is_number_integer() && obj.at(key).get<std::int64_t>() >= 0),
            "context_invalid", "byte count must be nonnegative integer");
  }
  return obj.at(key).get<T>();
}
template <class E> E choice(const std::string &value, const std::map<std::string, E> &choices) {
  const auto it = choices.find(value);
  require(it != choices.end(), "context_invalid", "unknown enum: " + value);
  return it->second;
}
void check(const Validation &v) {
  for (const auto &i : v.issues) {
    require(i.kind != IssueKind::Invalid, i.code, i.message);
  }
}
json issues(const Validation &v) {
  json result = json::array();
  for (const auto &i : v.issues) {
    result.push_back({{"code", i.code},
                      {"object", i.object},
                      {"message", i.message},
                      {"kind", i.kind == IssueKind::Invalid ? "invalid" : "unresolved"}});
  }
  return result;
}
} // namespace
json inspectContext(const std::string &path, const frontend::ImportResult &model) {
  std::ifstream input(path);
  require(bool(input), "context_file_open", "cannot open context file");
  json doc;
  input >> doc;
  fields(doc, {"schema", "id", "platform", "representations", "inputs", "outputs",
               "available_bytes_by_pool", "available_resources", "timing"});
  require(text(doc, "schema") == "opfusion-context/1", "context_invalid", "unknown context schema");
  const auto &p = doc.at("platform");
  fields(p, {"id", "devices", "pools", "resources", "links"});
  std::vector<std::unique_ptr<const Device>> devices;
  std::vector<MemoryPool> pools;
  std::vector<Resource> resources;
  std::vector<Link> links;
  for (const auto &d : p.at("devices")) {
    fields(d, {"id", "node_id", "kind", "architecture", "instance", "configuration", "backends",
               "cpu_allocation", "environments"});
    DeviceDescription x;
    x.id = text(d, "id");
    x.nodeId = text(d, "node_id");
    x.kind = choice<DeviceKind>(text(d, "kind"), {{"cpu", DeviceKind::Cpu},
                                                  {"gpu", DeviceKind::Gpu},
                                                  {"accelerator", DeviceKind::Accelerator},
                                                  {"other", DeviceKind::Other},
                                                  {"unknown", DeviceKind::Unknown}});
    x.architecture = d.value("architecture", std::string{});
    x.instance = d.value("instance", std::string{});
    x.configuration = d.value("configuration", std::string{});
    x.backends = d.value("backends", std::vector<std::string>{});
    if (d.contains("cpu_allocation")) {
      const auto &c = d.at("cpu_allocation");
      fields(c, {"quota_cores", "affinity", "numa_binding", "numa_node"});
      CpuAllocation allocation;
      allocation.quotaCores = optional<double>(c, "quota_cores");
      allocation.affinity = optional<std::vector<std::uint32_t>>(c, "affinity");
      allocation.numaNode = optional<std::uint32_t>(c, "numa_node");
      allocation.numaBinding = choice<NumaBinding>(c.value("numa_binding", std::string("unknown")),
                                                   {{"unknown", NumaBinding::Unknown},
                                                    {"not_applicable", NumaBinding::NotApplicable},
                                                    {"node", NumaBinding::Node}});
      x.cpuAllocation = std::move(allocation);
    }
    for (const auto &e : d.value("environments", json::array())) {
      fields(e,
             {"backend", "version", "abi", "runtime_version", "requires_driver", "driver_version"});
      BackendEnvironment environment;
      environment.backend = text(e, "backend");
      environment.version = text(e, "version");
      environment.abi = text(e, "abi");
      environment.runtimeVersion = text(e, "runtime_version");
      environment.requiresDriver = optional<bool>(e, "requires_driver");
      environment.driverVersion = optional<std::string>(e, "driver_version");
      x.environments.push_back(std::move(environment));
    }
    if (x.kind == DeviceKind::Cpu) {
      devices.push_back(std::make_unique<CpuDevice>(std::move(x)));
    } else {
      devices.push_back(std::make_unique<DeclaredDevice>(std::move(x)));
    }
  }
  for (const auto &pool : p.at("pools")) {
    fields(pool, {"id", "physical_identity", "physical_capacity_bytes", "accessible_devices"});
    pools.push_back({text(pool, "id"), text(pool, "physical_identity"),
                     optional<Bytes>(pool, "physical_capacity_bytes"),
                     pool.at("accessible_devices").get<std::vector<std::string>>()});
  }
  for (const auto &item : p.value("resources", json::array())) {
    fields(item, {"id", "kind", "scope", "unit", "capacity", "device", "pool", "shared_devices",
                  "sharing_ref"});
    Resource r;
    r.id = text(item, "id");
    r.unit = text(item, "unit");
    r.kind = choice<ResourceKind>(text(item, "kind"), {{"compute", ResourceKind::Compute},
                                                       {"copy", ResourceKind::Copy},
                                                       {"link", ResourceKind::Link},
                                                       {"threads", ResourceKind::Threads},
                                                       {"registers", ResourceKind::Registers},
                                                       {"on_chip", ResourceKind::OnChip},
                                                       {"other", ResourceKind::Other}});
    r.scope = choice<ResourceScope>(text(item, "scope"), {{"device", ResourceScope::Device},
                                                          {"pool", ResourceScope::Pool},
                                                          {"link", ResourceScope::Link},
                                                          {"thread", ResourceScope::Thread},
                                                          {"block", ResourceScope::Block}});
    r.capacity = optional<double>(item, "capacity");
    r.device = optional<std::string>(item, "device");
    r.pool = optional<std::string>(item, "pool");
    r.sharedDevices = item.value("shared_devices", std::vector<std::string>{});
    r.sharingRef = item.value("sharing_ref", std::string{});
    resources.push_back(std::move(r));
  }
  for (const auto &item : p.value("links", json::array())) {
    fields(item, {"id", "source_pool", "target_pool", "resources", "bandwidth_bytes_per_us",
                  "startup_us"});
    Link link;
    link.id = text(item, "id");
    link.sourcePool = text(item, "source_pool");
    link.targetPool = text(item, "target_pool");
    link.bandwidthBytesPerUs = optional<double>(item, "bandwidth_bytes_per_us");
    link.startupUs = optional<double>(item, "startup_us");
    for (const auto &use : item.value("resources", json::array())) {
      fields(use, {"resource", "amount"});
      link.resources.push_back({text(use, "resource"), use.at("amount").get<double>()});
    }
    links.push_back(std::move(link));
  }
  Platform platform(text(p, "id"), model.graph->registryId(), std::move(devices), std::move(pools),
                    std::move(resources), std::move(links));
  Validation validation = platform.validate();
  Boundary boundary;
  boundary.id = text(doc, "id");
  boundary.platformId = platform.id();
  std::map<std::string, TensorId> names;
  for (const auto &t : model.graph->tensors()) {
    require(names.emplace(t.name(), t.id()).second, "context_tensor_ambiguous",
            "tensor names are not unique");
  }
  auto tensor = [&](const json &j) {
    const auto name = text(j, "tensor");
    require(names.count(name), "context_tensor_missing", "unknown tensor name: " + name);
    return names.at(name);
  };
  std::map<std::string, Representation> representations;
  for (const auto &item : doc.at("representations")) {
    fields(item,
           {"id", "tensor", "pool", "dtype", "layout", "strides_bytes", "padding_elements_by_axis",
            "alignment_bytes", "actual_bytes", "packing", "backend_context", "quantization"});
    Representation r;
    r.id = text(item, "id");
    r.tensor = tensor(item);
    r.pool = text(item, "pool");
    r.dtype = choice<DataType>(text(item, "dtype"), {{"float32", DataType::Float32},
                                                     {"float16", DataType::Float16},
                                                     {"float64", DataType::Float64},
                                                     {"int64", DataType::Int64},
                                                     {"int32", DataType::Int32},
                                                     {"int8", DataType::Int8},
                                                     {"bool", DataType::Bool}});
    if (item.contains("layout") && !item.at("layout").is_null()) {
      r.layout = choice<Layout>(text(item, "layout"), {{"row_major", Layout::RowMajor},
                                                       {"column_major", Layout::ColumnMajor},
                                                       {"nchw", Layout::NCHW},
                                                       {"nhwc", Layout::NHWC},
                                                       {"oihw", Layout::OIHW},
                                                       {"hwio", Layout::HWIO},
                                                       {"strided", Layout::Strided}});
    }
    r.stridesBytes = optional<std::vector<Bytes>>(item, "strides_bytes");
    r.paddingElementsByAxis = optional<std::vector<Bytes>>(item, "padding_elements_by_axis");
    r.alignmentBytes = optional<Bytes>(item, "alignment_bytes");
    r.actualBytes = optional<Bytes>(item, "actual_bytes");
    if (item.contains("packing")) {
      r.packing = choice<PackingKind>(text(item, "packing"), {{"dense", PackingKind::Dense}});
    }
    if (item.contains("backend_context")) {
      const auto &b = item.at("backend_context");
      fields(b, {"backend", "version", "abi", "context"});
      r.backendContext = BackendContext{text(b, "backend"), text(b, "version"), text(b, "abi"),
                                        text(b, "context")};
    }
    if (item.contains("quantization")) {
      const auto &q = item.at("quantization");
      fields(q, {"kind", "scales", "zero_points", "axis"});
      Quantization quantization;
      quantization.kind =
          choice<QuantizationKind>(text(q, "kind"), {{"none", QuantizationKind::None},
                                                     {"per_tensor", QuantizationKind::PerTensor},
                                                     {"per_axis", QuantizationKind::PerAxis}});
      quantization.scales = q.value("scales", std::vector<double>{});
      quantization.zeroPoints = q.value("zero_points", Shape{});
      quantization.axis = optional<std::size_t>(q, "axis");
      r.quantparams = std::move(quantization);
    }
    validation.append(r.validate(model.graph->tensor(r.tensor), platform));
    require(representations.emplace(r.id, std::move(r)).second, "context_representation_duplicate",
            "duplicate representation");
  }
  auto repr = [&](const json &j, TensorId id) {
    const auto name = text(j, "representation");
    require(representations.count(name) && representations.at(name).tensor == id,
            "context_representation_invalid",
            "representation belongs to another tensor or is absent");
    return name;
  };
  std::set<TensorId> supplied, targets;
  for (const auto &item : doc.at("inputs")) {
    fields(item, {"id", "tensor", "representation", "data_ref", "ready_us"});
    const auto id = tensor(item);
    require(model.graph->tensor(id).kind() == ValueKind::Input ||
                model.graph->tensor(id).kind() == ValueKind::Parameter,
            "context_input_invalid", "residency must reference an input or parameter");
    supplied.insert(id);
    boundary.inputs.push_back({text(item, "id"), id, repr(item, id), text(item, "data_ref"),
                               optional<double>(item, "ready_us")});
  }
  for (const auto &item : doc.at("outputs")) {
    fields(item, {"id", "tensor", "representation"});
    const auto id = tensor(item);
    require(model.graph->modelOutputs() &&
                std::find(model.graph->modelOutputs()->begin(), model.graph->modelOutputs()->end(),
                          id) != model.graph->modelOutputs()->end(),
            "context_output_invalid", "target must reference a declared model output");
    targets.insert(id);
    boundary.outputs.push_back({text(item, "id"), id, repr(item, id)});
  }
  if (model.graph->modelInputs()) {
    for (auto id : *model.graph->modelInputs()) {
      require(supplied.count(id), "context_input_missing", "model input residency missing");
    }
  }
  if (model.graph->modelOutputs()) {
    for (auto id : *model.graph->modelOutputs()) {
      require(targets.count(id), "context_output_missing", "model output target missing");
    }
  }
  if (doc.contains("available_bytes_by_pool")) {
    for (auto it = doc.at("available_bytes_by_pool").begin();
         it != doc.at("available_bytes_by_pool").end(); ++it) {
      json wrapper = {{"value", it.value()}};
      boundary.availableBytesByPool[it.key()] = optional<Bytes>(wrapper, "value");
    }
  }
  if (doc.contains("available_resources")) {
    for (auto it = doc.at("available_resources").begin(); it != doc.at("available_resources").end();
         ++it) {
      json wrapper = {{"value", it.value()}};
      boundary.availableResources[it.key()] = optional<double>(wrapper, "value");
    }
  }
  for (const auto &pool : platform.pools()) {
    if (!boundary.availableBytesByPool.count(pool.id)) {
      boundary.availableBytesByPool[pool.id] = std::nullopt;
    }
  }
  for (const auto &resource : platform.resources()) {
    if (!boundary.availableResources.count(resource.id)) {
      boundary.availableResources[resource.id] = std::nullopt;
    }
  }
  TimingPolicy policy;
  if (doc.contains("timing")) {
    const auto &t = doc.at("timing");
    fields(t, {"start", "finish", "includes_input_transfers", "includes_output_transfers",
               "includes_initialization"});
    policy.start = choice<TimingStart>(
        text(t, "start"), {{"inputs_available", TimingStart::InputsAvailable},
                           {"first_region_start", TimingStart::FirstRegionStart},
                           {"before_initialization", TimingStart::BeforeInitialization}});
    policy.finish = choice<TimingFinish>(
        text(t, "finish"), {{"region_outputs_ready", TimingFinish::RegionOutputsReady},
                            {"outputs_delivered", TimingFinish::OutputsDelivered}});
    policy.includesInputTransfers = t.value("includes_input_transfers", false);
    policy.includesOutputTransfers = t.value("includes_output_transfers", false);
    policy.includesInitialization = t.value("includes_initialization", false);
  }
  auto boundaryValidation = boundary.validate(platform);
  // 部署口径已声明时，缺少的是未来计划的具体事件，不能要求用户编造事件 ID。
  if (doc.contains("timing")) {
    for (auto &issue : boundaryValidation.issues) {
      if (issue.code == "timing_scope_unknown") {
        issue.code = "timing_events_pending_plan";
        issue.message = "Timing policy is declared; concrete events will be bound by a future Plan";
      }
    }
  }
  validation.append(boundaryValidation);
  validation.append(policy.validate());
  check(validation);
  json descriptions = json::array();
  for (const auto &entry : representations) {
    const auto &r = entry.second;
    descriptions.push_back(
        {{"id", r.id},
         {"tensor", r.tensor},
         {"tensor_name", model.graph->tensor(r.tensor).name()},
         {"pool", r.pool},
         {"dtype", toString(r.dtype)},
         {"actual_bytes", r.actualBytes ? json(*r.actualBytes) : json(nullptr)}});
  }
  return {{"schema", "opfusion-context-inspection/1"},
          {"id", boundary.id},
          {"platform_id", platform.id()},
          {"device_count", platform.devices().size()},
          {"representation_count", representations.size()},
          {"input_residency_count", boundary.inputs.size()},
          {"output_target_count", boundary.outputs.size()},
          {"declarations_validated", true},
          {"validation_status", validation.valid() ? "valid" : "unresolved"},
          {"representations", descriptions},
          {"execution_validated", false},
          {"timing_policy", doc.value("timing", json(nullptr))},
          {"timing_event_binding", "pending_plan"},
          {"issues", issues(validation)}};
}
} // namespace opfusion::app
