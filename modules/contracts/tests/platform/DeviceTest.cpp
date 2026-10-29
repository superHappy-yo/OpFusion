// 负责 Device 多态快照、平台唯一库存和执行资源请求的纯契约回归。
// 所有设备、环境、配额和共享关系均为测试声明；不探测硬件、不执行工作，也不产生设备证据。
#include "opfusion/common/Error.h"
#include "opfusion/common/Validation.h"
#include "opfusion/platform/CpuDevice.h"
#include "opfusion/platform/DeclaredDevice.h"
#include "opfusion/platform/Device.h"
#include "opfusion/platform/Platform.h"
#include "opfusion/platform/Resource.h"

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
int checks = 0;

void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string diagnostics(const Validation& result) {
  std::string details;
  for (const auto& item : result.issues) {
    details += "\n  " + item.code + " [" + item.object + "] " + item.message;
  }
  return details;
}

void valid(const Validation& result, const std::string& message) {
  check(result.valid(), message + diagnostics(result));
}

void invalid(const Validation& result, const std::string& code) {
  check(result.status() == ValidationStatus::Invalid &&
        std::any_of(result.issues.begin(), result.issues.end(), [&](const ValidationIssue& item) {
          return item.kind == IssueKind::Invalid && item.code == code;
        }), "Expected Invalid " + code + diagnostics(result));
}

// 规则/容量/共享关系未知时，既不接受别的未知项，也不把它们折算成零或默认支持。
void unresolvedExactly(const Validation& result,
                       std::initializer_list<std::string> expected,
                       const std::string& message) {
  std::set<std::string> actual;
  for (const auto& item : result.issues) {
    check(item.kind == IssueKind::Unresolved, message + diagnostics(result));
    actual.insert(item.code);
  }
  check(actual == std::set<std::string>(expected), message + diagnostics(result));
  check(result.status() == ValidationStatus::Unresolved, message + diagnostics(result));
}

void error(const std::function<void()>& operation, const std::string& code) {
  try {
    operation();
  } catch (const Error& caught) {
    check(caught.code() == code, "Expected error " + code + ", got " + caught.code());
    return;
  }
  check(false, "Expected error " + code);
}

DeviceDescription description(std::string id = "gpu", DeviceKind kind = DeviceKind::Gpu) {
  DeviceDescription result;
  result.id = std::move(id);
  result.nodeId = "synthetic-node";
  result.kind = kind;
  result.architecture = "synthetic-architecture";
  result.instance = result.id + "-instance";
  result.configuration = "test-fixed-configuration";
  result.backends = {"synthetic-backend"};
  result.environments = {{"synthetic-backend", "1", "test-abi", "test-runtime", false,
                          std::nullopt}};
  return result;
}

DeviceDescription cpuDescription(std::string id = "cpu") {
  auto result = description(std::move(id), DeviceKind::Cpu);
  result.cpuAllocation = CpuAllocation{1.5, std::vector<std::uint32_t>{0, 1},
                                      NumaBinding::NotApplicable, std::nullopt};
  return result;
}

DeviceResourceRules rulesFor(std::vector<ResourceUse> uses = {{"compute", 1.0}}) {
  DeviceResourceRules result;
  result.compute = uses;
  result.localConversion = std::move(uses);
  return result;
}

Resource computeResource(std::string id = "compute", DeviceId owner = "gpu",
                         std::optional<double> capacity = 4.0, std::string unit = "slots") {
  Resource result;
  result.id = std::move(id);
  result.kind = ResourceKind::Compute;
  result.scope = ResourceScope::Device;
  result.unit = std::move(unit);
  result.capacity = capacity;
  result.device = std::move(owner);
  return result;
}

std::unique_ptr<const Device> makeDevice(DeviceDescription record, DeviceResourceRules rules) {
  if (record.kind == DeviceKind::Cpu) {
    return std::make_unique<CpuDevice>(std::move(record), std::move(rules));
  }
  return std::make_unique<DeclaredDevice>(std::move(record), std::move(rules));
}

Platform platformWith(DeviceDescription record, DeviceResourceRules rules,
                      std::vector<Resource> resources,
                      std::vector<MemoryPool> pools = {}) {
  std::vector<std::unique_ptr<const Device>> devices;
  devices.push_back(makeDevice(std::move(record), std::move(rules)));
  return Platform("test-platform", "test-registry", std::move(devices), std::move(pools),
                   std::move(resources), {});
}

Validation execution(const Platform& platform, const std::vector<ResourceUse>& uses,
                     DeviceWork work = DeviceWork::Compute, const DeviceId& device = "gpu") {
  return validateExecutionResourceUses(platform, device, work, uses, "test-work");
}

bool sameUses(const std::optional<std::vector<ResourceUse>>& first,
              const std::optional<std::vector<ResourceUse>>& second) {
  if (first.has_value() != second.has_value()) {
    return false;
  }
  if (!first) {
    return true;
  }
  if (first->size() != second->size()) {
    return false;
  }
  for (std::size_t i = 0; i < first->size(); ++i) {
    if ((*first)[i].resource != (*second)[i].resource || (*first)[i].amount != (*second)[i].amount) {
      return false;
    }
  }
  return true;
}

bool sameDescription(const DeviceDescription& first, const DeviceDescription& second) {
  if (first.id != second.id || first.nodeId != second.nodeId || first.kind != second.kind ||
      first.architecture != second.architecture || first.instance != second.instance ||
      first.configuration != second.configuration || first.backends != second.backends ||
      first.cpuAllocation.has_value() != second.cpuAllocation.has_value() ||
      first.environments.size() != second.environments.size()) {
    return false;
  }
  if (first.cpuAllocation) {
    const auto& a = *first.cpuAllocation;
    const auto& b = *second.cpuAllocation;
    if (a.quotaCores != b.quotaCores || a.affinity != b.affinity ||
        a.numaBinding != b.numaBinding || a.numaNode != b.numaNode) {
      return false;
    }
  }
  for (std::size_t i = 0; i < first.environments.size(); ++i) {
    const auto& a = first.environments[i];
    const auto& b = second.environments[i];
    if (a.backend != b.backend || a.version != b.version || a.abi != b.abi ||
        a.runtimeVersion != b.runtimeVersion || a.requiresDriver != b.requiresDriver ||
        a.driverVersion != b.driverVersion) {
      return false;
    }
  }
  return true;
}

// 测试直接派生 Device 的可替换查询，规则和标签均按值拥有；没有外部可变闭包或硬件句柄。
class FixtureDevice final : public Device {
 public:
  FixtureDevice(DeviceDescription record, DeviceResourceRules rules, std::string ruleLabel)
      : Device(std::move(record)), mRules(std::move(rules)), mRuleLabel(std::move(ruleLabel)) {}

  ResourceRequirements requiredResources(const Platform&, DeviceWork work) const override {
    return mRules.requirements(work, description().id);
  }

  std::unique_ptr<const Device> clone() const override {
    return std::make_unique<FixtureDevice>(*this);
  }

  const std::string& ruleLabel() const noexcept { return mRuleLabel; }

 private:
  DeviceResourceRules mRules;
  std::string mRuleLabel;
};

enum class CloneFailure { Null, ChangedIdentity };

// 故意违反 clone 的两个最小实现，只用于验证 Platform 不发布空或变更身份的复制品。
class BrokenCloneDevice final : public Device {
 public:
  BrokenCloneDevice(DeviceDescription record, CloneFailure failure)
      : Device(std::move(record)), mFailure(failure) {}

  ResourceRequirements requiredResources(const Platform&, DeviceWork work) const override {
    return rulesFor().requirements(work, description().id);
  }

  std::unique_ptr<const Device> clone() const override {
    if (mFailure == CloneFailure::Null) {
      return nullptr;
    }
    auto changed = description();
    changed.id = "changed-identity";
    return std::make_unique<DeclaredDevice>(std::move(changed), rulesFor());
  }

 private:
  CloneFailure mFailure;
};

void polymorphicSnapshotTests() {
  auto record = description("custom", DeviceKind::Other);
  auto rules = rulesFor({{"custom-compute", 2.0}});
  rules.localConversion = std::vector<ResourceUse>{{"custom-compute", 0.5}};
  const auto expectedDescription = record;
  const auto expectedRules = rules;
  auto owned = std::make_unique<FixtureDevice>(record, rules, "rules-v1");
  record.id = "caller-modified";
  record.nodeId = "caller-node";
  record.architecture = "caller-architecture";
  record.backends.clear();
  record.environments[0].runtimeVersion = "caller-runtime";
  (*rules.compute)[0].amount = 99.0;
  (*rules.localConversion)[0].resource = "caller-resource";
  std::vector<std::unique_ptr<const Device>> devices;
  devices.push_back(std::move(owned));
  check(!owned, "Platform fixture transfers unique device ownership");
  Platform platform("test-platform", "test-registry", std::move(devices), {},
                     {computeResource("custom-compute", "custom")}, {});
  valid(platform.validate(), "Custom derived device has a complete synthetic description");
  const Device& base = platform.device("custom");
  const auto* derived = dynamic_cast<const FixtureDevice*>(&base);
  check(derived && derived->ruleLabel() == "rules-v1", "Lookup returns the original dynamic type");
  check(sameDescription(base.description(), expectedDescription),
        "Caller edits cannot change the owned description/environment snapshot");
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    const auto requirements = base.requiredResources(platform, work);
    valid(requirements.validation, "Virtual resource query returns the derived rule");
    check(sameUses(requirements.minimumUses,
                   work == DeviceWork::Compute ? expectedRules.compute : expectedRules.localConversion),
          "Caller edits cannot change the owned work-specific rules");
  }
  valid(execution(platform, {{"custom-compute", 2.0}}, DeviceWork::Compute, "custom"),
        "A direct Device subclass supplies the enforced compute minimum");
  invalid(execution(platform, {{"custom-compute", 1.0}}, DeviceWork::Compute, "custom"),
          "execution_resource_missing");
  valid(execution(platform, {{"custom-compute", 0.5}}, DeviceWork::LocalConversion, "custom"),
        "Local conversion dispatches its separate fractional rule");
  const auto clone = base.clone();
  check(clone && clone.get() != &base && dynamic_cast<const FixtureDevice*>(clone.get()),
        "clone preserves dynamic type with independent object ownership");
  check(sameDescription(clone->description(), expectedDescription), "clone retains every description field");
  const Platform copied(platform);
  check(&copied.device("custom") != &base &&
        dynamic_cast<const FixtureDevice*>(&copied.device("custom")),
        "Platform copy deep-clones the Device rather than slicing or copying unique_ptrs");
  check(copied.id() == platform.id() && copied.registryId() == platform.registryId() &&
        sameDescription(copied.device("custom").description(), expectedDescription),
        "Platform copy preserves snapshot identities and description values");
  check(copied.resources().size() == 1 && copied.resources().data() != platform.resources().data() &&
        copied.resource("custom-compute").capacity == 4.0,
        "Copy owns one independent resource inventory, not one copy per device rule");
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    const auto expected = work == DeviceWork::Compute ? expectedRules.compute : expectedRules.localConversion;
    const auto clonedRules = clone->requiredResources(copied, work);
    const auto copiedRules = copied.device("custom").requiredResources(copied, work);
    valid(clonedRules.validation, "Cloned query remains valid");
    valid(copiedRules.validation, "Copied platform query remains valid");
    check(sameUses(clonedRules.minimumUses, expected) && sameUses(copiedRules.minimumUses, expected),
          "clone and Platform copy retain both complete rule snapshots");
  }
  // 返回记录也是值；调用方修改一次查询结果不能改变下一次查询或克隆品。
  auto editable = copied.device("custom").requiredResources(copied, DeviceWork::Compute);
  (*editable.minimumUses)[0].amount = 100.0;
  check(sameUses(copied.device("custom").requiredResources(copied, DeviceWork::Compute).minimumUses,
                 expectedRules.compute), "Editing query output cannot mutate the immutable rules");

  for (const auto failure : {CloneFailure::Null, CloneFailure::ChangedIdentity}) {
    std::vector<std::unique_ptr<const Device>> broken;
    broken.push_back(std::make_unique<BrokenCloneDevice>(description(), failure));
    const Platform source("test-platform", "test-registry", std::move(broken), {},
                           {computeResource()}, {});
    error([&] { const Platform rejected(source); },
          failure == CloneFailure::Null ? "device_clone_null" : "device_clone_identity");
  }
}

void identityAndLookupTests() {
  error([] { DeclaredDevice rejected(description(""), rulesFor()); }, "identity_invalid");
  error([] {
    std::vector<std::unique_ptr<const Device>> devices;
    devices.push_back(nullptr);
    Platform rejected("test-platform", "test-registry", std::move(devices), {}, {}, {});
  }, "device_null");
  error([] {
    std::vector<std::unique_ptr<const Device>> devices;
    devices.push_back(makeDevice(description(), rulesFor()));
    devices.push_back(makeDevice(description(), rulesFor()));
    Platform rejected("test-platform", "test-registry", std::move(devices), {}, {}, {});
  }, "identity_invalid");
  error([] {
    std::vector<std::unique_ptr<const Device>> devices;
    devices.push_back(makeDevice(description(), rulesFor()));
    Platform rejected("", "test-registry", std::move(devices), {}, {}, {});
  }, "platform_identity");
  error([] {
    auto first = computeResource();
    (void)platformWith(description(), rulesFor(), {first, first});
  }, "identity_invalid");
  auto platform = platformWith(description(), rulesFor(), {computeResource()},
                               {{"pool", "synthetic-physical-pool", 1024, {"gpu"}}});
  valid(platform.validate(), "Baseline inventory is structurally valid without executing a device");
  check(&platform.device("gpu") == platform.devices()[0].get(), "Device lookup borrows the owned instance");
  check(&platform.resource("compute") == &platform.resources()[0], "Resource lookup borrows the unique inventory entry");
  check(platform.accessible("pool", "gpu"), "Access requires an explicit listed device");
  error([&] { (void)platform.device("missing"); }, "device_missing");
  invalid(platform.validateDevice("missing"), "device_missing");
  error([&] { (void)platform.pool("missing"); }, "pool_missing");
  error([&] { (void)platform.resource("missing"); }, "resource_missing");
  error([&] { (void)platform.link("missing"); }, "link_missing");
  error([&] { (void)platform.accessible("missing", "gpu"); }, "pool_missing");
  error([&] { (void)platform.accessible("pool", "missing"); }, "device_missing");
  invalid(execution(platform, {{"compute", 1}}, DeviceWork::Compute, "missing"), "device_missing");
  const Platform copied(platform);
  check(dynamic_cast<const DeclaredDevice*>(&copied.device("gpu")) != nullptr &&
        sameDescription(copied.device("gpu").description(), platform.device("gpu").description()),
        "DeclaredDevice survives Platform copying without slicing");
  check(copied.pools().size() == 1 && copied.pool("pool").physicalIdentity == "synthetic-physical-pool" &&
        copied.pool("pool").physicalCapacityBytes == 1024 && copied.pool("pool").accessibleDevices == std::vector<DeviceId>{"gpu"},
        "Platform copy retains pool inventory and access values");
  const Platform empty("empty", "test-registry", {}, {}, {}, {});
  unresolvedExactly(empty.validate(), {"devices_unknown"}, "An empty platform is unknown, not a default CPU");
  std::vector<std::unique_ptr<const Device>> devices;
  devices.push_back(makeDevice(description(), rulesFor()));
  const Platform missingRegistry("test-platform", "", std::move(devices), {}, {computeResource()}, {});
  unresolvedExactly(missingRegistry.validate(), {"registry_unknown"}, "Registry identity cannot be inferred");
}

void unknownAndEmptyRuleTests() {
  const auto unknown = platformWith(description(), {}, {computeResource()});
  valid(unknown.validate(), "Unused work categories may retain unknown rules in a platform snapshot");
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    const auto requirements = unknown.device("gpu").requiredResources(unknown, work);
    check(!requirements.minimumUses, "nullopt means an unknown rule, not an empty rule");
    unresolvedExactly(requirements.validation, {"device_resource_rules_unknown"},
                       "Virtual query preserves unknown work requirements");
    unresolvedExactly(execution(unknown, {{"compute", 1}}, work), {"device_resource_rules_unknown"},
                       "An actual Compute request cannot establish an unknown device rule");
    unresolvedExactly(execution(unknown, {}, work),
                       {"device_resource_rules_unknown", "compute_resources_unknown"},
                       "Unknown rules and absent reservations remain two explicit gaps");
  }
  auto partialRules = rulesFor();
  partialRules.localConversion.reset();
  const auto partial = platformWith(description(), partialRules, {computeResource()});
  valid(execution(partial, {{"compute", 1}}), "Known Compute rule is independent of unknown LocalConversion");
  unresolvedExactly(execution(partial, {{"compute", 1}}, DeviceWork::LocalConversion),
                     {"device_resource_rules_unknown"}, "Compute rules are not substituted for conversion rules");
  const auto empty = platformWith(description(), rulesFor({}), {computeResource()});
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    const auto requirements = empty.device("gpu").requiredResources(empty, work);
    check(requirements.minimumUses && requirements.minimumUses->empty(), "Known empty and nullopt are distinguishable");
    valid(requirements.validation, "Rule query itself preserves a known empty declaration");
    invalid(execution(empty, {{"compute", 1}}, work), "device_compute_requirement");
    invalid(execution(empty, {}, work), "execution_compute_resource");
  }
  const auto invalidWork = static_cast<DeviceWork>(999);
  invalid(rulesFor().requirements(invalidWork, "gpu").validation, "device_work");
  invalid(partial.device("gpu").requiredResources(partial, invalidWork).validation, "device_work");
  invalid(execution(partial, {{"compute", 1}}, invalidWork), "device_work");
  const auto missingRule = platformWith(description(), rulesFor({{"missing", 1}}), {computeResource()});
  invalid(execution(missingRule, {{"compute", 1}}), "resource_missing");
}

std::vector<Resource> mixedResources() {
  auto registers = computeResource("registers");
  registers.kind = ResourceKind::Registers;
  registers.scope = ResourceScope::Thread;
  registers.unit = "registers";
  registers.capacity = 64.0;
  auto copy = computeResource("copy");
  copy.kind = ResourceKind::Copy;
  auto threadCompute = computeResource("thread-compute");
  threadCompute.scope = ResourceScope::Thread;
  auto blockCompute = computeResource("block-compute");
  blockCompute.scope = ResourceScope::Block;
  return {computeResource(), registers, copy, threadCompute, blockCompute};
}

void executionResourceTests() {
  const auto resources = mixedResources();
  const auto platform = platformWith(description(), rulesFor(), resources);
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    valid(execution(platform, {{"compute", 1}}, work), "Device-scoped Compute satisfies a known execution minimum");
    valid(execution(platform, {{"compute", 1}, {"registers", 8}}, work),
          "Thread-local registers may supplement, but not replace, Compute");
    invalid(execution(platform, {}, work), "execution_resource_missing");
    invalid(execution(platform, {}, work), "execution_compute_resource");
    for (const std::string pseudoCompute : {"registers", "copy", "thread-compute", "block-compute"}) {
      // 非空资源表仍可能完全没有整任务 Compute；kind 与 scope 两个维度都要核验。
      const std::vector<ResourceUse> requested{{pseudoCompute, 1}};
      invalid(execution(platform, requested, work), "execution_resource_missing");
      invalid(execution(platform, requested, work), "execution_compute_resource");
      const auto badRule = platformWith(description(), rulesFor(requested), resources);
      invalid(execution(badRule, requested, work), "device_compute_requirement");
      invalid(execution(badRule, requested, work), "execution_compute_resource");
    }
    const auto minimumTwo = platformWith(description(), rulesFor({{"compute", 2}}), resources);
    invalid(execution(minimumTwo, {{"compute", 1}}, work), "execution_resource_missing");
    valid(execution(minimumTwo, {{"compute", 2}}, work), "Actual amount must cover the exact minimum");
    invalid(execution(platform, {{"compute", 1}, {"compute", 1}}, work), "resource_duplicate");
    invalid(execution(platform, {{"compute", 5}}, work), "resource_capacity");
    invalid(execution(platform, {{"missing", 1}}, work), "resource_missing");
    for (const double amount : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()}) {
      invalid(execution(platform, {{"compute", amount}}, work), "resource_amount");
      const auto badMinimum = platformWith(description(), rulesFor({{"compute", amount}}), resources);
      invalid(execution(badMinimum, {{"compute", 1}}, work), "resource_amount");
    }
  }
  const auto duplicateMinimum = platformWith(description(), rulesFor({{"compute", 1}, {"compute", 1}}), resources);
  invalid(execution(duplicateMinimum, {{"compute", 1}}), "resource_duplicate");
  const auto oversizedMinimum = platformWith(description(), rulesFor({{"compute", 5}}), resources);
  invalid(execution(oversizedMinimum, {{"compute", 1}}), "resource_capacity");
  const auto unknownCapacity = platformWith(description(), rulesFor(), {computeResource("compute", "gpu", std::nullopt)});
  unresolvedExactly(unknownCapacity.validate(), {"resource_capacity_unknown"}, "Unknown physical capacity remains unknown");
  unresolvedExactly(execution(unknownCapacity, {{"compute", 1}}), {"resource_capacity_unknown"},
                     "Requests and minima cannot turn unknown capacity into known capacity");
  const auto zeroCapacity = platformWith(description(), rulesFor(), {computeResource("compute", "gpu", 0.0)});
  valid(zeroCapacity.validate(), "Known zero physical capacity is a valid unavailable resource description");
  check(zeroCapacity.resource("compute").capacity == 0.0, "Known zero capacity is not nullopt");
  invalid(execution(zeroCapacity, {{"compute", 1}}), "resource_capacity");
}

void ownershipAndDescriptionTests() {
  std::vector<std::unique_ptr<const Device>> devices;
  devices.push_back(makeDevice(description("gpu"), rulesFor()));
  devices.push_back(makeDevice(description("other"), rulesFor({{"other-compute", 1}})));
  const Platform platform("test-platform", "test-registry", std::move(devices),
                          {{"private-pool", "synthetic-private-pool", 1024, {"gpu"}}},
                          {computeResource(), computeResource("other-compute", "other")}, {});
  invalid(execution(platform, {{"compute", 1}, {"other-compute", 1}}), "resource_device");
  invalid(validateResourceUses(platform, {{"compute", 1}}, "test-use", "other"), "resource_device");
  auto pooled = computeResource("pooled");
  pooled.device.reset();
  pooled.pool = "private-pool";
  invalid(validateResourceUses(platformWith(description("other"), rulesFor({{"pooled", 1}}),
                                            {pooled}, {{"private-pool", "private", 1024, {}}}),
                                {{"pooled", 1}}, "test-use", "other"), "resource_pool_access");
  auto unowned = computeResource();
  unowned.device.reset();
  const auto unknownOwner = platformWith(description(), rulesFor(), {unowned});
  unresolvedExactly(execution(unknownOwner, {{"compute", 1}}), {"resource_owner_unknown"},
                     "A task-scoped Compute name does not establish ownership");
  auto shared = computeResource("shared");
  shared.device.reset();
  shared.sharedDevices = {"other"};
  shared.sharingRef = "synthetic:explicit-other-owner";
  // 使用公开描述校验器核对错误归属，不修改已发布的平台库存。
  auto conflict = shared;
  conflict.device = "gpu";
  invalid(validateResourceDescription(platform, conflict), "resource_ownership");
  auto duplicateOwner = shared;
  duplicateOwner.sharedDevices = {"gpu", "gpu"};
  invalid(validateResourceDescription(platform, duplicateOwner), "resource_ownership");
  auto missingSharing = shared;
  missingSharing.sharingRef.clear();
  unresolvedExactly(validateResourceDescription(platform, missingSharing), {"resource_sharing_unknown"},
                     "Shared owners need an explicit sharing basis");
  auto missingDevice = computeResource();
  missingDevice.device = "absent";
  invalid(validateResourceDescription(platform, missingDevice), "device_missing");
  auto missingPool = pooled;
  missingPool.pool = "absent";
  invalid(validateResourceDescription(platform, missingPool), "pool_missing");
  for (const double capacity : {-1.0, std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()}) {
    auto bad = computeResource();
    bad.capacity = capacity;
    invalid(validateResourceDescription(platform, bad), "resource_capacity");
    error([&] { (void)platformWith(description(), rulesFor(), {bad}); }, "resource_capacity");
  }
  auto noUnit = computeResource();
  noUnit.unit.clear();
  unresolvedExactly(validateResourceDescription(platform, noUnit), {"resource_unit_unknown"},
                     "Units cannot be guessed from resource identity");
  auto badKind = computeResource();
  badKind.kind = static_cast<ResourceKind>(999);
  invalid(validateResourceDescription(platform, badKind), "resource_kind");
  auto badScope = computeResource();
  badScope.scope = static_cast<ResourceScope>(999);
  invalid(validateResourceDescription(platform, badScope), "resource_scope");
  auto unknownKind = description();
  unknownKind.kind = DeviceKind::Unknown;
  const auto unknown = platformWith(unknownKind, rulesFor(), {computeResource()});
  unresolvedExactly(unknown.validate(), {"device_kind_unknown"}, "Unknown classification is not an implicit CPU");
  auto invalidKind = description();
  invalidKind.kind = static_cast<DeviceKind>(999);
  error([&] { (void)platformWith(invalidKind, rulesFor(), {computeResource()}); }, "device_kind");
}

void cpuAllocationTests() {
  const auto record = cpuDescription();
  valid(validateDeviceDescription(record), "Fractional CPU quota with explicit affinity/NUMA is legal");
  const auto rules = rulesFor({{"compute", 0.5}});
  const auto platform = platformWith(record, rules, {computeResource("compute", "cpu", 8.0, "cores")});
  valid(platform.validate(), "CPU description does not create another physical-core inventory");
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    valid(execution(platform, {{"compute", 1.5}}, work, "cpu"), "Fractional core request can equal a fractional quota");
    invalid(execution(platform, {{"compute", 1.75}}, work, "cpu"), "cpu_resource_quota");
    const auto excessiveMinimum = platformWith(record, rulesFor({{"compute", 1.75}}),
                                               {computeResource("compute", "cpu", 8.0, "cores")});
    invalid(excessiveMinimum.device("cpu").requiredResources(excessiveMinimum, work).validation,
            "cpu_resource_quota");
    invalid(execution(excessiveMinimum, {{"compute", 1.75}}, work, "cpu"), "cpu_resource_quota");
  }
  const auto cloned = platform.device("cpu").clone();
  const Platform copied(platform);
  check(cloned && dynamic_cast<const CpuDevice*>(cloned.get()) &&
        dynamic_cast<const CpuDevice*>(&copied.device("cpu")),
        "CPU clone and Platform copy retain the CPU-specific behavior");
  check(sameDescription(cloned->description(), record) &&
        sameDescription(copied.device("cpu").description(), record),
        "CPU clone/copy preserve quota, affinity, NUMA and environment snapshot");
  invalid(execution(copied, {{"compute", 1.75}}, DeviceWork::Compute, "cpu"), "cpu_resource_quota");
  auto externalCpu = record;
  auto externalRules = rules;
  CpuDevice isolated(externalCpu, externalRules);
  externalCpu.cpuAllocation->quotaCores = 0.0;
  externalCpu.cpuAllocation->affinity->clear();
  (*externalRules.compute)[0].amount = 9.0;
  check(sameDescription(isolated.description(), record), "CPU allocation is stored by value, not caller alias");
  check(sameUses(isolated.requiredResources(platform, DeviceWork::Compute).minimumUses, rules.compute),
        "CPU resource rules are stored by value");

  auto fractional = cpuDescription();
  fractional.cpuAllocation->quotaCores = 0.5;
  fractional.cpuAllocation->affinity = std::vector<std::uint32_t>{0};
  const auto slots = platformWith(fractional, rulesFor({{"compute", 3.0}}),
                                  {computeResource("compute", "cpu", 8.0, "slots")});
  valid(slots.device("cpu").requiredResources(slots, DeviceWork::Compute).validation,
        "Three slots are not three cores and cannot be compared with a 0.5-core quota");
  valid(execution(slots, {{"compute", 4.0}}, DeviceWork::Compute, "cpu"),
        "A slot request uses its own declared physical capacity without an implicit core conversion");
  const auto cores = platformWith(fractional, rulesFor({{"compute", 3.0}}),
                                  {computeResource("compute", "cpu", 8.0, "cores")});
  invalid(execution(cores, {{"compute", 4.0}}, DeviceWork::Compute, "cpu"), "cpu_resource_quota");

  for (const auto affinity : {std::vector<std::uint32_t>{0}, std::vector<std::uint32_t>{}}) {
    auto unavailable = cpuDescription();
    unavailable.cpuAllocation->quotaCores = 0.0;
    unavailable.cpuAllocation->affinity = affinity;
    valid(validateDeviceDescription(unavailable), "Known zero quota/empty affinity describe an unavailable CPU");
    const auto zero = platformWith(unavailable, rulesFor({{"compute", 1.0}}),
                                   {computeResource("compute", "cpu", 8.0, "slots")});
    invalid(execution(zero, {{"compute", 1.0}}, DeviceWork::Compute, "cpu"), "cpu_unavailable");
  }
  auto missingAllocation = record;
  missingAllocation.cpuAllocation.reset();
  unresolvedExactly(validateDeviceDescription(missingAllocation), {"cpu_allocation_unknown"},
                     "A CPU label does not establish allocation conditions");
  auto missingQuota = record;
  missingQuota.cpuAllocation->quotaCores.reset();
  unresolvedExactly(validateDeviceDescription(missingQuota), {"cpu_quota_unknown"}, "Missing CPU quota remains unknown");
  auto missingAffinity = record;
  missingAffinity.cpuAllocation->affinity.reset();
  unresolvedExactly(validateDeviceDescription(missingAffinity), {"cpu_affinity_unknown"},
                     "Missing affinity does not mean all host CPUs");
  for (const double quota : {-0.5, 2.5, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    auto bad = record;
    bad.cpuAllocation->quotaCores = quota;
    invalid(validateDeviceDescription(bad), "cpu_quota");
  }
  auto duplicateAffinity = record;
  duplicateAffinity.cpuAllocation->affinity = std::vector<std::uint32_t>{0, 0};
  invalid(validateDeviceDescription(duplicateAffinity), "cpu_affinity");
  auto unknownNuma = record;
  unknownNuma.cpuAllocation->numaBinding = NumaBinding::Unknown;
  unresolvedExactly(validateDeviceDescription(unknownNuma), {"numa_unknown"}, "NUMA applicability is not inferred");
  auto node = record;
  node.cpuAllocation->numaBinding = NumaBinding::Node;
  unresolvedExactly(validateDeviceDescription(node), {"numa_unknown"}, "NUMA-node binding needs an explicit node");
  node.cpuAllocation->numaNode = 2;
  valid(validateDeviceDescription(node), "Explicit NUMA-node allocation is legal");
  auto nonNumaWithNode = record;
  nonNumaWithNode.cpuAllocation->numaNode = 2;
  invalid(validateDeviceDescription(nonNumaWithNode), "numa_binding");
  auto invalidNuma = record;
  invalidNuma.cpuAllocation->numaBinding = static_cast<NumaBinding>(999);
  invalid(validateDeviceDescription(invalidNuma), "numa_binding");
  auto wrongDevice = description();
  wrongDevice.cpuAllocation = record.cpuAllocation;
  invalid(validateDeviceDescription(wrongDevice), "cpu_allocation_device");
  error([] { CpuDevice rejected(description(), rulesFor()); }, "cpu_device_kind");
  error([] { DeclaredDevice rejected(cpuDescription(), rulesFor()); }, "declared_cpu_device");
}

// 描述为 CPU 的直接 Device 子类也须遵守公共分配约束，不能靠替换派生类绕过。
void commonCpuInvariantTests() {
  const auto customCpu = [](DeviceDescription record) {
    std::vector<std::unique_ptr<const Device>> devices;
    devices.push_back(std::make_unique<FixtureDevice>(
        std::move(record), rulesFor({{"compute", 0.5}}), "custom-cpu-rules"));
    return Platform("test-platform", "test-registry", std::move(devices), {},
                    {computeResource("compute", "cpu", 8.0, "cores")}, {});
  };
  const auto platform = customCpu(cpuDescription());
  check(dynamic_cast<const FixtureDevice*>(&platform.device("cpu")) != nullptr,
        "CPU invariants are exercised through a direct Device subclass");
  valid(execution(platform, {{"compute", 1.5}}, DeviceWork::Compute, "cpu"),
        "A custom CPU obeys the same fractional quota");
  invalid(execution(platform, {{"compute", 1.75}}, DeviceWork::Compute, "cpu"),
          "cpu_resource_quota");
  auto missingAffinity = cpuDescription();
  missingAffinity.cpuAllocation->affinity.reset();
  const auto unknown = customCpu(missingAffinity);
  unresolvedExactly(execution(unknown, {{"compute", 1}}, DeviceWork::Compute, "cpu"),
                     {"cpu_affinity_unknown"}, "A custom CPU cannot infer missing affinity");
  auto unavailable = cpuDescription();
  unavailable.cpuAllocation->quotaCores = 0.0;
  unavailable.cpuAllocation->affinity = std::vector<std::uint32_t>{};
  const auto zero = customCpu(unavailable);
  invalid(execution(zero, {{"compute", 0.5}}, DeviceWork::Compute, "cpu"), "cpu_unavailable");
  auto duplicate = cpuDescription();
  duplicate.cpuAllocation->affinity = std::vector<std::uint32_t>{0, 0};
  error([&] { (void)customCpu(duplicate); }, "cpu_affinity");
}

Platform cpuPair(DeviceDescription first, DeviceDescription second, bool declareShared,
                 bool includeSharingRef = true) {
  std::vector<std::unique_ptr<const Device>> devices;
  devices.push_back(std::make_unique<CpuDevice>(first, rulesFor({{"private-a", 0.5}})));
  devices.push_back(std::make_unique<CpuDevice>(second, rulesFor({{"private-b", 0.5}})));
  std::vector<Resource> resources{computeResource("private-a", first.id, 1.5, "cores"),
                                  computeResource("private-b", second.id, 1.5, "cores")};
  if (declareShared) {
    Resource shared = computeResource("shared-cores", first.id, 2.0, "cores");
    shared.device.reset();
    shared.sharedDevices = {first.id, second.id};
    if (includeSharingRef) {
      shared.sharingRef = "synthetic:one-shared-core-inventory";
    }
    resources.push_back(std::move(shared));
  }
  return Platform("test-platform", "test-registry", std::move(devices), {}, std::move(resources), {});
}

void cpuSharingTests() {
  auto first = cpuDescription("cpu-a");
  auto second = cpuDescription("cpu-b");
  second.cpuAllocation->affinity = std::vector<std::uint32_t>{1, 2};
  const auto unknown = cpuPair(first, second, false);
  unresolvedExactly(unknown.validate(), {"cpu_sharing_unknown"},
                     "Same-node overlapping affinity needs a common resource constraint");
  unresolvedExactly(unknown.validateDevice("cpu-a"), {"cpu_sharing_unknown"},
                     "Instance quotas alone do not prove physical CPU isolation");
  unresolvedExactly(execution(unknown, {{"private-a", 0.5}}, DeviceWork::Compute, "cpu-a"),
                     {"cpu_sharing_unknown"}, "Execution cannot invent missing affinity-sharing constraints");
  const auto shared = cpuPair(first, second, true);
  valid(shared.validate(), "Explicit shared Compute constraint establishes the declared sharing relation");
  check(shared.resources().size() == 3 &&
        std::count_if(shared.resources().begin(), shared.resources().end(), [](const Resource& resource) {
          return resource.id == "shared-cores";
        }) == 1,
        "Both CPU instances share one physical capacity entry, never one duplicated entry per device");
  check(shared.resource("shared-cores").sharedDevices == std::vector<DeviceId>({"cpu-a", "cpu-b"}) &&
        shared.resource("shared-cores").capacity == 2.0 && !shared.resource("shared-cores").device,
        "Shared owner list and capacity live only in the platform inventory");
  for (const auto work : {DeviceWork::Compute, DeviceWork::LocalConversion}) {
    // 知道存在共享资源还不够；实际请求必须进入同一容量账本，否则私有配额可重复占核。
    invalid(execution(shared, {{"private-a", 0.5}}, work, "cpu-a"), "cpu_shared_resource_missing");
    invalid(execution(shared, {{"private-b", 0.5}}, work, "cpu-b"), "cpu_shared_resource_missing");
    valid(execution(shared, {{"private-a", 0.5}, {"shared-cores", 0.5}}, work, "cpu-a"),
          "First CPU requests both its local minimum and the common physical constraint");
    valid(execution(shared, {{"private-b", 0.5}, {"shared-cores", 0.5}}, work, "cpu-b"),
          "Second CPU references the exact same shared resource ID");
  }
  const Platform copied(shared);
  valid(copied.validate(), "Deep cloning CPU instances preserves shared-resource semantics");
  check(copied.resources().size() == shared.resources().size() &&
        &copied.resource("shared-cores") != &shared.resource("shared-cores"),
        "Copied platform owns one independent copy of the shared inventory");
  invalid(execution(copied, {{"private-a", 0.5}}, DeviceWork::Compute, "cpu-a"),
          "cpu_shared_resource_missing");
  const auto missingRef = cpuPair(first, second, true, false);
  unresolvedExactly(missingRef.validate(), {"cpu_sharing_unknown", "resource_sharing_unknown"},
                     "A bare shared-owner list is not proof of a common constraint");
  second.nodeId = "other-synthetic-node";
  const auto otherNode = cpuPair(first, second, false);
  valid(otherNode.validate(), "Equal numeric CPU IDs on different nodes are not an affinity collision");
  valid(execution(otherNode, {{"private-a", 0.5}}, DeviceWork::Compute, "cpu-a"),
        "Cross-node affinity numbers do not require a fabricated shared resource");
  second.nodeId = first.nodeId;
  second.cpuAllocation->affinity = std::vector<std::uint32_t>{2, 3};
  const auto disjoint = cpuPair(first, second, false);
  valid(disjoint.validate(), "Known disjoint affinity on one node needs no overlapping-core constraint");
  valid(execution(disjoint, {{"private-b", 0.5}}, DeviceWork::Compute, "cpu-b"),
        "Disjoint instance obeys its own resource and quota declarations");
}
}  // namespace

int main() {
  try {
    static_assert(std::is_abstract<Device>::value, "Device is a behavior port, not a value record");
    static_assert(!std::is_copy_assignable<Device>::value, "Published device descriptions are not assignable");
    static_assert(!std::is_copy_assignable<Platform>::value, "Published platforms are not assignable");
    static_assert(std::is_same<decltype(std::declval<const Device&>().description()),
                               const DeviceDescription&>::value,
                  "Device description is a read-only borrowed view");
    polymorphicSnapshotTests();
    identityAndLookupTests();
    unknownAndEmptyRuleTests();
    executionResourceTests();
    ownershipAndDescriptionTests();
    cpuAllocationTests();
    commonCpuInvariantTests();
    cpuSharingTests();
    std::cout << "device contract tests: " << checks << " checks passed (declarations only; no device execution)\n";
    return 0;
  } catch (const std::exception& caught) {
    std::cerr << "device contract tests failed: " << caught.what() << '\n';
    return 1;
  }
}
