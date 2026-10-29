// 仅核验实现和设备的环境声明；版本匹配不等于制品已加载或真实执行通过。
#include "opfusion/model/Model.h"
#include "opfusion/common/Error.h"
#include <algorithm>

namespace opfusion {
Validation Model::validateImplementation(const Implementation& impl) const {
  Validation v;
  if (impl.registryId.empty() || platform().registryId().empty())
    v.unresolved("implementation_registry_unknown", impl.id, "Implementation registry conditions are unknown");
  else if (impl.registryId != platform().registryId())
    v.invalid("implementation_registry", impl.id, "Implementation uses another registry");
  if (impl.backend.empty() || impl.backendVersion.empty() || impl.provider.id.empty() ||
      impl.provider.version.empty() || impl.provider.digest.empty())
    v.unresolved("implementation_conditions_unknown", impl.id, "Exact backend and provider content identity are required");
  if (impl.environment.abi.empty() || impl.environment.runtimeVersion.empty())
    v.unresolved("implementation_environment_unknown", impl.id, "Required ABI and runtime versions must be explicit");
  if (impl.environment.driverVersion && impl.environment.driverVersion->empty())
    v.invalid("implementation_driver", impl.id, "A driver constraint cannot be empty");
  if (!impl.artifactRef || impl.artifactRef->empty())
    v.unresolved("artifact_unknown", impl.id, "An executable library/program artifact reference is required");
  return v;
}
Validation Model::validateExecution(const ImplementationId& id, const DeviceId& deviceId,
                                   const std::string& object) const {
  Validation v;
  try {
    const auto& impl = implementation(id);
    const auto& d = platform().device(deviceId).description();
    v.append(validateImplementation(impl)); v.append(platform().validateDevice(deviceId));
    if (!d.backends.empty() && std::find(d.backends.begin(), d.backends.end(), impl.backend) == d.backends.end())
      v.invalid("execution_backend", object, "Backend is not available on the execution device");
    const auto found = std::find_if(d.environments.begin(), d.environments.end(),
        [&](const BackendEnvironment& e) { return e.backend == impl.backend; });
    if (found == d.environments.end()) {
      v.unresolved("backend_environment_unknown", object, "Selected backend environment is unknown"); return v;
    }
    const auto& e = *found;
    if (e.driverVersion && e.driverVersion->empty())
      v.invalid("driver_version", object, "An explicitly supplied driver version cannot be empty");
    // 缺版本与已知不匹配是两类结果；不能由 backend 名称推断 ABI/runtime 兼容。
    const auto match = [&](const std::string& actual, const std::string& expected, const char* field) {
      if (actual.empty() || expected.empty()) v.unresolved("backend_environment_unknown", object, field);
      else if (actual != expected) v.invalid("backend_environment_mismatch", object, field);
    };
    match(e.version, impl.backendVersion, "backend version");
    match(e.abi, impl.environment.abi, "backend ABI");
    match(e.runtimeVersion, impl.environment.runtimeVersion, "runtime version");
    if (!e.requiresDriver) v.unresolved("driver_requirement_unknown", object, "Driver applicability is unknown");
    if (e.requiresDriver && *e.requiresDriver && (!e.driverVersion || e.driverVersion->empty()))
      v.unresolved("driver_version_unknown", object, "Required driver version is unknown");
    if (impl.environment.driverVersion) {
      if (!e.driverVersion) v.unresolved("driver_version_unknown", object, "Implementation requires a known driver version");
      else match(*e.driverVersion, *impl.environment.driverVersion, "driver version");
    }
    if (d.kind == DeviceKind::Cpu && d.cpuAllocation &&
        ((d.cpuAllocation->quotaCores && *d.cpuAllocation->quotaCores == 0) ||
         (d.cpuAllocation->affinity && d.cpuAllocation->affinity->empty())))
      v.invalid("cpu_unavailable", object, "The selected CPU has no assigned execution capacity");
  } catch (const Error& e) { v.invalid(e.code(), object, e.what()); }
  return v;
}
}  // namespace opfusion
