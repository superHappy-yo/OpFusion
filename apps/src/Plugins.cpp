/// @file
/// 校验静态插件 manifest、依赖和工厂贡献，再显式冻结应用拥有的 Registry。
/// 这里只装配已链接工厂，不加载动态库、不探测设备，也不提供 CPU 回退。
#include "LinkedPlugins.h"
#include "opfusion/common/Error.h"
#include "opfusion/frontend/BuiltinSourceBindings.h"
#include "opfusion/registry/BuiltinOperators.h"
#include "opfusion/registry/Registry.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <toml.hpp>
#include <vector>

namespace opfusion::app {
namespace {

using Version = std::array<unsigned long, 3>;
using Providers = std::map<std::string, std::string>;
using Dependencies = std::map<std::string, std::vector<std::string>>;

/// 一次装配期间的包描述与校验状态；尚未验证的记录不发布到冻结 Registry。
struct Package {
  std::string directory;
  std::string manifest;
  std::string pluginVersion;
  /// 仅磁盘发现时有值；编译清单模式改用 linked 的文件清单校验引用。
  std::filesystem::path path;
  toml::table parsed;
  /// 借用 loadRegistry 局部 compiled 表，且只在该次调用内使用。
  const LinkedPlugin *linked = nullptr;
  std::set<std::string> contributions;
};

/// 解析无前导零的三段数字版本；格式或数值范围非法时抛 manifest_invalid。
Version version(const std::string &text) {
  require(std::regex_match(text, std::regex("(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)")),
          "manifest_invalid", "expected three-part version: " + text);
  Version result{};
  std::istringstream input(text);
  std::string part;
  for (auto &field : result) {
    std::getline(input, part, '.');
    try {
      field = std::stoul(part);
    } catch (const std::exception &) {
      throw Error("manifest_invalid", "version component out of range");
    }
  }
  return result;
}

/// 顺序求值逗号连接的比较式，首个不满足项返回 false；已求值项非法则抛错。
/// 工厂协议范围与 contracts 提供者版本分别检查，不能据 contractVersion 改工厂 API。
bool inRange(const std::string &actual, const std::string &range) {
  const auto current = version(actual);
  require(!range.empty() && range.back() != ',', "manifest_invalid",
          "empty or malformed version range");
  std::istringstream input(range);
  std::string term;
  while (std::getline(input, term, ',')) {
    term.erase(
        std::remove_if(term.begin(), term.end(), [](unsigned char c) { return std::isspace(c); }),
        term.end());
    std::smatch match;
    require(std::regex_match(term, match, std::regex("(>=|<=|>|<|=)?([0-9]+\\.[0-9]+\\.[0-9]+)")),
            "manifest_invalid", "unsupported version range: " + range);
    const auto target = version(match[2]);
    const auto comparison = match[1].str();
    if (comparison == ">=" && !(current >= target)) {
      return false;
    }
    if (comparison == "<=" && !(current <= target)) {
      return false;
    }
    if (comparison == ">" && !(current > target)) {
      return false;
    }
    if (comparison == "<" && !(current < target)) {
      return false;
    }
    if ((comparison.empty() || comparison == "=") && current != target) {
      return false;
    }
  }
  return true;
}

/// 读取必需的非空字符串；不把缺失字段或其他 TOML 类型隐式转换成文本。
std::string text(const toml::table &table, const char *field) {
  const auto value = table[field].value<std::string>();
  require(value && !value->empty(), "manifest_invalid",
          std::string("missing/string field: ") + field);
  return *value;
}

/// 借用必需数组，仅在原 table 存活且未修改时有效；缺失或类型不符明确失败。
const toml::array &array(const toml::table &table, const char *field) {
  const auto *value = table[field].as_array();
  require(value != nullptr, "manifest_invalid", std::string("missing/array field: ") + field);
  return *value;
}

/// 按字节读取完整 manifest，保留与编译清单精确比较的原文；读取失败不返回部分文本。
std::string read(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "manifest_invalid", "cannot read plugin manifest");
  std::ostringstream output;
  output << input.rdbuf();
  require(!input.bad(), "manifest_invalid", "failed reading plugin manifest");
  return output.str();
}

/// 校验包内相对文件引用；磁盘模式拒绝符号链接越界，编译模式核对固化文件清单。
/// 不执行所引用的文件，也不允许通过父目录引用隐含依赖其他包。
void reference(const Package &package, const std::string &ref) {
  const auto path = std::filesystem::path(ref);
  require(!ref.empty() && !path.is_absolute(), "manifest_invalid",
          "package reference must be relative");
  for (const auto &component : path) {
    require(component != ".." && component != ".", "manifest_invalid",
            "package reference escapes package");
  }
  if (!package.path.empty()) {
    const auto root = std::filesystem::canonical(package.path);
    const auto resolved = std::filesystem::weakly_canonical(root / path);
    const auto relative = resolved.lexically_relative(root);
    require(!relative.empty() && *relative.begin() != ".." &&
                std::filesystem::is_regular_file(resolved),
            "manifest_invalid", "package reference missing or outside package: " + ref);
  } else if (package.linked) {
    std::istringstream files(package.linked->packageFiles);
    std::string file;
    bool found = false;
    while (std::getline(files, file)) {
      if (file == path.generic_string()) {
        found = true;
      }
    }
    require(found, "manifest_invalid", "package reference not in compiled package: " + ref);
  }
}

/// 从指定 operators 目录或已编译清单收集包；排序仅固定诊断顺序，不解决冲突。
/// 此阶段只读取描述，不调用工厂；显式空目录仍是合法的可选插件集合。
std::vector<Package> discoverPackages(const char *pluginRoot,
                                      const std::vector<LinkedPlugin> &compiled) {
  std::vector<Package> packages;
  if (pluginRoot) {
    const auto category = std::filesystem::path(pluginRoot) / "operators";
    require(std::filesystem::is_directory(category), "manifest_invalid",
            "plugin root has no operators directory");
    const auto root = std::filesystem::canonical(category);
    for (const auto &entry : std::filesystem::directory_iterator(category)) {
      if (!entry.is_directory()) {
        continue;
      }
      const auto resolved = std::filesystem::canonical(entry.path());
      const auto relative = resolved.lexically_relative(root);
      require(!relative.empty() && *relative.begin() != "..", "manifest_invalid",
              "plugin directory escapes discovery root");
      const auto manifest = entry.path() / "plugin.toml";
      if (std::filesystem::exists(manifest)) {
        packages.push_back(
            {entry.path().filename().string(), read(manifest), {}, entry.path(), {}, nullptr, {}});
      }
    }
  } else {
    for (const auto &linked : compiled) {
      packages.push_back({linked.id, linked.manifest, {}, {}, {}, &linked, {}});
    }
  }
  std::sort(packages.begin(), packages.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.directory < rhs.directory; });
  return packages;
}

/// 依次校验 manifest/API、身份冲突、文件引用与贡献工厂，并收集提供者版本。
/// 未链接的算子贡献在此阶段失败，不能假定之后的依赖校验已发生。
void describePackage(Package &package, const std::vector<LinkedPlugin> &compiled,
                     Providers &providers, std::set<std::string> &contributionIds) {
  try {
    package.parsed = toml::parse(package.manifest);
  } catch (const toml::parse_error &) {
    throw Error("manifest_invalid", "invalid TOML in " + package.directory);
  }
  const auto &table = package.parsed;
  const auto id = text(table, "plugin_id");
  require(id == package.directory && std::regex_match(id, std::regex("[a-z][a-z0-9_.-]*")),
          "manifest_invalid", "plugin ID must match its directory");
  require(text(table, "manifest_schema_version") == "2.0.0", "manifest_invalid",
          "unknown manifest schema");
  require(text(table, "kind") == "operators", "manifest_invalid",
          "static profile supports operators only");
  package.pluginVersion = text(table, "plugin_version");
  (void)version(package.pluginVersion);
  require(inRange(operatorApiVersion, text(table, "plugin_api_range")), "api_incompatible",
          "requires the current Operator protocol; legacy APIs are not accepted");
  require(providers.emplace(id, package.pluginVersion).second, "duplicate_contribution",
          "duplicate plugin ID");
  if (!package.linked) {
    for (const auto &linked : compiled) {
      if (id == linked.id) {
        package.linked = &linked;
      }
    }
  }
  for (const auto &ref : array(table, "artifact_refs")) {
    const auto value = ref.value<std::string>();
    require(value.has_value(), "manifest_invalid", "artifact reference must be a string");
    reference(package, *value);
  }
  for (const auto &node : array(table, "contributions")) {
    const auto *contribution = node.as_table();
    require(contribution != nullptr, "manifest_invalid", "contribution must be a table");
    require(text(*contribution, "type") == "operator", "manifest_invalid",
            "unsupported contribution type");
    const auto contributionId = text(*contribution, "id");
    require(!contribution->contains("semantic_version"), "manifest_invalid",
            "semantic_version was removed in manifest schema 2");
    require(contributionIds.insert(contributionId).second, "duplicate_contribution",
            "duplicate contribution ID");
    require(providers.emplace(contributionId, package.pluginVersion).second,
            "duplicate_contribution", "provider ID conflict");
    package.contributions.insert(contributionId);
    reference(package, text(*contribution, "description_ref"));
    require(package.linked != nullptr, "provider_unavailable",
            "operator package is not linked; rebuild required");
    require(text(*contribution, "factory_ref") == package.linked->factorySymbol,
            "provider_unavailable", "factory is not linked for this package");
  }
}

/// 将已满足的 requires 记录为 owner 的依赖；缺失/不兼容或外部依赖明确失败。
/// 此静态 profile 不安装第三方依赖，也不把未验证的环境条件视作可用。
void checkDependencies(const toml::table &table, const std::string &owner,
                       const Providers &providers, Dependencies &dependencies) {
  require(!table.contains("requires") || table["requires"].is_array(), "manifest_invalid",
          "requires must be an array");
  require(!table.contains("external_dependencies") || table["external_dependencies"].is_array(),
          "manifest_invalid", "external_dependencies must be an array");
  if (const auto *required = table["requires"].as_array()) {
    for (const auto &node : *required) {
      const auto *dependency = node.as_table();
      require(dependency != nullptr, "manifest_invalid", "dependency must be a table");
      const auto id = text(*dependency, "id");
      const auto found = providers.find(id);
      const bool compatible =
          found != providers.end() && inRange(found->second, text(*dependency, "version_range"));
      require(compatible, "dependency_missing", "missing or incompatible provider: " + id);
      dependencies[owner].push_back(id);
    }
  }
  if (const auto *external = table["external_dependencies"].as_array()) {
    require(external->empty(), "dependency_missing",
            "static profile cannot validate requested external provider dependencies");
  }
}

/// 在所有包完成描述后校验包级/贡献级依赖及其环；不运行任何工厂。
/// 贡献隐式依赖所属包，因此不能靠包与自身贡献互相引用掩盖依赖环。
void validateDependencies(const std::vector<Package> &packages, const Providers &providers) {
  Dependencies dependencies;
  for (const auto &package : packages) {
    (void)array(package.parsed, "requires");
    (void)array(package.parsed, "external_dependencies");
    checkDependencies(package.parsed, package.directory, providers, dependencies);
    for (const auto &node : array(package.parsed, "contributions")) {
      const auto &contribution = *node.as_table();
      const auto id = text(contribution, "id");
      dependencies[id].push_back(package.directory);
      checkDependencies(contribution, id, providers, dependencies);
    }
  }
  std::map<std::string, unsigned> marks;
  std::function<void(const std::string &)> visit = [&](const std::string &id) {
    require(marks[id] != 1, "dependency_cycle", "plugin dependency cycle");
    if (marks[id] == 2) {
      return;
    }
    marks[id] = 1;
    for (const auto &dependency : dependencies[id]) {
      visit(dependency);
    }
    marks[id] = 2;
  };
  for (const auto &entry : providers) {
    visit(entry.first);
  }
}

/// 核对编译 manifest 原文后显式调用工厂，再比较实际新增语义与声明贡献集合。
/// 失败时局部 Registry 不会发布；不尝试加载未链接工厂或接受部分贡献。
void installPackage(const Package &package, const Providers &providers, Registry &registry) {
  require(package.linked != nullptr && package.linked->factory != nullptr, "provider_unavailable",
          "package is not linked");
  require(package.manifest == package.linked->manifest, "provider_unavailable",
          "manifest differs from compiled package; rebuild required");
  const auto before = registry.registrations().size();
  package.linked->factory(registry);
  std::set<std::string> registered;
  for (std::size_t i = before; i < registry.registrations().size(); ++i) {
    const auto &semantic = registry.registrations()[i].semantic;
    registered.insert(semantic.key());
    const auto declared = providers.find(semantic.key());
    require(declared != providers.end() && declared->second == package.pluginVersion,
            "factory_invalid", "factory provider differs from manifest");
  }
  require(registered == package.contributions, "factory_invalid",
          "factory contributions differ from manifest");
}

} // namespace

Registry loadRegistry(const char *pluginRoot) {
  const auto compiled = linkedPlugins();
  auto packages = discoverPackages(pluginRoot, compiled);
  Registry registry;
  // 内置语义先于来源绑定；可选插件集合为空也不能移除这两类内置贡献。
  registerBuiltinOperators(registry);
  frontend::registerBuiltinSourceBindings(registry);
  // contracts 跟随领域契约版本，两个内置提供者及工厂兼容判断仍使用扩展 API 版本。
  Providers providers{{"contracts", contractVersion},
                      {"builtin-operators", operatorApiVersion},
                      {"builtin-source-bindings", operatorApiVersion}};
  std::set<std::string> contributionIds;
  for (const auto &entry : registry.registrations()) {
    contributionIds.insert(entry.semantic.key());
    require(providers.emplace(entry.semantic.key(), operatorApiVersion).second,
            "duplicate_contribution", "builtin provider identity conflict");
  }
  for (auto &package : packages) {
    describePackage(package, compiled, providers, contributionIds);
  }
  validateDependencies(packages, providers);
  // /3 是身份格式而非 contractVersion；摘要由构建按当前内容和配置重新生成。
  std::string identity = std::string("registry/cpp-static/4:builtin-operators:") +
                         builtinOperatorsDigest() +
                         ":builtin-source-bindings:" + frontend::builtinSourceBindingsDigest();
  for (const auto &package : packages) {
    installPackage(package, providers, registry);
    identity += ":" + package.directory + ":" + package.linked->contentDigest;
  }
  registry.freeze(identity);
  return registry;
}

} // namespace opfusion::app
