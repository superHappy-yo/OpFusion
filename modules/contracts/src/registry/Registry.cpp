#include "opfusion/registry/Registry.h"

#include "opfusion/common/Error.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

// 实例级目录的装配/冻结协议；匹配仅看公开 selector，不按具体算子名称分支。
namespace opfusion {
namespace {

void validateIdentity(const SemanticIdentity &semantic) {
  require(!semantic.name.empty() && !semantic.nameSpace.empty(), "semantic_invalid",
          "incomplete semantic identity");
}

// 缺省 operation 是通配符，不能通过登记顺序或更具体名称消解重叠。
bool overlaps(const SourceSelector &a, const SourceSelector &b) {
  return a.format == b.format && a.type == b.type &&
         (!a.operation || !b.operation || a.operation == b.operation);
}

bool matches(const SourceSelector &selector, const SourceNode &source) {
  return selector.format == source.format && selector.type == source.type &&
         (!selector.operation || selector.operation == source.operation);
}

} // namespace

void Registry::add(Registration registration) {
  require(!mFrozen, "registry_frozen", "cannot mutate a frozen registry");
  validateIdentity(registration.semantic);
  for (const auto &entry : mRegistrations) {
    require(entry.semantic.key() != registration.semantic.key(), "duplicate_contribution",
            "duplicate semantic identity: " + registration.semantic.key());
  }
  mRegistrations.push_back(std::move(registration));
}

void Registry::bind(SourceBinding binding) {
  require(!mFrozen, "registry_frozen", "cannot mutate a frozen registry");
  validateIdentity(binding.semantic);
  const auto target =
      std::find_if(mRegistrations.begin(), mRegistrations.end(), [&](const Registration &entry) {
        return entry.semantic.key() == binding.semantic.key();
      });
  require(target != mRegistrations.end(), "binding_target_missing",
          "source binding requires a registered semantic identity: " + binding.semantic.key());
  require(!binding.selectors.empty(), "binding_invalid", "source binding needs a selector");
  require(static_cast<bool>(binding.factory), "factory_missing",
          "source binding requires a factory");

  for (std::size_t i = 0; i < binding.selectors.size(); ++i) {
    const auto &selector = binding.selectors[i];
    require(!selector.format.empty() && !selector.type.empty(), "binding_invalid",
            "empty source selector");
    for (std::size_t j = 0; j < i; ++j) {
      require(!overlaps(selector, binding.selectors[j]), "binding_ambiguous",
              "overlapping selectors in one source binding");
    }
    for (const auto &entry : mSourceBindings) {
      for (const auto &existing : entry.selectors) {
        require(!overlaps(selector, existing), "binding_ambiguous", "overlapping source bindings");
      }
    }
  }
  // 完整校验后才移动整个绑定；任一 selector 失败都不能污染已装配目录。
  mSourceBindings.push_back(std::move(binding));
}

void Registry::freeze(std::string identity) {
  require(!mFrozen, "registry_frozen", "registry already frozen");
  require(!identity.empty(), "id_invalid", "registry identity cannot be empty");
  mId = std::move(identity);
  mFrozen = true;
}

std::unique_ptr<Operator> Registry::create(const SourceNode &source) const {
  require(mFrozen, "registry_not_frozen", "freeze registry before use");
  // 指针仅借用已冻结 vector 中的绑定，限本次同步 create；不跨调用缓存。
  const SourceBinding *match = nullptr;
  for (const auto &entry : mSourceBindings) {
    for (const auto &selector : entry.selectors) {
      if (!matches(selector, source)) {
        continue;
      }
      require(match == nullptr, "binding_ambiguous", "ambiguous source binding");
      match = &entry;
    }
  }
  require(match != nullptr, "operator_unknown",
          "no binding for " + source.type + (source.operation ? "/" + *source.operation : ""));
  auto result = match->factory(source);
  require(result != nullptr && result->semantic().key() == match->semantic.key(), "factory_invalid",
          "factory returned wrong semantic identity");
  return result;
}

SourceExpansion Registry::expand(const SourceNode &source) const {
  require(mFrozen, "registry_not_frozen", "freeze registry before use");
  for (const auto &binding : mSourceBindings) {
    for (const auto &selector : binding.selectors) {
      if (matches(selector, source) && binding.normalizer) {
        auto result = binding.normalizer(source);
        require(!result.nodes.empty() && result.nodes.front() &&
                    result.nodes.front()->semantic().key() == binding.semantic.key(),
                "factory_invalid", "normalizer returned wrong primary semantic");
        for (const auto &node : result.nodes) {
          require(node != nullptr, "factory_invalid", "null normalized node");
          require(std::any_of(
                      mRegistrations.begin(), mRegistrations.end(),
                      [&](const auto &r) { return r.semantic.key() == node->semantic().key(); }),
                  "factory_invalid", "unregistered normalized semantic");
        }
        return result;
      }
    }
  }
  SourceExpansion result;
  result.nodes.push_back(create(source));
  return result;
}

} // namespace opfusion
