// 汇总纯校验诊断，保持非法优先和未知传播；不执行修复或设备工作。
#include "opfusion/common/Validation.h"

#include <utility>

namespace opfusion {

ValidationStatus Validation::status() const noexcept {
  bool unknown = false;
  for (const auto& issue : issues) {
    // 非法的 IssueKind 也不能降级为成功。
    if (issue.kind != IssueKind::Unresolved) {
      return ValidationStatus::Invalid;
    }
    unknown = true;
  }
  return unknown ? ValidationStatus::Unresolved : ValidationStatus::Valid;
}

void Validation::invalid(std::string code, std::string object, std::string message) {
  issues.push_back({IssueKind::Invalid, std::move(code), std::move(object), std::move(message)});
}

void Validation::unresolved(std::string code, std::string object, std::string message) {
  issues.push_back({IssueKind::Unresolved, std::move(code), std::move(object), std::move(message)});
}

void Validation::append(const Validation& other, bool includeUnresolved) {
  for (const auto& issue : other.issues) {
    if (includeUnresolved || issue.kind != IssueKind::Unresolved) {
      issues.push_back(issue);
    }
  }
}

}  // namespace opfusion
