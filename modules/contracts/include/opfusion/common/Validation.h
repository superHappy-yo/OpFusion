#pragma once

// 纯领域校验结果；非法输入与必要条件未知分开，不表示设备执行证据。
#include <string>
#include <vector>

namespace opfusion {

/// Invalid 是已知违例；Unresolved 是证明所需的信息尚未建立。
enum class IssueKind { Invalid, Unresolved };

/// 单项诊断；默认保守视为非法，object 保存相关领域对象的身份。
struct ValidationIssue {
  IssueKind kind = IssueKind::Invalid;
  std::string code;
  std::string object;
  std::string message;
};

/// Valid 仅指当前校验器的显式条件通过，不证明原始证据的真实性。
enum class ValidationStatus { Valid, Invalid, Unresolved };

/// 拥有诊断的值记录；聚合时 Invalid 优先于 Unresolved，未知不是成功。
struct Validation {
  std::vector<ValidationIssue> issues;
  /// 汇总全部诊断；没有诊断才返回 Valid，无需设备或外部状态。
  ValidationStatus status() const noexcept;
  /// 是否不存在违例或未知条件；不能仅用“没有 Invalid”替代本查询。
  bool valid() const noexcept { return status() == ValidationStatus::Valid; }
  /// 追加已知违例；保存传入字符串，不借用调用方的诊断文本。
  void invalid(std::string code, std::string object, std::string message);
  /// 追加未建立的条件，不把缺失的容量、规则或证据补成零。
  void unresolved(std::string code, std::string object, std::string message);
  /// 复制追加诊断；排除未知项只用于已有明确语义的局部聚合。
  void append(const Validation& other, bool includeUnresolved = true);
};

}  // namespace opfusion
