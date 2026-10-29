#pragma once

// 领域错误边界：稳定错误码与可读上下文分离，不吞掉非法输入。
#include <stdexcept>
#include <string>
#include <utility>

namespace opfusion {

/// 拥有错误码与消息的领域异常；调用方按 code 处理，不解析消息文本。
class Error : public std::runtime_error {
 public:
  /// 保存错误码并将消息交给标准异常；不借用调用方字符串。
  Error(std::string code, std::string message)
      : std::runtime_error(std::move(message)), mCode(std::move(code)) {}
  /// 借用本异常的稳定错误码，有效期与异常对象相同。
  const std::string& code() const noexcept { return mCode; }
 private:
  std::string mCode;
};

/// 条件不成立时抛出 Error；用于可恢复的输入校验，不依赖 assert 开关。
inline void require(bool condition, const std::string& code,
                    const std::string& message) {
  if (!condition) throw Error(code, message);
}

}  // namespace opfusion
