#pragma once

#include "opfusion/registry/Registry.h"

namespace opfusion::frontend {

/// 向调用方拥有的可变 Registry 安装构建发现的来源绑定；仅在调用期间借用。
/// 目标语义须先登记；歧义、目标缺失或冻结后登记由 Registry 抛出 Error。
/// 不发现外部插件；逐项 bind，失败不回滚此前已成功安装的其他绑定。
void registerBuiltinSourceBindings(Registry& registry);
/// 返回覆盖来源适配、领域和构建条件的内容摘要；返回指针借用静态存储，不得释放。
/// 内容由构建阶段生成，不是导入模型的内容摘要或设备验证证据。
const char* builtinSourceBindingsDigest() noexcept;

}  // namespace opfusion::frontend
