#pragma once

#include "opfusion/frontend/ModelImporter.h"
#include "opfusion/registry/Registry.h"

#include <string>

namespace opfusion::frontend::detail {

/// 模块内部的 benchmark-json/1 读取入口；同步借用文本、冻结目录和诊断来源名。
/// 只经公共 Registry 创建节点，经 DAG 验证后返回自有结果，不保留传入引用。
/// 来源或公共校验失败抛出带 sourceName 的 Error，不提供部分图或设备执行。
ImportResult importJsonGraphText(const std::string& text, const Registry& registry,
                                 const std::string& sourceName);

}  // namespace opfusion::frontend::detail
