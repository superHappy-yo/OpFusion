#pragma once
#include "opfusion/frontend/ModelImporter.h"

#include <nlohmann/json.hpp>
namespace opfusion::app {
// apps 是唯一装配层；解析结果只声明部署，不执行设备工作。
nlohmann::json inspectContext(const std::string &path, const frontend::ImportResult &model);
} // namespace opfusion::app
