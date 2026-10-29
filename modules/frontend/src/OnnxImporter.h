#pragma once
#include "opfusion/frontend/ModelImporter.h"
namespace opfusion::frontend::detail {
ImportResult importOnnx(const std::string &bytes, const Registry &registry,
                        const std::string &path);
}
