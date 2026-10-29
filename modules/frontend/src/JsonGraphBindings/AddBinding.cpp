#include "AddBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Add.h"

namespace opfusion::frontend::jsonGraph {

SourceBinding makeAddBinding() {
  // 仅翻译 add 来源 schema；广播及 shape/dtype 规则仍由公共 Add 校验。
  return sourceBinding(operators::Add::semanticIdentity(), "PointwiseAttributes", "add",
                       [](const SourceNode& source) {
                         return pointwise<operators::Add>(
                             source, "add", true, operators::AddAttrs{});
                       });
}

}  // namespace opfusion::frontend::jsonGraph
