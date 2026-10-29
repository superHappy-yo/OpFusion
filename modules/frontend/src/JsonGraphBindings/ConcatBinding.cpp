#include "ConcatBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Concat.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace opfusion::frontend::jsonGraph {

SourceBinding makeConcatBinding() {
  auto factory = [](const SourceNode& source) {
    auto requirements = common(source, "ConcatenateAttributes");
    require(!source.operation, "invalid_source", "Concat has no operation selector");
    const auto& inputs = field(source.fields, "inputs");
    known(inputs, {"x_tensor_uids"});
    std::vector<TensorId> inputIds;
    for (const auto& item : field(inputs, "x_tensor_uids").array()) {
      inputIds.push_back(uid(item));
    }
    require(inputIds.size() >= 2 && inputIds.size() <= 4, "unsupported",
            "source Concat supports two to four inputs");
    const auto& params = field(source.fields, "parameters");
    known(params, {"axis", "in_place_index"});
    inactive(params, "in_place_index");
    operators::ConcatAttrs attrs{field(params, "axis").integer()};
    // axis 是逻辑维度下标，不按 tensor 名或物理 stride 改写；来源子集比公共 Concat 更窄。
    require(attrs.axis == 1, "unsupported", "source Concat supports axis one");
    return std::make_unique<operators::Concat>(nodeId(source), std::move(inputIds),
                                               singleOutput(source), attrs,
                                               std::move(requirements));
  };
  return sourceBinding(operators::Concat::semanticIdentity(), "ConcatenateAttributes", std::nullopt,
                       std::move(factory));
}

}  // namespace opfusion::frontend::jsonGraph
