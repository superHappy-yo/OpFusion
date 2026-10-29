#include "SliceBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Slice.h"

#include <memory>
#include <optional>
#include <utility>

namespace opfusion::frontend::jsonGraph {

SourceBinding makeSliceBinding() {
  auto factory = [](const SourceNode& source) {
    auto requirements = common(source, "SliceAttributes");
    require(!source.operation, "invalid_source", "Slice has no operation selector");
    const auto& params = field(source.fields, "parameters");
    known(params, {"start_indices", "limit_indices", "slice_strides"});
    // 按来源次序保留每个逻辑维度的 [start,limit) 与元素步长；不裁剪或规范化越界值。
    operators::SliceAttrs attrs{integers(field(params, "start_indices")),
                                integers(field(params, "limit_indices")),
                                integers(field(params, "slice_strides"))};
    return std::make_unique<operators::Slice>(nodeId(source), singleInput(source),
                                              singleOutput(source), std::move(attrs),
                                              std::move(requirements));
  };
  return sourceBinding(operators::Slice::semanticIdentity(), "SliceAttributes", std::nullopt,
                       std::move(factory));
}

}  // namespace opfusion::frontend::jsonGraph
