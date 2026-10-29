#include "MaxPoolBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/MaxPool.h"

#include <memory>
#include <utility>
#include <vector>

namespace opfusion::frontend::jsonGraph {

SourceBinding makeMaxPoolBinding() {
  auto factory = [](const SourceNode &source) {
    // 池化来源是顶层扁平字段，不重造 inputs/outputs 分组；窗口/步长/padding 单位为逻辑元素。
    auto requirements = common(source, "ResampleFwdAttributes", true);
    require(source.operation && *source.operation == "maxpool" &&
                field(source.fields, "resample_mode").string() == "maxpool",
            "unsupported", "unsupported resample mode");
    require(field(source.fields, "padding_mode").string() == "neg_inf_pad", "unsupported",
            "padding must be negative infinity");
    operators::MaxPoolAttrs attrs;
    attrs.window = integers(field(source.fields, "window"));
    attrs.strides = integers(field(source.fields, "stride"));
    attrs.pads = pads(source.fields);
    attrs.returnIndices = field(source.fields, "generate_index").boolean();
    return std::make_unique<operators::MaxPool>(
        nodeId(source), std::vector<TensorId>{uid(source.fields, "x_tensor_uid")},
        std::vector<TensorId>{uid(source.fields, "y_tensor_uid")}, attrs, std::move(requirements));
  };
  return sourceBinding(operators::MaxPool::semanticIdentity(), "ResampleFwdAttributes", "maxpool",
                       std::move(factory));
}

} // namespace opfusion::frontend::jsonGraph
