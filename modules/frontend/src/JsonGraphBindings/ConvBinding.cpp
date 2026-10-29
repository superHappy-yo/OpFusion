#include "ConvBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Conv.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace opfusion::frontend::jsonGraph {

SourceBinding makeConvBinding() {
  auto factory = [](const SourceNode &source) {
    auto requirements = common(source, "ConvolutionFwdAttributes");
    require(!source.operation, "invalid_source", "Conv has no operation selector");
    const auto &inputs = field(source.fields, "inputs");
    known(inputs, {"x_tensor_uid", "w_tensor_uid"});
    const auto &params = field(source.fields, "parameters");
    known(params, {"conv_mode", "dilation", "stride", "pre_padding", "post_padding", "groups"});
    // 来源只表达无偏置互相关；不反转 kernel、不补 BiasAdd，H/W 步长及 padding 按逻辑元素解释。
    require(field(params, "conv_mode").string() == "CROSS_CORRELATION", "unsupported",
            "no kernel reversal");
    operators::ConvAttrs attrs;
    attrs.strides = integers(field(params, "stride"));
    attrs.dilations = integers(field(params, "dilation"));
    attrs.pads = pads(params);
    require(attrs.dilations == Shape{1, 1}, "unsupported", "source dilation must be one");
    for (auto stride : attrs.strides) {
      require(stride == 1 || stride == 2, "unsupported", "source stride must be one or two");
    }
    for (auto pad : attrs.pads) {
      require(pad == 0 || pad == 1, "unsupported", "source padding must be zero or one");
    }
    if (const auto *groups = params.find("groups")) {
      attrs.groups = groups->integer();
    }
    return std::make_unique<operators::Conv>(
        nodeId(source),
        std::vector<TensorId>{uid(inputs, "x_tensor_uid"), uid(inputs, "w_tensor_uid")},
        singleOutput(source), attrs, std::move(requirements));
  };
  return sourceBinding(operators::Conv::semanticIdentity(), "ConvolutionFwdAttributes",
                       std::nullopt, std::move(factory));
}

} // namespace opfusion::frontend::jsonGraph
