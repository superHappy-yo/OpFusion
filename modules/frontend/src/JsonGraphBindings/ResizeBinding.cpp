#include "ResizeBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Resize.h"

#include <memory>
#include <optional>
#include <utility>

namespace opfusion::frontend::jsonGraph {

SourceBinding makeResizeBinding() {
  auto factory = [](const SourceNode& source) {
    auto requirements = common(source, "ResizeAttributes");
    require(!source.operation, "invalid_source", "Resize has no operation selector");
    const auto& params = field(source.fields, "parameters");
    known(params, {"antialias", "axes", "coordinate_transformation_mode", "cubic_coeff_a",
                    "exclude_outside", "extrapolation_value", "keep_aspect_ratio_policy",
                    "nearest_mode", "resize_mode", "roi_end", "roi_start", "scales", "sizes"});
    // 坐标/取整政策影响数学结果，必须逐项匹配，不能只凭 Resize 名称采用默认插值。
    require(field(params, "resize_mode").string() == "nearest" &&
                field(params, "coordinate_transformation_mode").string() == "asymmetric" &&
                field(params, "nearest_mode").string() == "floor" &&
                field(params, "keep_aspect_ratio_policy").string() == "stretch",
            "unsupported", "unsupported Resize coordinate/interpolation policy");
    require(field(params, "roi_start").array().empty() && field(params, "roi_end").array().empty(),
            "unsupported", "Resize ROI must be empty");
    operators::ResizeAttrs attrs;
    attrs.axes = integers(field(params, "axes"));
    attrs.sizes = integers(field(params, "sizes"));
    attrs.scales = numbers(field(params, "scales"));
    attrs.antialias = field(params, "antialias").boolean();
    attrs.excludeOutside = field(params, "exclude_outside").boolean();
    attrs.cubicCoefficient = field(params, "cubic_coeff_a").number();
    attrs.extrapolationValue = field(params, "extrapolation_value").number();
    return std::make_unique<operators::Resize>(nodeId(source), singleInput(source),
                                               singleOutput(source), std::move(attrs),
                                               std::move(requirements));
  };
  return sourceBinding(operators::Resize::semanticIdentity(), "ResizeAttributes", std::nullopt,
                       std::move(factory));
}

}  // namespace opfusion::frontend::jsonGraph
