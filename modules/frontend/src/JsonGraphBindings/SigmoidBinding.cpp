#include "SigmoidBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Sigmoid.h"

namespace opfusion::frontend::jsonGraph {

SourceBinding makeSigmoidBinding() {
  // 来源标签选择单输入 Sigmoid 语义；此绑定不提供数值 kernel 或近似实现。
  return sourceBinding(operators::Sigmoid::semanticIdentity(), "PointwiseAttributes", "sigmoid_fwd",
                       [](const SourceNode& source) {
                         return pointwise<operators::Sigmoid>(source, "sigmoid_fwd", false,
                                                              operators::SigmoidAttrs{});
                       });
}

}  // namespace opfusion::frontend::jsonGraph
