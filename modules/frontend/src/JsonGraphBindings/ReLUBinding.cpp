#include "ReLUBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/ReLU.h"

namespace opfusion::frontend::jsonGraph {

SourceBinding makeReLUBinding() {
  // 只绑定标准单输入 ReLU；共享读取器要求额外输入和裁剪/斜率字段缺失或为 null。
  return sourceBinding(operators::ReLU::semanticIdentity(), "PointwiseAttributes", "relu_fwd",
                       [](const SourceNode& source) {
                         return pointwise<operators::ReLU>(source, "relu_fwd", false,
                                                           operators::ReLUAttrs{});
                       });
}

}  // namespace opfusion::frontend::jsonGraph
