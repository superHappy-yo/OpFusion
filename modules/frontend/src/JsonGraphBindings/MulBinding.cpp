#include "MulBinding.h"

#include "SourceFields.h"
#include "opfusion/operators/Mul.h"

namespace opfusion::frontend::jsonGraph {

SourceBinding makeMulBinding() {
  // 复用严格双输入 schema，不去重操作数，避免将 Mul(U,U) 的两次使用合并。
  return sourceBinding(operators::Mul::semanticIdentity(), "PointwiseAttributes", "mul",
                       [](const SourceNode& source) {
                         return pointwise<operators::Mul>(
                             source, "mul", true, operators::MulAttrs{});
                       });
}

}  // namespace opfusion::frontend::jsonGraph
