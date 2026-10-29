#pragma once

#include "opfusion/common/Error.h"
#include "opfusion/graph/Operator.h"
#include "opfusion/graph/Tensor.h"
#include "opfusion/registry/Registry.h"
#include "opfusion/registry/Source.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace opfusion::frontend::jsonGraph {

/// 本头仅共享 JSON 来源字段校验，不提供领域语义或参与 *Binding.h 构建发现。
/// 所有 const 引用/字符串参数仅在调用期间借用；类型/范围/支持性失败抛出 Error。

/// 仅 null、空数组和空对象视为空占位；false、0、空字符串仍是有效非空值。
bool empty(const SourceValue &value);
/// 核对对象字段白名单；未知字段只允许空占位，不默默丢弃带值的新语义。
void known(const SourceValue &value, std::initializer_list<const char *> names);
/// 未支持的可选字段只能缺失或为 null，不能用其他空容器代替 null。
void inactive(const SourceValue &value, const char *name);
/// 必填字段缺失时报错；返回引用借用自 value，不能超出其有效期。
const SourceValue &field(const SourceValue &value, const char *name);
/// 必需端口须为正 int64 UID；null 不是占位张量，不能转成 0。
TensorId uid(const SourceValue &value);
/// 从对象的必填字段读取正 int64 UID，不按表下标解释。
TensorId uid(const SourceValue &value, const char *name);
/// 严格读取 int64 数组；单位由属性决定，不将小数/布尔转成整数。
Shape integers(const SourceValue &value);
/// 读取有限数值数组；允许整数转 double，不接收字符串或布尔。
std::vector<double> numbers(const SourceValue &value);
/// 读取恰有两个整数的属性；次序及元素/步长单位由具体绑定约定。
std::array<std::int64_t, 2> pair(const SourceValue &value, const char *name);
/// 将 pre/post 的 [H,W] 合成 [前H,前W,后H,后W]，单位为逻辑元素。
Shape pads(const SourceValue &value);
/// 返回 source.index+1；溢出时报错，来源 index 本身从 0 开始。
OperatorId nodeId(const SourceNode &source);
/// 核对来源 format/type、通用字段及可选 FP32 compute 声明，不推断全图数值政策。
/// topLevelPool=true 选择池化扁平 schema；false 选择 inputs/outputs/parameters schema。
/// 缺少 compute_data_type 时相应数值要求保持未知，不补 FP32 默认值。
NumericalRequirements common(const SourceNode &source, const char *expectedType,
                             bool topLevelPool = false);
/// 读取分组 inputs 中唯一 x_tensor_uid；保持逻辑值引用，不创建外部输入节点。
std::vector<TensorId> singleInput(const SourceNode &source);
/// 读取分组 outputs 中唯一 y_tensor_uid，不把它升级为模型完整输出。
std::vector<TensorId> singleOutput(const SourceNode &source);
/// 拥有传入语义/工厂的绑定值，复制 type 字符串；此处不登记或校验目标语义。
/// operation=nullopt 是该 format/type 的通配选择器，不表示来源操作未知时自动支持。
SourceBinding sourceBinding(SemanticIdentity semantic, const char *type,
                            std::optional<std::string> operation, OperatorFactory factory);

/// 将已匹配的逐点来源转成公共具体类，返回新算子的独占所有权。
/// binary 决定严格的 1/2 输入 schema；保留重复 UID，额外端口/未支持参数不忽略。
/// source 仅同步借用，属性和数值要求按值传入具体类，不执行数值 kernel。
template <class Concrete, class Attrs>
std::unique_ptr<Operator> pointwise(const SourceNode &source, const char *expectedOperation,
                                    bool binary, Attrs attrs) {
  auto requirements = common(source, "PointwiseAttributes");
  const auto &inputs = field(source.fields, "inputs");
  known(inputs, {"axis_tensor_uid", "elu_alpha", "in_0_tensor_uid", "in_1_tensor_uid",
                 "in_2_tensor_uid", "operation", "relu_lower_clip", "relu_lower_clip_slope",
                 "relu_upper_clip", "softplus_beta", "swish_beta"});
  require(source.operation && *source.operation == expectedOperation &&
              field(inputs, "operation").string() == expectedOperation,
          "invalid_source", "Pointwise operation metadata mismatch");
  for (const auto *name :
       {"axis_tensor_uid", "in_2_tensor_uid", "elu_alpha", "relu_lower_clip",
        "relu_lower_clip_slope", "relu_upper_clip", "softplus_beta", "swish_beta"}) {
    inactive(inputs, name);
  }
  if (const auto *params = source.fields.find("parameters")) {
    require(empty(*params), "unsupported_attribute",
            "Pointwise parameters are not part of this source schema");
  }
  const auto &outputs = field(source.fields, "outputs");
  known(outputs, {"out_0_tensor_uid", "out_1_tensor_uid"});
  inactive(outputs, "out_1_tensor_uid");
  std::vector<TensorId> inputIds{uid(inputs, "in_0_tensor_uid")};
  std::vector<TensorId> outputIds{uid(outputs, "out_0_tensor_uid")};
  // 按语义角色组装端口，不依赖对象键排序，也不对重复 UID 去重。
  if (binary) {
    inputIds.push_back(uid(inputs, "in_1_tensor_uid"));
  } else {
    inactive(inputs, "in_1_tensor_uid");
  }
  return std::make_unique<Concrete>(nodeId(source), std::move(inputIds), std::move(outputIds),
                                    std::move(attrs), std::move(requirements));
}

} // namespace opfusion::frontend::jsonGraph
