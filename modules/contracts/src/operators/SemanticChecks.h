#pragma once

#include "opfusion/common/Error.h"
#include "opfusion/graph/Operator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace opfusion::operators::detail {

// 仅供同模块具体算子复用的纯校验/索引构造，不是公共扩展接口。
// 所有只读参数只在调用期间借用；返回的签名与访问关系均拥有自己的记录。

/// 将具体类绑定到命名空间与名称，不混入来源或 API 版本。
SemanticIdentity identity(const char *name);
/// 检查固定数量输入和单输出，保留输入端口顺序及重复引用。
void ports(const Operator &op, std::size_t count);
/// 先执行公共校验，再检查输入数、同 dtype 及可选浮点限制；未知 dtype 不通过。
void signatures(const Operator &op, const std::vector<TensorSignature> &inputs, std::size_t count,
                bool floatsOnly = false);
/// 非负静态维度的受检加法，溢出抛出 shape_overflow。
std::int64_t add(std::int64_t a, std::int64_t b);
/// 非负静态维度的受检乘法，溢出抛出 shape_overflow。
std::int64_t multiply(std::int64_t a, std::int64_t b);

/// 窗口、步长和 dilation 必须为正，不能用零表示尚未知。
template <std::size_t N> void positive(const std::array<std::int64_t, N> &values) {
  for (auto value : values) {
    require(value > 0, "invalid_attribute", "window/stride/dilation must be positive");
  }
}

/// 四边逻辑 padding 必须非负，单位为元素而非字节。
void padding(const std::array<std::int64_t, 4> &values);
void positive(const Shape &values);
void padding(const Shape &values);
/// 在正 kernel/dilation/stride、非负 padding 下计算 floor 空间输出尺寸。
/// 调用方先校验属性；无有效窗口或计算溢出时明确失败。
std::int64_t spatial(std::int64_t input, std::int64_t kernel, std::int64_t dilation,
                     std::int64_t before, std::int64_t after, std::int64_t stride);
/// 检查签名 rank，不根据名称、stride 或输出声明修正 shape。
void rank(const TensorSignature &input, std::size_t expected);
/// 构造新的 NCHW 逻辑轴记录。
LogicalAxes nchw();
/// 构造新的 OIHW 权重逻辑轴记录。
LogicalAxes oihw();
/// 构造二维 Row/Column 轴记录，归约含义由具体访问关系给出。
LogicalAxes matrixAxes();
/// 推导双输入尾轴对齐广播，只合并兼容轴知识，0 与 1 广播为 0。
std::vector<TensorSignature> binary(const Operator &op, const std::vector<TensorSignature> &inputs);
/// 校验并复制单输入签名，不回读声明输出；未知轴仍保持未知。
std::vector<TensorSignature> unary(const Operator &op, const std::vector<TensorSignature> &inputs,
                                   bool floatsOnly = false);
/// 构造单迭代器索引项；合法域、除数和边界由完整 IndexSemantics 校验。
IndexExpression index(std::size_t axis, std::int64_t coefficient = 1, std::int64_t divisor = 1,
                      std::int64_t offset = 0);
/// 由已验证静态 shape 构造非归约半开域，不物化元素数据。
std::vector<IterationAxis> iteration(const Shape &shape);
/// 为给定 rank 构造恒等坐标，rank=0 保留标量的空坐标。
std::vector<IndexExpression> identityCoordinates(std::size_t rank);
/// 基于已推导输出建立逐元素/广播访问，单例广播读取常量零坐标。
IndexSemantics pointwiseIndex(const std::vector<TensorSignature> &inputs,
                              const TensorSignature &output);
/// 取得完整 piece 的值并标记 Known；不替代完整关系校验。
IndexSemantics known(IndexPiece piece);

} // namespace opfusion::operators::detail
