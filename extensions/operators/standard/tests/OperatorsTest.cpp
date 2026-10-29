#include "opfusion/common/Error.h"
#include "opfusion/operators/Add.h"
#include "opfusion/operators/Concat.h"
#include "opfusion/operators/Conv.h"
#include "opfusion/operators/MatMul.h"
#include "opfusion/operators/MaxPool.h"
#include "opfusion/operators/Mul.h"
#include "opfusion/operators/ReLU.h"
#include "opfusion/operators/Resize.h"
#include "opfusion/operators/Sigmoid.h"
#include "opfusion/operators/Slice.h"
#include "opfusion/registry/BuiltinOperators.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// 历史 standard 测试只保留 contracts 纯语义，不恢复旧生产包或来源解析实现。
// 原来源 fixture/sourceTests 完整迁往 frontend/tests/BenchmarkTest.cpp；两处合计保留原意图。
namespace {
using namespace opfusion;
using namespace opfusion::operators;

// 编译期保证具体领域类、只读属性与逻辑轴类型，不接受物理 Layout 作为替身。
template <class Concrete, class Attrs> constexpr bool concreteOperator() {
  return std::is_base_of_v<Operator, Concrete> && std::is_final_v<Concrete> &&
         std::is_aggregate_v<Attrs> &&
         std::is_same_v<decltype(std::declval<const Concrete &>().attrs()), const Attrs &>;
}
static_assert(concreteOperator<Conv, ConvAttrs>());
static_assert(concreteOperator<Add, AddAttrs>());
static_assert(concreteOperator<ReLU, ReLUAttrs>());
static_assert(concreteOperator<Sigmoid, SigmoidAttrs>());
static_assert(concreteOperator<Mul, MulAttrs>());
static_assert(concreteOperator<Slice, SliceAttrs>());
static_assert(concreteOperator<Concat, ConcatAttrs>());
static_assert(concreteOperator<Resize, ResizeAttrs>());
static_assert(concreteOperator<MaxPool, MaxPoolAttrs>());
static_assert(concreteOperator<MatMul, MatMulAttrs>());
static_assert(std::is_enum_v<Layout> && !std::is_convertible_v<std::string, Layout>);
static_assert(std::is_enum_v<AxisOrder> && !std::is_convertible_v<Layout, AxisOrder>);
static_assert(std::is_same_v<decltype(ConvAttrs{}.dataAxes), AxisOrder>);
static_assert(std::is_same_v<decltype(ConvAttrs{}.weightAxes), AxisOrder>);
static_assert(std::is_same_v<decltype(ResizeAttrs{}.dataAxes), AxisOrder>);
static_assert(std::is_same_v<decltype(MaxPoolAttrs{}.dataAxes), AxisOrder>);
static_assert(!std::is_assignable_v<decltype(ConvAttrs{}.dataAxes) &, Layout>);
static_assert(!std::is_assignable_v<decltype(ConvAttrs{}.weightAxes) &, Layout>);
static_assert(!std::is_assignable_v<decltype(ResizeAttrs{}.dataAxes) &, Layout>);
static_assert(!std::is_assignable_v<decltype(MaxPoolAttrs{}.dataAxes) &, Layout>);
static_assert(std::is_same_v<decltype(ResizeAttrs{}.mode), ResizeMode>);
static_assert(std::is_same_v<decltype(ResizeAttrs{}.coordinateTransform), CoordinateTransform>);
static_assert(std::is_same_v<decltype(ResizeAttrs{}.nearestMode), NearestMode>);
static_assert(std::is_same_v<decltype(ResizeAttrs{}.aspectPolicy), AspectPolicy>);
static_assert(std::is_same_v<decltype(MaxPoolAttrs{}.padding), PoolPadding>);

int checks = 0;

void check(bool condition, const std::string &message) {
  ++checks;
  if (!condition)
    throw std::runtime_error(message);
}

// 非法输入必须抛公共 Error；指定错误码时精确核对，不放宽为任意异常。
template <class F> void rejects(F action, const std::string &code = {}) {
  ++checks;
  try {
    action();
  } catch (const Error &error) {
    if (!code.empty() && error.code() != code) {
      throw std::runtime_error("expected " + code + ", got " + error.code() + ": " + error.what());
    }
    return;
  }
  throw std::runtime_error("expected opfusion::Error");
}

TensorSignature fp(Shape shape) {
  return {std::move(shape), DataType::Float32};
}
// 期望尺寸由用例显式给出；再核对当前索引协议，输入/算子均只在调用内借用。
void shapeIs(const Operator &op, const std::vector<TensorSignature> &inputs, const Shape &expected,
             DataType dtype = DataType::Float32) {
  const auto result = op.infer(inputs);
  check(result.size() == 1 && result[0].shape == expected && result[0].dtype == dtype,
        op.semantic().name + ": wrong inferred shape/dtype");
  const auto semantics = op.indexSemantics(inputs);
  check(semantics.status == SemanticStatus::Known,
        op.semantic().name + ": supported inputs need complete index semantics");
  semantics.validate(inputs, result);
}
void traitIs(const Operator &op, bool elementwise, bool reduction, bool broadcasting) {
  const auto traits = op.traits();
  check(traits.pure == true && traits.elementwise == elementwise && traits.reduction == reduction &&
            traits.broadcasting == broadcasting,
        op.semantic().name + ": wrong traits");
}

// 保留无 bias 卷积、非对称 padding/dilation、分组及所有非法 shape/属性反例。
void convolutionTests() {
  ConvAttrs attrs;
  attrs.pads = {1, 1, 1, 1};
  Conv conv(1, {1, 2}, {3}, attrs);
  shapeIs(conv, {fp({1, 16, 32, 32}), fp({32, 16, 3, 3})}, {1, 32, 32, 32});
  // 同一对象改用完全不同的输入签名，证明推导不读取声明输出元数据。
  shapeIs(conv, {fp({2, 3, 7, 8}), fp({5, 3, 3, 3})}, {2, 5, 7, 8});
  attrs.strides = {2, 2};
  Conv strided(2, {1, 2}, {3}, attrs);
  shapeIs(strided, {fp({1, 3, 7, 8}), fp({4, 3, 3, 3})}, {1, 4, 4, 4});
  attrs.strides = {2, 3};
  attrs.dilations = {2, 1};
  attrs.pads = {1, 0, 2, 1};
  Conv asymmetric(3, {1, 2}, {3}, attrs);
  shapeIs(asymmetric, {fp({1, 3, 9, 11}), fp({4, 3, 3, 2})}, {1, 4, 4, 4});
  attrs = {};
  attrs.groups = 2;
  Conv grouped(4, {1, 2}, {3}, attrs);
  shapeIs(grouped, {fp({1, 4, 5, 5}), fp({6, 2, 3, 3})}, {1, 6, 3, 3});
  rejects([&] { conv.infer({fp({1, 4, 5, 5}), fp({6, 2, 3, 3})}); }, "invalid_shape");
  rejects([&] { grouped.infer({fp({1, 4, 5, 5}), fp({5, 2, 3, 3})}); }, "invalid_shape");
  rejects([&] { conv.infer({fp({1, 3, 1, 1}), fp({4, 3, 7, 7})}); }, "invalid_shape");
  rejects([&] { conv.infer({fp({1, 3, 5}), fp({4, 3, 3, 3})}); }, "invalid_shape");
  attrs.groups = 0;
  rejects([&] { Conv bad(5, {1, 2}, {3}, attrs); }, "invalid_attribute");
  traitIs(conv, false, true, false);
}

// 原物理排列误用由类型系统拒绝；仍逐项覆盖全部逻辑轴约定及非法枚举。
void layoutTests() {
  const std::array<AxisOrder, 6> orders{AxisOrder::NCHW,
                                        AxisOrder::NHWC,
                                        AxisOrder::OIHW,
                                        AxisOrder::HWIO,
                                        static_cast<AxisOrder>(-1),
                                        static_cast<AxisOrder>(99)};
  for (const auto order : orders) {
    ConvAttrs data;
    data.dataAxes = order;
    if (order == AxisOrder::NCHW) {
      Conv valid(1, {1, 2}, {3}, data);
      check(valid.attrs().dataAxes == AxisOrder::NCHW, "typed Conv data axes lost");
    } else {
      rejects([&] { Conv invalid(1, {1, 2}, {3}, data); }, "unsupported_layout");
    }
    ConvAttrs weight;
    weight.weightAxes = order;
    if (order == AxisOrder::OIHW) {
      Conv valid(1, {1, 2}, {3}, weight);
      check(valid.attrs().weightAxes == AxisOrder::OIHW, "typed Conv weight axes lost");
    } else {
      rejects([&] { Conv invalid(1, {1, 2}, {3}, weight); }, "unsupported_layout");
    }
    ResizeAttrs resize;
    resize.dataAxes = order;
    if (order == AxisOrder::NCHW) {
      Resize valid(1, {1}, {2}, resize);
      check(valid.attrs().dataAxes == AxisOrder::NCHW, "typed Resize axes lost");
    } else {
      rejects([&] { Resize invalid(1, {1}, {2}, resize); }, "unsupported_layout");
    }
    MaxPoolAttrs pool;
    pool.dataAxes = order;
    if (order == AxisOrder::NCHW) {
      MaxPool valid(1, {1}, {2}, pool);
      check(valid.attrs().dataAxes == AxisOrder::NCHW, "typed Pool axes lost");
    } else {
      rejects([&] { MaxPool invalid(1, {1}, {2}, pool); }, "unsupported_layout");
    }
  }
  ResizeAttrs resize;
  resize.mode = static_cast<ResizeMode>(99);
  rejects([&] { Resize invalid(1, {1}, {2}, resize); }, "unsupported");
  resize = {};
  resize.coordinateTransform = static_cast<CoordinateTransform>(99);
  rejects([&] { Resize invalid(1, {1}, {2}, resize); }, "unsupported");
  resize = {};
  resize.nearestMode = static_cast<NearestMode>(99);
  rejects([&] { Resize invalid(1, {1}, {2}, resize); }, "unsupported");
  resize = {};
  resize.aspectPolicy = static_cast<AspectPolicy>(99);
  rejects([&] { Resize invalid(1, {1}, {2}, resize); }, "unsupported");
  MaxPoolAttrs pool;
  pool.padding = static_cast<PoolPadding>(99);
  rejects([&] { MaxPool invalid(1, {1}, {2}, pool); }, "unsupported");
}

// 保留空维/标量广播、重复使用、类型冲突、未知 dtype 和数值约束反例。
void pointwiseTests() {
  Add add(1, {1, 1}, {2});
  Mul mul(2, {1, 2}, {3});
  ReLU relu(3, {1}, {2});
  Sigmoid sigmoid(4, {1}, {2});
  shapeIs(add, {fp({2, 1, 4}), fp({3, 1})}, {2, 3, 4});
  shapeIs(add, {fp({0, 3}), fp({1, 3})}, {0, 3});
  shapeIs(mul, {fp({}), fp({2, 3})}, {2, 3});
  shapeIs(mul, {{{2, 1}, DataType::Int32}, {{1, 4}, DataType::Int32}}, {2, 4}, DataType::Int32);
  shapeIs(relu, {fp({1, 4, 7, 9})}, {1, 4, 7, 9});
  shapeIs(sigmoid, {fp({})}, {});
  shapeIs(sigmoid, {{{3}, DataType::Float64}}, {3}, DataType::Float64);
  rejects([&] { add.infer({fp({2, 3}), fp({2})}); }, "invalid_shape");
  rejects([&] { mul.infer({fp({2, 3}), fp({4, 3})}); }, "invalid_shape");
  rejects([&] { add.infer({fp({2}), {{2}, DataType::Int32}}); }, "invalid_dtype");
  rejects([&] { relu.infer({fp({-1, 3})}); }, "shape_unresolved");
  // 公共 validate 先拒绝未知类型和签名数量，准确核对现行错误边界。
  rejects([&] { relu.infer({{{2}, DataType::Unknown}}); }, "dtype_unresolved");
  rejects([&] { sigmoid.infer({{{2}, DataType::Int32}}); }, "unsupported_dtype");
  rejects([&] { relu.infer({fp({2}), fp({2})}); }, "port_invalid");
  rejects([&] { Add invalid(8, {1}, {2}); });
  NumericalRequirements numerics;
  numerics.allowedTypes = {DataType::Float16};
  ReLU restricted(9, {1}, {2}, {}, numerics);
  rejects([&] { restricted.infer({fp({2})}); });
  check(add.inputs() == std::vector<TensorId>({1, 1}), "duplicate operand uses must survive");
  traitIs(add, true, false, true);
  traitIs(mul, true, false, true);
  traitIs(relu, true, false, false);
  traitIs(sigmoid, true, false, false);
}

// 正步长右开切片允许空结果，但不截断越界或归一化负索引。
void sliceTests() {
  Slice slice(1, {1}, {2}, {{0, 1, 0, 1}, {2, 7, 5, 7}, {1, 2, 2, 3}});
  shapeIs(slice, {fp({2, 8, 5, 7})}, {2, 3, 3, 2});
  Slice odd(2, {1}, {2}, {{2}, {7}, {2}});
  shapeIs(odd, {fp({8})}, {3}); // 实际索引为 2、4、6，右开边界为 7。
  Slice empty(3, {1}, {2}, {{2}, {2}, {1}});
  shapeIs(empty, {fp({5})}, {0});
  rejects([&] { Slice bad(4, {1}, {2}, {{0}, {4}, {0}}); }, "invalid_attribute");
  rejects([&] { Slice bad(4, {1}, {2}, {{-1}, {4}, {1}}); }, "invalid_attribute");
  rejects([&] { Slice bad(4, {1}, {2}, {{3}, {2}, {1}}); }, "invalid_attribute");
  rejects([&] { Slice bad(4, {1}, {2}, {{0}, {4, 5}, {1}}); }, "invalid_attribute");
  rejects([&] { odd.infer({fp({6})}); }, "invalid_shape");
  rejects([&] { odd.infer({fp({2, 8})}); }, "invalid_shape");
  traitIs(slice, false, false, false);
}

// 连接顺序、合法负轴、非连接维一致性和受检尺寸累加必须同时保留。
void concatTests() {
  Concat concat(1, {1, 2, 3}, {4}, {1});
  shapeIs(concat, {fp({1, 2, 2, 3}), fp({1, 3, 2, 3}), fp({1, 4, 2, 3})}, {1, 9, 2, 3});
  Concat negative(2, {1, 2}, {3}, {-1});
  shapeIs(negative, {fp({2, 3}), fp({2, 4})}, {2, 7});
  rejects([&] { negative.infer({fp({2, 3}), fp({4, 3})}); }, "invalid_shape");
  rejects([&] { negative.infer({fp({2, 3}), fp({3})}); }, "invalid_shape");
  Concat outOfRange(3, {1, 2}, {3}, {2});
  rejects([&] { outOfRange.infer({fp({2, 3}), fp({2, 3})}); }, "invalid_attribute");
  Concat overflow(4, {1, 2}, {3}, {0});
  rejects([&] { overflow.infer({fp({std::numeric_limits<std::int64_t>::max()}), fp({1})}); },
          "shape_overflow");
  rejects([&] { Concat invalid(5, {}, {1}); }, "invalid_ports");
  traitIs(concat, false, false, false);
}

// 固定 2x 政策之外逐项拒绝，不能只改输出尺寸冒充支持更一般的重采样。
void resizeTests() {
  Resize resize(1, {1}, {2});
  shapeIs(resize, {fp({1, 7, 3, 5})}, {1, 7, 6, 10});
  shapeIs(resize, {{{2, 3, 1, 2}, DataType::Int8}}, {2, 3, 2, 4}, DataType::Int8);
  rejects([&] { resize.infer({fp({1, 3, 5})}); }, "invalid_shape");
  rejects([&] { resize.infer({fp({1, 1, std::numeric_limits<std::int64_t>::max(), 1})}); },
          "shape_overflow");
  const auto unsupported = [](ResizeAttrs attrs) {
    rejects([&] { Resize invalid(2, {1}, {2}, attrs); }, "unsupported");
  };
  ResizeAttrs attrs;
  attrs.mode = ResizeMode::Linear;
  unsupported(attrs);
  attrs = {};
  attrs.coordinateTransform = CoordinateTransform::HalfPixel;
  unsupported(attrs);
  attrs = {};
  attrs.nearestMode = NearestMode::Ceil;
  unsupported(attrs);
  attrs = {};
  attrs.scales = {1, 1, 3, 2};
  unsupported(attrs);
  attrs = {};
  attrs.sizes = {1, 1, 6, 6};
  unsupported(attrs);
  attrs = {};
  attrs.axes = {2, 3};
  unsupported(attrs);
  attrs = {};
  attrs.roi = {0, 0, 1, 1};
  unsupported(attrs);
  attrs = {};
  attrs.aspectPolicy = AspectPolicy::NotLarger;
  unsupported(attrs);
  attrs = {};
  attrs.antialias = true;
  unsupported(attrs);
  attrs = {};
  attrs.excludeOutside = true;
  unsupported(attrs);
  attrs = {};
  attrs.cubicCoefficient = -0.5;
  unsupported(attrs);
  attrs = {};
  attrs.extrapolationValue = 1;
  unsupported(attrs);
  traitIs(resize, false, false, false);
}

// floor 窗口和负无穷边界不可被零填充或 indices 输出静默替代。
void poolingTests() {
  MaxPool pool(1, {1}, {2});
  shapeIs(pool, {fp({1, 16, 9, 13})}, {1, 16, 9, 13});
  MaxPoolAttrs attrs;
  attrs.window = {3, 2};
  attrs.strides = {2, 3};
  attrs.pads = {1, 0, 0, 1};
  MaxPool strided(2, {1}, {2}, attrs);
  shapeIs(strided, {fp({1, 4, 6, 7})}, {1, 4, 3, 3});
  attrs.padding = PoolPadding::Zero;
  rejects([&] { MaxPool invalid(3, {1}, {2}, attrs); }, "unsupported");
  attrs = {};
  attrs.returnIndices = true;
  rejects([&] { MaxPool invalid(3, {1}, {2}, attrs); }, "unsupported");
  attrs = {};
  attrs.strides = {0, 1};
  rejects([&] { MaxPool invalid(3, {1}, {2}, attrs); }, "invalid_attribute");
  rejects([&] { pool.infer({{{1, 1, 5, 5}, DataType::Int32}}); }, "unsupported_dtype");
  rejects([&] { pool.infer({fp({1, 1, 5})}); }, "invalid_shape");
  traitIs(pool, false, true, false);
}

void matmulTests() {
  // MatMul 只有程序化构造，不虚构来源绑定；独占返回值经公共协议安全销毁。
  const auto factory = []() -> std::unique_ptr<Operator> {
    return std::make_unique<MatMul>(1, std::vector<TensorId>{1, 2}, std::vector<TensorId>{3});
  };
  auto dynamic = factory();
  check(dynamic_cast<MatMul *>(dynamic.get()) != nullptr,
        "MatMul factory must return its concrete class");
  MatMul matmul(1, {1, 2}, {3});
  shapeIs(matmul, {fp({2, 3}), fp({3, 2})}, {2, 2});
  shapeIs(matmul, {fp({7, 4}), fp({4, 5})}, {7, 5});
  shapeIs(matmul, {fp({2, 0}), fp({0, 3})}, {2, 3});
  rejects([&] { matmul.infer({fp({2, 3}), fp({4, 2})}); }, "invalid_shape");
  shapeIs(matmul, {fp({1, 2, 3}), fp({3, 2})}, {1, 2, 2});
  rejects([&] { matmul.infer({{{2, 3}, DataType::Int32}, {{3, 2}, DataType::Int32}}); },
          "unsupported_dtype");
  rejects([&] { matmul.infer({fp({2, 3}), {{3, 2}, DataType::Float64}}); }, "invalid_dtype");
  traitIs(matmul, false, true, true);
}

// 语义目录单独登记十个具体类，不借用或恢复已删除的 standard 工厂/兼容头。
void registrationTests() {
  Registry registry;
  registerBuiltinOperators(registry);
  check(registry.registrations().size() == 22, "ten concrete semantics must register");
  check(registry.sourceBindings().empty(),
        "semantic registration must not install source adapters");
  for (const auto &registration : registry.registrations()) {
    check(registration.semantic.nameSpace == "opfusion" &&
              registration.semantic.key().find('@') == std::string::npos,
          "wrong mathematical semantic identity");
  }
  registry.freeze("contracts-standard-test/4");
  rejects([&] { registerBuiltinOperators(registry); }, "registry_frozen");
  Registry duplicate;
  registerBuiltinOperators(duplicate);
  rejects([&] { registerBuiltinOperators(duplicate); }, "duplicate_contribution");
}
} // namespace

int main() {
  try {
    convolutionTests();
    layoutTests();
    pointwiseTests();
    sliceTests();
    concatTests();
    resizeTests();
    poolingTests();
    matmulTests();
    registrationTests();
    std::cout << "standard operator checks: " << checks << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "standard operator test failure: " << error.what() << '\n';
    return 1;
  }
}
