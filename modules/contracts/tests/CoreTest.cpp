#include "opfusion/common/Error.h"
#include "opfusion/graph/DAG.h"
#include "opfusion/registry/Registry.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// 纯契约回归：逻辑大小、端口依赖、图不变量和 Registry 装配/冻结，不执行设备。

using namespace opfusion;
static_assert(std::is_enum_v<Layout>);
static_assert(!std::is_convertible_v<const char *, Layout>);
static_assert(!std::is_convertible_v<int, Layout>);
static_assert(!std::is_copy_assignable_v<DAG>);
static_assert(!std::is_move_assignable_v<DAG>);
static_assert(!std::is_move_constructible_v<DAG>);
namespace {
int checks = 0;
void check(bool condition, const char *message) {
  ++checks;
  if (!condition) {
    throw std::runtime_error(message);
  }
}
void fails(const std::string &code, const std::function<void()> &action) {
  try {
    action();
  } catch (const Error &error) {
    check(error.code() == code, ("expected " + code + ", got " + error.code()).c_str());
    return;
  }
  throw std::runtime_error("expected error: " + code);
}
/// 测试专用首输入复制算子，拥有端口和 copies；不构造生产算子的兼容壳。
/// 额外声明输入仍参与 DAG 的使用边，但数学结果只读取首输入。
class Echo final : public Operator {
public:
  /// 按值保存端口；故意不强制 outputs.size()==copies，以测试 DAG 的推导数量核对。
  Echo(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
       std::size_t copies = 1)
      : Operator(id, {"test", "Echo"}, std::move(inputs), std::move(outputs)), mCopies(copies) {
    validateAttributes();
  }
  /// 只借用输入并复制首签名，不读取输出声明；未知 dtype 遵循公共校验失败。
  std::vector<TensorSignature> infer(const std::vector<TensorSignature> &inputs) const override {
    validate(inputs);
    return std::vector<TensorSignature>(mCopies, inputs.front());
  }
  /// 测试行为是纯逐元素复制，无广播和归约。
  Traits traits() const override { return {true, true, false, false}; }
  /// 每个推导输出都读取首输入的相同坐标；不能为未使用的额外输入伪造读取。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &inputs) const override {
    const auto outputs = infer(inputs);
    IndexPiece piece;
    std::vector<IndexExpression> coordinates;
    for (std::size_t axis = 0; axis < inputs.front().shape.size(); ++axis) {
      piece.domain.push_back({"i" + std::to_string(axis), 0, inputs.front().shape[axis], false});
      coordinates.push_back({{{axis, 1, 1}}, 0});
    }
    piece.inputs = {{0, coordinates, FillValue::None, inputs.front().axes, true}};
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      piece.outputs.push_back({port, coordinates});
    }
    return {SemanticStatus::Known, {}, {std::move(piece)}};
  }

protected:
  /// 首输入与正副本数是实际复制行为的前提，输出数量冲突由图检查捕获。
  void validateAttributes() const override {
    require(!inputs().empty() && mCopies > 0, "port_invalid", "Echo needs input and copies");
  }

private:
  std::size_t mCopies;
};
static_assert(!std::is_abstract_v<Echo>);

// 未提供轴或来源角色的值必须保持未知，不借用临时 shape。
Tensor value(TensorId id, Shape shape = {2, 3}) {
  return Tensor(id, "test", std::move(shape), DataType::Float32);
}
// 同时保留重复端口、多输出、扇出，以及缺值/重复生产/环/声明不一致反例。
void graphs() {
  std::vector<std::unique_ptr<Operator>> nodes;
  nodes.push_back(
      std::make_unique<Echo>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2, 3}, 2));
  nodes.push_back(std::make_unique<Echo>(2, std::vector<TensorId>{2, 2}, std::vector<TensorId>{4}));
  nodes.push_back(std::make_unique<Echo>(3, std::vector<TensorId>{2}, std::vector<TensorId>{5}));
  DAG graph("g", "fanout", {value(1), value(2), value(3), value(4), value(5)}, std::move(nodes),
            std::vector<TensorId>{1, 3, 4, 5});
  check(graph.edges().size() == 3, "duplicate uses must produce separate edges");
  check(graph.tensors().size() == 5, "uses must not duplicate logical tensors");
  check(graph.edges()[0].tensor == graph.edges()[1].tensor, "duplicate ports share tensor");
  check(graph.edges()[0].inputPort != graph.edges()[1].inputPort, "consumer ports retained");
  check(graph.externalInputs() == std::vector<TensorId>{1}, "external boundary");
  check(graph.terminalOutputs() == std::vector<TensorId>({3, 4, 5}), "terminal values");
  check(graph.modelOutputs()->front() == 1, "input may also be model output");
  fails("duplicate_tensor", [] { DAG bad("g", "", {value(1), value(1)}, {}); });
  fails("tensor_missing", [] {
    std::vector<std::unique_ptr<Operator>> n;
    n.push_back(std::make_unique<Echo>(1, std::vector<TensorId>{9}, std::vector<TensorId>{2}));
    DAG bad("g", "", {value(2)}, std::move(n));
  });
  fails("multiple_producers", [] {
    std::vector<std::unique_ptr<Operator>> n;
    for (int i = 1; i <= 2; ++i) {
      n.push_back(std::make_unique<Echo>(i, std::vector<TensorId>{1}, std::vector<TensorId>{2}));
    }
    DAG bad("g", "", {value(1), value(2)}, std::move(n));
  });
  fails("graph_cycle", [] {
    std::vector<std::unique_ptr<Operator>> n;
    n.push_back(std::make_unique<Echo>(1, std::vector<TensorId>{2}, std::vector<TensorId>{1}));
    n.push_back(std::make_unique<Echo>(2, std::vector<TensorId>{1}, std::vector<TensorId>{2}));
    DAG bad("g", "", {value(1), value(2)}, std::move(n));
  });
  fails("port_invalid", [] {
    std::vector<std::unique_ptr<Operator>> n;
    n.push_back(std::make_unique<Echo>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2, 3}));
    DAG bad("g", "", {value(1), value(2), value(3)}, std::move(n));
  });
  fails("shape_mismatch", [] {
    std::vector<std::unique_ptr<Operator>> n;
    n.push_back(std::make_unique<Echo>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2}));
    DAG bad("g", "", {value(1), value(2, {3, 2})}, std::move(n));
  });
}
// 语义声明与来源绑定分开：未登记目标、通配歧义及冻结后修改必须明确失败。
void registry() {
  Registry r;
  auto factory = [](const SourceNode &) {
    return std::make_unique<Echo>(1, std::vector<TensorId>{1}, std::vector<TensorId>{2});
  };
  const SemanticIdentity echo{"test", "Echo"};
  const SourceSelector selector{"benchmark-json/1", "Echo", std::nullopt};
  r.add({echo});
  check(r.sourceBindings().empty(), "semantic registration must not invent a source binding");
  fails("binding_target_missing", [&] { r.bind({{"test", "Missing"}, {selector}, factory}); });
  fails("binding_invalid", [&] { r.bind({echo, {}, factory}); });
  fails("factory_missing", [&] { r.bind({echo, {selector}, {}}); });
  fails("binding_invalid", [&] { r.bind({echo, {{"", "Echo", std::nullopt}}, factory}); });
  fails("binding_ambiguous",
        [&] { r.bind({echo, {selector, {"benchmark-json/1", "Echo", "operation"}}, factory}); });
  check(r.sourceBindings().empty(), "failed binding must not publish a partial selector list");
  r.bind({echo, {selector}, factory});
  fails("duplicate_contribution", [&] { r.add({echo}); });
  r.add({{"test", "Other"}});
  fails("binding_ambiguous",
        [&] { r.bind({{"test", "Other"}, {{"benchmark-json/1", "Echo", "operation"}}, factory}); });
  check(r.sourceBindings().size() == 1, "ambiguous cross-binding must leave the original intact");
  SourceNode source;
  source.format = "benchmark-json/1";
  source.type = "Echo";
  fails("registry_not_frozen", [&] { (void)r.create(source); });
  fails("id_invalid", [&] { r.freeze(""); });
  check(!r.frozen(), "failed freeze must keep the registry mutable");
  r.freeze("test-registry");
  check(r.create(source)->semantic().name == "Echo", "registered factory");
  fails("registry_frozen", [&] { r.add({{"test", "New"}}); });
  fails("registry_frozen", [&] { r.bind({echo, {selector}, factory}); });
  fails("registry_frozen", [&] { r.freeze("another-registry"); });
  source.type = "Missing";
  fails("operator_unknown", [&] { (void)r.create(source); });
  check(
      !r.create(SourceNode{"benchmark-json/1", "Echo", {}, "", 0, {}})->traits().reduction.value(),
      "traits retained");

  // 匹配成功也不能信任工厂：完整语义键（命名空间与名称）及非空独占结果都要核对。
  Registry wrong;
  const SemanticIdentity changedIdentity{"different-namespace", "Echo"};
  wrong.add({changedIdentity});
  wrong.bind({changedIdentity, {selector}, factory});
  wrong.freeze("wrong-factory");
  source.type = "Echo";
  fails("factory_invalid", [&] { (void)wrong.create(source); });
  Registry empty;
  empty.add({echo});
  empty.bind(
      {echo, {selector}, [](const SourceNode &) -> std::unique_ptr<Operator> { return nullptr; }});
  empty.freeze("empty-factory");
  fails("factory_invalid", [&] { (void)empty.create(source); });

  check(std::string(contractVersion) == "6.1.0",
        "operator identity and normalization require contract API 5");
  check(std::string(operatorApiVersion) == "5.0.0", "rich access relations require operator API 5");
}
} // namespace
int main() {
  try {
    for (const auto &item :
         std::vector<std::pair<Layout, std::string>>{{Layout::RowMajor, "row_major"},
                                                     {Layout::ColumnMajor, "column_major"},
                                                     {Layout::NCHW, "NCHW"},
                                                     {Layout::NHWC, "NHWC"},
                                                     {Layout::OIHW, "OIHW"},
                                                     {Layout::HWIO, "HWIO"},
                                                     {Layout::Strided, "strided"},
                                                     {Layout::Opaque, "opaque"}}) {
      check(toString(item.first) == item.second, "layout enum serialization");
    }
    fails("layout_invalid", [] { (void)toString(static_cast<Layout>(-1)); });
    fails("dtype_invalid", [] {
      NumericalRequirements requirements;
      requirements.allowedTypes = {static_cast<DataType>(-1)};
      requirements.validate();
    });
    fails("dtype_invalid", [] {
      NumericalRequirements requirements;
      requirements.accumulationType = static_cast<DataType>(-1);
      requirements.validate();
    });
    check(logicalBytes({512, 1024}, DataType::Float16) == 1048576, "FP16 input bytes");
    check(logicalBytes({1024, 2048}, DataType::Float16) == 4194304, "FP16 weight bytes");
    check(!logicalBytes({1}, DataType::Unknown), "unknown dtype is not zero bytes");
    check(elements({}) == 1, "scalar element count");
    check(elements({std::numeric_limits<std::int64_t>::max(), 3, 0}) == 0,
          "empty tensor does not overflow");
    fails("shape_unresolved", [] { (void)elements({-1}); });
    fails("size_overflow", [] { (void)elements({std::numeric_limits<std::int64_t>::max(), 3}); });
    fails("size_overflow", [] { (void)checkedAdd(std::numeric_limits<Bytes>::max(), 1); });
    graphs();
    registry();
    std::cout << checks << " contracts checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
