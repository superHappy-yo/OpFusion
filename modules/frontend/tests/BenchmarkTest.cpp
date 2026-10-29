#include "opfusion/common/Error.h"
#include "opfusion/frontend/BuiltinSourceBindings.h"
#include "opfusion/frontend/ModelImporter.h"
#include "opfusion/graph/Axis.h"
#include "opfusion/graph/DAG.h"
#include "opfusion/graph/IndexSemantics.h"
#include "opfusion/graph/Operator.h"
#include "opfusion/graph/Tensor.h"
#include "opfusion/operators/Add.h"
#include "opfusion/operators/Concat.h"
#include "opfusion/operators/Conv.h"
#include "opfusion/operators/MaxPool.h"
#include "opfusion/operators/Mul.h"
#include "opfusion/operators/ReLU.h"
#include "opfusion/operators/Resize.h"
#include "opfusion/operators/Sigmoid.h"
#include "opfusion/operators/Slice.h"
#include "opfusion/registry/BuiltinOperators.h"
#include "opfusion/registry/Registry.h"
#include "opfusion/registry/Source.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace opfusion;
using namespace opfusion::frontend;
using Json = nlohmann::json;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

/// 单线程 locale 反例夹具；下划线虚函数名由 std::numpunct 协议规定，不能机械改名。
class GroupedDigits final : public std::numpunct<char> {
protected:
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\1"; }
};

/// 中立测试语义：每个输入投影到同端口输出，不属于生产算子集合。
/// 推导与索引关系都只读输入，不读声明输出的数量/shape/dtype 作 oracle。
class Projection final : public Operator {
public:
  /// 取得端口列表所有权；rejectInfer 只注入推导失败，不改变工厂失败路径。
  Projection(OperatorId id, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
             bool rejectInfer)
      : Operator(id, {"frontend-test", "projection"}, std::move(inputs), std::move(outputs)),
        mRejectInfer(rejectInfer) {
    validateAttributes();
  }

  /// 同步借用等签名输入；返回一输入一输出的副本，非法输入或注入故障抛出 Error。
  std::vector<TensorSignature>
  infer(const std::vector<TensorSignature> &signatures) const override {
    validate(signatures);
    require(!mRejectInfer, "test_infer_rejected",
            "test inference rejected; unrelated node 1 (mentioned inside the error)");
    require(!signatures.empty(), "test_input_missing", "projection needs an input");
    for (const auto &signature : signatures)
      require(signature == signatures.front(), "test_input_mismatch",
              "test inputs must have equal signatures");
    return signatures;
  }
  /// 本夹具不为 trait 提供证明；未知项保持 nullopt，不补成 false 或支持结论。
  Traits traits() const override { return {}; }

  /// 每个输入单独形成非归约的逻辑元素半开域，保持对应端口与轴的恒等访问。
  IndexSemantics indexSemantics(const std::vector<TensorSignature> &signatures) const override {
    const auto inferred = infer(signatures);
    IndexSemantics result;
    result.status = SemanticStatus::Known;
    for (std::size_t port = 0; port < inferred.size(); ++port) {
      IndexPiece piece;
      std::vector<IndexExpression> coordinates;
      for (std::size_t dimension = 0; dimension < inferred[port].shape.size(); ++dimension) {
        piece.domain.push_back(
            {"axis" + std::to_string(dimension), 0, inferred[port].shape[dimension], false});
        IndexExpression coordinate;
        coordinate.terms.push_back({dimension, 1, 1});
        coordinates.push_back(std::move(coordinate));
      }
      InputAccess input;
      input.port = port;
      input.coordinates = coordinates;
      input.axes = signatures[port].axes;
      input.preservesAxes = true;
      piece.inputs.push_back(std::move(input));
      piece.outputs.push_back({port, std::move(coordinates)});
      result.pieces.push_back(std::move(piece));
    }
    return result;
  }

protected:
  /// 只校验本语义所需输入；输出数量特意留给 DAG 对比独立推导，以保留虚假输出反例。
  void validateAttributes() const override {
    require(!inputs().empty(), "test_input_missing", "projection needs an input");
  }

private:
  bool mRejectInfer;
};

std::vector<TensorId> references(const SourceValue &values) {
  std::vector<TensorId> result;
  for (const auto &value : values.array()) {
    if (value.isNull())
      continue;
    const auto id = value.integer();
    require(id >= 0, "test_reference_invalid", "negative tensor reference");
    result.push_back(static_cast<TensorId>(id));
  }
  return result;
}

Registry registry(std::function<void(const SourceNode &)> inspect = {}) {
  Registry result;
  // 语义与来源分开装配；inspect 捕获的测试状态须活到同步导入完成。
  result.add({{"frontend-test", "projection"}});
  result.bind({{"frontend-test", "projection"},
               {{"benchmark-json/1", "FixtureRoute", "PROJECT"}},
               [inspect = std::move(inspect)](const SourceNode &source) {
                 if (inspect)
                   inspect(source);
                 const auto &inputs = source.fields.at("inputs");
                 if (const auto *reject = inputs.find("reject_factory"))
                   require(!reject->boolean(), "test_factory_rejected", "test factory rejected");
                 bool rejectInfer = false;
                 if (const auto *reject = inputs.find("reject_infer"))
                   rejectInfer = reject->boolean();
                 // Typed attribute validation belongs to this injected factory.
                 if (const auto *attribute = inputs.find("typed_flag"))
                   (void)attribute->boolean();
                 return std::make_unique<Projection>(
                     static_cast<OperatorId>(source.index) + 1, references(inputs.at("tensors")),
                     references(source.fields.at("outputs").at("tensors")), rejectInfer);
               }});
  result.freeze("frontend-test-registry/1");
  return result;
}

void expectError(const std::function<void()> &action, const std::string &code = {},
                 const std::vector<std::string> &context = {}) {
  try {
    action();
  } catch (const Error &error) {
    check(code.empty() || error.code() == code,
          "unexpected error code " + error.code() + ": " + error.what());
    for (const auto &fragment : context)
      check(std::string(error.what()).find(fragment) != std::string::npos,
            "missing error context: " + fragment + " in " + error.what());
    return;
  }
  throw std::runtime_error("expected opfusion::Error, but import succeeded");
}

bool hasDiagnostic(const ImportResult &result, const std::string &code) {
  return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const auto &diagnostic) { return diagnostic.code == code; });
}

std::string withRawNumber(Json document, const std::string &token) {
  document["nodes"][0]["inputs"]["number"] = "RAW_NUMBER";
  auto text = document.dump();
  const auto start = text.find("\"RAW_NUMBER\"");
  check(start != std::string::npos, "raw-number test marker missing");
  text.replace(start, std::string("\"RAW_NUMBER\"").size(), token);
  return text;
}

// 中立来源的名称/stride 不触发布局推断；多输出、重复使用与完整递归元数据必须同时保留。
void preserveGraph(const Json &document) {
  std::vector<SourceNode> observed;
  const auto factories = registry([&](const SourceNode &source) { observed.push_back(source); });
  const auto result = ModelImporter(factories).importText(document.dump(), ModelFormat::JsonGraph,
                                                          "fixture-memory");
  const auto &graph = *result.graph;
  check(result.profile == "benchmark-json/1", "profile identity changed");
  check(graph.name() == "neutral-nhwc-named-fixture", "graph name lost");
  check(graph.registryId() == factories.id(), "registry identity lost");
  check(graph.tensors().size() == 5, "unused tensors must be retained");
  check(graph.tensor(99).name() == "unused", "UID was mistaken for table index");
  check(graph.tensor(700).shape() == Shape({1, 3, 5, 7}), "logical dims were reordered");
  check(graph.tensor(700).dtype() == DataType::Float32, "dtype mapping failed");
  check(result.tensorOrigins.size() == 5, "tensor provenance is incomplete");
  const auto &origin = result.tensorOrigins.at(700);
  check(origin.strides == Shape({105, 1, 21, 3}), "source strides were changed");
  check(origin.alignment == 16, "alignment hint lost");
  check(origin.virtualTensor == false, "virtual hint lost");
  check(origin.runtimePassByValue == false, "pass-by-value hint lost");
  check(graph.nodes().size() == 2, "node count changed");
  check(result.nodeOrigins.size() == 2 && result.nodeOrigins[1].id == 2 &&
            result.nodeOrigins[1].index == 1 && result.nodeOrigins[1].name == "second",
        "node provenance changed");
  check(graph.node(2).inputs() == std::vector<TensorId>({42, 42}),
        "repeated operands or null ports were changed");
  check(graph.node(2).outputs() == std::vector<TensorId>({170, 990}),
        "multi-output/null source handling changed");
  check(graph.edges().size() == 2, "repeated input uses must produce two edges");
  for (std::size_t i = 0; i < graph.edges().size(); ++i) {
    const auto &edge = graph.edges()[i];
    check(edge.producer == 1 && edge.outputPort == 0 && edge.consumer == 2 && edge.inputPort == i &&
              edge.tensor == 42,
          "port-preserving edge derivation failed");
  }
  check(std::find(graph.externalInputs().begin(), graph.externalInputs().end(), 700) !=
            graph.externalInputs().end(),
        "external input was lost");
  check(graph.terminalOutputs() == std::vector<TensorId>({170, 990}),
        "terminal outputs were not inferred");
  check(!graph.modelOutputs().has_value(), "terminal outputs are not explicit model outputs");
  check(observed.size() == 2 && observed[0].index == 0 && observed[0].type == "FixtureRoute" &&
            observed[0].operation == "PROJECT" && observed[0].format == "benchmark-json/1",
        "generic factory source is incomplete");
  check(observed[1].fields.at("inputs").at("tensors").array()[2].isNull(),
        "null source port must reach the factory unchanged");
  const auto &metadata = observed[0].fields.at("extra_metadata");
  check(metadata.at("flag").boolean() && metadata.at("nested").array()[0].isNull() &&
            metadata.at("nested").array()[1].string() == "opaque",
        "complete, recursive node source was not preserved");
  for (const auto &code : {"source", "external_inputs_inferred", "terminal_outputs_inferred",
                           "weight_contents_unavailable", "execution_conditions_unestablished",
                           "candidates_not_searched", "physical_hints_only"})
    check(hasDiagnostic(result, code), std::string("missing diagnostic: ") + code);
}

// 必须使用原始数字 token，避免测试 JSON 序列化先替实现舍入/转换而掩盖整数溢出。
void sourceNumbers(const Json &document) {
  for (const auto &token : {"9223372036854775807", "-9223372036854775808"}) {
    bool seen = false;
    const auto factories = registry([&](const SourceNode &source) {
      if (source.index != 0)
        return;
      const auto value = source.fields.at("inputs").at("number").integer();
      check(value == (token[0] == '-' ? std::numeric_limits<std::int64_t>::min()
                                      : std::numeric_limits<std::int64_t>::max()),
            "signed 64-bit limit was not preserved");
      seen = true;
    });
    (void)ModelImporter(factories).importText(withRawNumber(document, token));
    check(seen, "integer attribute never reached factory");
  }
  bool seenDouble = false;
  const auto factories = registry([&](const SourceNode &source) {
    if (source.index == 0)
      seenDouble = std::holds_alternative<double>(source.fields.at("inputs").at("number").value);
  });
  const ModelImporter importer(factories);
  (void)importer.importText(withRawNumber(document, "1.0"));
  check(seenDouble, "floating token was coerced to integer");
  for (const auto &token : {"9223372036854775808", "18446744073709551615", "18446744073709551616",
                            "-9223372036854775809"})
    expectError([&] { (void)importer.importText(withRawNumber(document, token)); },
                "benchmark_integer_range");
  for (const auto &token : {"1e400", "-1e400"})
    expectError([&] { (void)importer.importText(withRawNumber(document, token)); },
                "benchmark_number_range");
}

void groupedGraphIdentity(const std::string &text) {
  // 只有此单线程测试改全局 locale，成功或异常都恢复；生产导入器不得依赖数字分组设置。
  const auto previous = std::locale();
  std::locale::global(std::locale(previous, new GroupedDigits));
  try {
    const auto factories = registry();
    const auto result = ModelImporter(factories).importText(text);
    const auto &id = result.graph->id();
    check(id.size() == 32 &&
              std::all_of(id.begin(), id.end(),
                          [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
          "ambient C++ locale changed opaque graph-token encoding");
  } catch (...) {
    std::locale::global(previous);
    throw;
  }
  std::locale::global(previous);
}

void localizedSourceNumbers(const Json &document) {
  const auto *active = std::setlocale(LC_NUMERIC, nullptr);
  check(active != nullptr, "cannot inspect current numeric locale");
  const std::string previous = active;
  bool selected = false;
  for (const auto *localeName : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"}) {
    if (std::setlocale(LC_NUMERIC, localeName) != nullptr &&
        std::localeconv()->decimal_point[0] == ',') {
      selected = true;
      break;
    }
  }
  if (!selected) {
    check(std::setlocale(LC_NUMERIC, previous.c_str()) != nullptr, "cannot restore numeric locale");
    std::cout << "SKIP CommaDecimalLocale: no installed comma-decimal C locale\n";
    return;
  }
  try {
    sourceNumbers(document);
  } catch (...) {
    check(std::setlocale(LC_NUMERIC, previous.c_str()) != nullptr, "cannot restore numeric locale");
    throw;
  }
  check(std::setlocale(LC_NUMERIC, previous.c_str()) != nullptr, "cannot restore numeric locale");
}

void malformedDocuments(const Json &base) {
  const auto factories = registry();
  const auto reject = [&](const Json &document, const std::string &code = {}) {
    expectError([&] { (void)ModelImporter(factories).importText(document.dump()); }, code);
  };
  auto document = base;
  document["tensors"][1]["uid"] = document["tensors"][0]["uid"];
  reject(document, "benchmark_duplicate_uid");
  document = base;
  document["tensors"][0]["uid"] = true;
  reject(document, "benchmark_schema_type");
  document = base;
  document["tensors"][0]["uid"] = 7.0;
  reject(document, "benchmark_schema_type");
  document = base;
  document["tensors"][0]["data_type"] = "half";
  reject(document, "benchmark_profile");
  document = base;
  document["tensors"][0]["dims"][0] = "1";
  reject(document, "benchmark_schema_type");
  document = base;
  document["tensors"][0]["alignment"] = -1;
  reject(document, "benchmark_integer_range");
  document = base;
  document["tensors"][0]["strides"][0] = 1.0;
  reject(document, "benchmark_schema_type");
  document = base;
  document["tensors"][0]["virtual"] = "true";
  reject(document, "benchmark_schema_type");
  document = base;
  document["nodes"][0]["type"] = 1;
  reject(document, "benchmark_schema_type");
  document = base;
  document["nodes"][0]["inputs"]["operation"] = nullptr;
  reject(document, "benchmark_schema_type");
  document = base;
  document["model_outputs"] = Json::array({170});
  reject(document, "benchmark_schema_field");
  document = base;
  document.erase("tensors");
  reject(document, "benchmark_schema_missing");
  document = base;
  document["nodes"] = Json::object();
  reject(document, "benchmark_schema_type");
  for (const auto &key : {"compute_data_type", "intermediate_data_type", "io_data_type"}) {
    document = base;
    document[key] = "half";
    reject(document, "benchmark_profile");
  }
  document = base;
  document["is_override_shape_enabled"] = true;
  reject(document, "benchmark_profile");
  document = base;
  document["is_override_shape_enabled"] = "false";
  reject(document, "benchmark_schema_type");
  document = base;
  document["min_required_engine_api_version"] = "1.0.0";
  reject(document, "benchmark_schema_type");
  document = base;
  document["min_required_engine_api_version"]["major"] = 1.0;
  reject(document, "benchmark_schema_type");
  document = base;
  document["min_required_engine_api_version"]["patch"] = 1;
  reject(document, "benchmark_profile");
  document = base;
  document["tensors"][0]["physical_buffer"] = "not-established";
  reject(document, "benchmark_schema_field");
  expectError(
      [&] {
        (void)ModelImporter(factories).importText(std::string(257, '[') + "0" +
                                                  std::string(257, ']'));
      },
      "benchmark_json_depth");
  const auto text = base.dump();
  const ModelImporter importer(factories);
  expectError([&] { (void)importer.importText("{\"name\":\"duplicate\"," + text.substr(1)); },
              "benchmark_duplicate_key");
  expectError([&] { (void)importer.importText(text + " trailing"); }, "benchmark_json_syntax");
  expectError([&] { (void)importer.importText(text + std::string(1, '\0') + " trailing"); },
              "benchmark_json_syntax");
  expectError([&] { (void)importer.importText("["); }, "benchmark_json_syntax");
}

// 工厂与 DAG 失败各自保留 code/context；错误正文提及 node 1 不得误标为该节点来源。
void delegatedFailures(const Json &base) {
  const auto factories = registry();
  const ModelImporter importer(factories);
  auto document = base;
  document["nodes"][1]["inputs"]["reject_factory"] = true;
  expectError(
      [&] { (void)importer.importText(document.dump(), ModelFormat::JsonGraph, "factory-case"); },
      "test_factory_rejected", {"factory-case", "source node[1]", "second"});
  document = base;
  document["nodes"][1]["inputs"]["reject_infer"] = true;
  expectError(
      [&] { (void)importer.importText(document.dump(), ModelFormat::JsonGraph, "infer-case"); },
      "test_infer_rejected", {"infer-case", "source node[1]", "second"});
  document = base;
  document["nodes"][0]["inputs"]["typed_flag"] = "not-a-boolean";
  expectError([&] { (void)importer.importText(document.dump()); }, {}, {"source node[0]", "first"});
  document = base;
  document["nodes"][0]["type"] = "UnregisteredFixture";
  expectError([&] { (void)importer.importText(document.dump()); }, {}, {"source node[0]", "first"});
  document = base;
  document["nodes"][1]["inputs"]["tensors"][0] = 123456;
  expectError(
      [&] {
        (void)importer.importText(document.dump(), ModelFormat::JsonGraph, "missing-ref-case");
      },
      {}, {"missing-ref-case"});
  document = base;
  document["tensors"][3]["dims"][1] = 4;
  expectError(
      [&] {
        (void)importer.importText(document.dump(), ModelFormat::JsonGraph, "false-output-case");
      },
      {}, {"false-output-case", "source node[1]", "second"});
  document = base;
  document["nodes"][1]["outputs"]["tensors"] = Json::array({170});
  expectError(
      [&] {
        (void)importer.importText(document.dump(), ModelFormat::JsonGraph, "output-count-case");
      },
      {}, {"output-count-case", "source node[1]", "second"});
  document = base;
  document["nodes"][0]["inputs"]["tensors"] = Json::array({170});
  expectError(
      [&] { (void)importer.importText(document.dump(), ModelFormat::JsonGraph, "cycle-case"); }, {},
      {"cycle-case"});
  document = base;
  document["nodes"][1]["outputs"]["tensors"] = Json::array({42, 990});
  expectError([&] { (void)importer.importText(document.dump()); });
}

// 真实内置来源绑定的回归单独装配，不让纯领域/插件测试反向依赖 frontend。
namespace sourceBindingCases {
using namespace opfusion::operators;

TensorSignature fp(Shape shape) {
  return {std::move(shape), DataType::Float32};
}

// 期望 shape 由测试向量提供，不读取 factory 的输出声明作为推导答案。
void shapeIs(const Operator &op, const std::vector<TensorSignature> &inputs, const Shape &expected,
             DataType dtype = DataType::Float32) {
  const auto result = op.infer(inputs);
  check(result.size() == 1 && result[0].shape == expected && result[0].dtype == dtype,
        op.semantic().name + ": wrong inferred shape/dtype");
}

template <class Concrete> void factoryIs(const Registry &registry, const SourceNode &source) {
  const auto actual = registry.create(source);
  check(dynamic_cast<const Concrete *>(actual.get()) != nullptr,
        "source factory returned a different concrete C++ class");
}

// 类型化 SourceValue 夹具直接检验 factory 边界，与上方 JSON token 严格性用例分工。
using V = SourceValue;
V text(const char *value) {
  return V(std::string(value));
}
V integer(std::int64_t value) {
  return V(value);
}
V array(std::initializer_list<V> values) {
  return V(V::Array(values));
}
V object(std::initializer_list<std::pair<const std::string, V>> values) {
  return V(V::Object(values));
}
V &member(V &value, const char *name) {
  return std::get<V::Object>(value.value).at(name);
}
void set(V &value, const char *name, V item) {
  std::get<V::Object>(value.value)[name] = std::move(item);
}

SourceNode parameterized(const char *type, V inputs, V outputs, V parameters) {
  SourceNode result;
  result.format = "benchmark-json/1";
  result.type = type;
  result.name = "fixture";
  result.index = 9;
  result.fields = object({{"type", text(type)},
                          {"name", text("fixture")},
                          {"compute_data_type", text("float")},
                          {"inputs", std::move(inputs)},
                          {"outputs", std::move(outputs)},
                          {"parameters", std::move(parameters)}});
  return result;
}
SourceNode convSource() {
  return parameterized("ConvolutionFwdAttributes",
                       object({{"w_tensor_uid", integer(7)}, {"x_tensor_uid", integer(41)}}),
                       object({{"y_tensor_uid", integer(99)}}),
                       object({{"conv_mode", text("CROSS_CORRELATION")},
                               {"dilation", array({integer(1), integer(1)})},
                               {"stride", array({integer(2), integer(2)})},
                               {"pre_padding", array({integer(1), integer(1)})},
                               {"post_padding", array({integer(1), integer(1)})}}));
}
SourceNode pointwiseSource(const char *operation, bool binary) {
  SourceNode result;
  result.format = "benchmark-json/1";
  result.type = "PointwiseAttributes";
  result.operation = operation;
  result.fields =
      object({{"type", text("PointwiseAttributes")},
              {"compute_data_type", text("float")},
              {"inputs", object({{"operation", text(operation)},
                                 {"in_0_tensor_uid", integer(41)},
                                 {"in_1_tensor_uid", binary ? integer(7) : V{}},
                                 {"in_2_tensor_uid", V{}},
                                 {"axis_tensor_uid", V{}},
                                 {"elu_alpha", V{}},
                                 {"relu_lower_clip", V{}},
                                 {"relu_lower_clip_slope", V{}},
                                 {"relu_upper_clip", V{}},
                                 {"softplus_beta", V{}},
                                 {"swish_beta", V{}}})},
              {"outputs", object({{"out_0_tensor_uid", integer(99)}, {"out_1_tensor_uid", V{}}})}});
  return result;
}
SourceNode resizeSource() {
  return parameterized("ResizeAttributes", object({{"x_tensor_uid", integer(41)}}),
                       object({{"y_tensor_uid", integer(99)}}),
                       object({{"antialias", V(false)},
                               {"axes", array({})},
                               {"coordinate_transformation_mode", text("asymmetric")},
                               {"cubic_coeff_a", V(-0.75)},
                               {"exclude_outside", V(false)},
                               {"extrapolation_value", V(0.0)},
                               {"keep_aspect_ratio_policy", text("stretch")},
                               {"nearest_mode", text("floor")},
                               {"resize_mode", text("nearest")},
                               {"roi_start", array({})},
                               {"roi_end", array({})},
                               {"scales", array({V(1.0), V(1.0), V(2.0), V(2.0)})},
                               {"sizes", array({})}}));
}
SourceNode poolSource() {
  SourceNode result;
  result.format = "benchmark-json/1";
  result.type = "ResampleFwdAttributes";
  result.operation = "maxpool";
  result.fields = object({{"type", text("ResampleFwdAttributes")},
                          {"compute_data_type", text("float")},
                          {"generate_index", V(false)},
                          {"padding_mode", text("neg_inf_pad")},
                          {"resample_mode", text("maxpool")},
                          {"window", array({integer(5), integer(5)})},
                          {"stride", array({integer(1), integer(1)})},
                          {"pre_padding", array({integer(2), integer(2)})},
                          {"post_padding", array({integer(2), integer(2)})},
                          {"x_tensor_uid", integer(41)},
                          {"y_tensor_uid", integer(99)}});
  return result;
}

void sourceTests() {
  Registry registry;
  registerBuiltinOperators(registry);
  registerBuiltinSourceBindings(registry);
  check(registry.registrations().size() == 22, "ten concrete semantics must register");
  for (const auto &registration : registry.registrations()) {
    check(registration.semantic.key().find('@') == std::string::npos, "wrong semantic version");
    if (registration.semantic.name == "MatMul") {
      check(std::none_of(registry.sourceBindings().begin(), registry.sourceBindings().end(),
                         [&](const auto &binding) {
                           return binding.semantic.key() == registration.semantic.key() &&
                                  binding.selectors.front().format == "benchmark-json/1";
                         }),
            "MatMul must not acquire an invented source binding");
    }
  }
  std::size_t bindings = 0;
  for (const auto &binding : registry.sourceBindings()) {
    bindings += binding.selectors.size();
  }
  check(bindings == 31, "exactly nine bounded source bindings expected");
  registry.freeze("standard-test/1");
  Registry duplicate;
  registerBuiltinOperators(duplicate);
  registerBuiltinSourceBindings(duplicate);
  expectError([&] { registerBuiltinOperators(duplicate); }, "duplicate_contribution");

  // 故意打乱 map/UID 次序，验证 [X,W] 的语义角色次序以及 index+1 身份未被排序替换。
  const auto conv = convSource();
  auto instance = registry.create(conv);
  const auto *concreteConv = dynamic_cast<const Conv *>(instance.get());
  check(concreteConv != nullptr, "Conv factory must return a concrete class");
  check(concreteConv->attrs().dataAxes == AxisOrder::NCHW &&
            concreteConv->attrs().weightAxes == AxisOrder::OIHW,
        "source factory must set typed Conv data/weight axes");
  check(instance->id() == 10 && instance->inputs() == std::vector<TensorId>({41, 7}) &&
            instance->outputs() == std::vector<TensorId>({99}),
        "role order is not map or UID order");
  check(instance->requirements().computeType == DataType::Float32 &&
            !instance->requirements().accumulationType,
        "compute type must be retained");
  shapeIs(*instance, {fp({1, 3, 7, 8}), fp({4, 3, 3, 3})}, {1, 4, 4, 4});
  shapeIs(*instance, {fp({2, 5, 11, 13}), fp({7, 5, 3, 3})}, {2, 7, 6, 7});
  expectError([&] { instance->infer({fp({1, 4, 7, 7}), fp({6, 2, 3, 3})}); }, "invalid_shape");
  auto changed = conv;
  set(member(changed.fields, "parameters"), "groups", integer(2));
  shapeIs(*registry.create(changed), {fp({1, 4, 7, 7}), fp({6, 2, 3, 3})}, {1, 6, 4, 4});
  changed = conv;
  set(member(changed.fields, "parameters"), "unknown", integer(0));
  expectError([&] { registry.create(changed); }, "unsupported_attribute");
  changed = conv;
  set(member(changed.fields, "parameters"), "unknown", V{});
  check(registry.create(changed) != nullptr, "unknown null extension does not add semantics");
  changed = conv;
  set(member(changed.fields, "parameters"), "unknown", array({}));
  check(registry.create(changed) != nullptr, "unknown empty array is inactive");
  changed = conv;
  set(member(changed.fields, "parameters"), "conv_mode", text("CONVOLUTION"));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = conv;
  set(member(changed.fields, "parameters"), "dilation", array({integer(2), integer(1)}));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = conv;
  set(member(changed.fields, "inputs"), "bias_tensor_uid", integer(8));
  expectError([&] { registry.create(changed); }, "unsupported_attribute");
  changed = conv;
  set(member(changed.fields, "inputs"), "x_tensor_uid", V{});
  expectError([&] { registry.create(changed); }, "invalid_ports");
  changed = conv;
  set(member(changed.fields, "inputs"), "x_tensor_uid", V(41.0));
  expectError([&] { registry.create(changed); }, "schema_invalid");
  changed = conv;
  changed.index = std::numeric_limits<std::size_t>::max();
  expectError([&] { registry.create(changed); }, "invalid_source");
  changed = conv;
  set(changed.fields, "compute_data_type", text("half"));
  expectError([&] { registry.create(changed); }, "unsupported_dtype");
  changed = conv;
  set(changed.fields, "type", text("OtherAttributes"));
  expectError([&] { registry.create(changed); }, "invalid_source");

  factoryIs<Add>(registry, pointwiseSource("add", true));
  factoryIs<ReLU>(registry, pointwiseSource("relu_fwd", false));
  factoryIs<Sigmoid>(registry, pointwiseSource("sigmoid_fwd", false));
  factoryIs<Mul>(registry, pointwiseSource("mul", true));
  for (const auto &entry :
       {std::make_pair("add", "Add"), std::make_pair("relu_fwd", "ReLU"),
        std::make_pair("sigmoid_fwd", "Sigmoid"), std::make_pair("mul", "Mul")}) {
    const bool binary = std::string(entry.first) == "add" || std::string(entry.first) == "mul";
    auto source = pointwiseSource(entry.first, binary);
    auto op = registry.create(source);
    check(op->semantic().name == entry.second && op->inputs().size() == (binary ? 2U : 1U) &&
              op->outputs().size() == 1,
          "Pointwise operation or inactive port mapping failed");
    if (binary) {
      check(op->inputs() == std::vector<TensorId>({41, 7}), "Pointwise role order changed");
      shapeIs(*op, {fp({1, 4, 3, 5}), fp({1, 4, 1, 1})}, {1, 4, 3, 5});
    } else {
      shapeIs(*op, {fp({1, 4, 3, 5})}, {1, 4, 3, 5});
      auto extra = source;
      set(member(extra.fields, "inputs"), "in_1_tensor_uid", integer(2));
      expectError([&] { registry.create(extra); }, "unsupported_attribute");
    }
    for (const auto *knob : {"elu_alpha", "relu_lower_clip", "relu_lower_clip_slope",
                             "relu_upper_clip", "softplus_beta", "swish_beta"}) {
      auto extra = source;
      set(member(extra.fields, "inputs"), knob, V(0.0));
      expectError([&] { registry.create(extra); }, "unsupported_attribute");
    }
    auto extra = source;
    set(member(extra.fields, "outputs"), "out_1_tensor_uid", integer(100));
    expectError([&] { registry.create(extra); }, "unsupported_attribute");
    extra = source;
    set(member(extra.fields, "inputs"), "in_2_tensor_uid", integer(2));
    expectError([&] { registry.create(extra); }, "unsupported_attribute");
  }
  auto repeated = pointwiseSource("add", true);
  set(member(repeated.fields, "inputs"), "in_1_tensor_uid", integer(41));
  check(registry.create(repeated)->inputs() == std::vector<TensorId>({41, 41}),
        "Add(U,U) uses lost");
  auto mismatch = pointwiseSource("add", true);
  mismatch.operation = "relu_fwd";
  expectError([&] { registry.create(mismatch); }, "invalid_source");
  mismatch = pointwiseSource("swish_fwd", false);
  expectError([&] { registry.create(mismatch); }, "operator_unknown");

  auto sliced = parameterized(
      "SliceAttributes", object({{"x_tensor_uid", integer(10)}}),
      object({{"y_tensor_uid", integer(12)}}),
      object({{"start_indices", array({integer(0), integer(64), integer(0), integer(0)})},
              {"limit_indices", array({integer(1), integer(128), integer(40), integer(40)})},
              {"slice_strides", array({integer(1), integer(1), integer(1), integer(1)})}}));
  factoryIs<Slice>(registry, sliced);
  shapeIs(*registry.create(sliced), {fp({1, 128, 40, 40})}, {1, 64, 40, 40});
  changed = sliced;
  set(member(changed.fields, "parameters"), "slice_strides",
      array({integer(1), integer(0), integer(1), integer(1)}));
  expectError([&] { registry.create(changed); }, "invalid_attribute");
  auto concatenated = parameterized(
      "ConcatenateAttributes",
      object({{"x_tensor_uids", array({integer(7), integer(2), integer(9), integer(3)})}}),
      object({{"y_tensor_uid", integer(11)}}),
      object({{"axis", integer(1)}, {"in_place_index", V{}}}));
  factoryIs<Concat>(registry, concatenated);
  check(registry.create(concatenated)->inputs() == std::vector<TensorId>({7, 2, 9, 3}),
        "Concat order lost");
  shapeIs(*registry.create(concatenated),
          {fp({1, 2, 3, 4}), fp({1, 3, 3, 4}), fp({1, 4, 3, 4}), fp({1, 5, 3, 4})}, {1, 14, 3, 4});
  for (const auto count : {2, 3}) {
    changed = concatenated;
    std::get<V::Array>(member(member(changed.fields, "inputs"), "x_tensor_uids").value)
        .resize(count);
    check(registry.create(changed)->inputs().size() == static_cast<std::size_t>(count),
          "Concat arity rejected");
  }
  changed = concatenated;
  set(member(changed.fields, "parameters"), "in_place_index", integer(0));
  expectError([&] { registry.create(changed); }, "unsupported_attribute");
  changed = concatenated;
  set(member(changed.fields, "parameters"), "axis", integer(0));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = concatenated;
  set(member(changed.fields, "inputs"), "x_tensor_uids", array({integer(1), V{}}));
  expectError([&] { registry.create(changed); }, "invalid_ports");

  const auto resized = resizeSource();
  const auto resizeInstance = registry.create(resized);
  const auto *concreteResize = dynamic_cast<const Resize *>(resizeInstance.get());
  check(concreteResize != nullptr, "Resize factory must return a concrete class");
  check(concreteResize->attrs().dataAxes == AxisOrder::NCHW &&
            concreteResize->attrs().mode == ResizeMode::Nearest &&
            concreteResize->attrs().coordinateTransform == CoordinateTransform::Asymmetric &&
            concreteResize->attrs().nearestMode == NearestMode::Floor &&
            concreteResize->attrs().aspectPolicy == AspectPolicy::Stretch,
        "source Resize policies/axes must be typed enums");
  shapeIs(*resizeInstance, {fp({1, 16, 20, 30})}, {1, 16, 40, 60});
  for (const auto *roi : {"roi_start", "roi_end", "axes", "sizes"}) {
    changed = resized;
    set(member(changed.fields, "parameters"), roi, array({integer(1)}));
    expectError([&] { registry.create(changed); }, "unsupported");
  }
  changed = resized;
  set(member(changed.fields, "parameters"), "scales", array({V(1.0), V(1.0), V(2.0), V(2.5)}));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = resized;
  set(member(changed.fields, "parameters"), "antialias", V(true));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = resized;
  set(member(changed.fields, "parameters"), "nearest_mode", text("round_prefer_floor"));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = resized;
  set(member(changed.fields, "parameters"), "roi_start", V{});
  expectError([&] { registry.create(changed); }, "schema_invalid");
  changed = resized;
  set(member(changed.fields, "parameters"), "unknown", V(false));
  expectError([&] { registry.create(changed); }, "unsupported_attribute");

  const auto pooled = poolSource();
  auto pool = registry.create(pooled);
  const auto *concretePool = dynamic_cast<const MaxPool *>(pool.get());
  check(concretePool && pool->inputs() == std::vector<TensorId>({41}) &&
            pool->outputs() == std::vector<TensorId>({99}),
        "top-level MaxPool roles not mapped");
  check(concretePool->attrs().dataAxes == AxisOrder::NCHW &&
            concretePool->attrs().padding == PoolPadding::NegativeInfinity,
        "source Pool padding/axes must be typed enums");
  shapeIs(*pool, {fp({1, 16, 40, 40})}, {1, 16, 40, 40});
  changed = pooled;
  set(changed.fields, "generate_index", V(true));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = pooled;
  set(changed.fields, "padding_mode", text("zero_pad"));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = pooled;
  set(changed.fields, "resample_mode", text("avgpool"));
  expectError([&] { registry.create(changed); }, "unsupported");
  changed = pooled;
  set(changed.fields, "dilation", array({integer(1), integer(1)}));
  expectError([&] { registry.create(changed); }, "unsupported_attribute");
  changed = pooled;
  set(changed.fields, "x_tensor_uid", integer(0));
  expectError([&] { registry.create(changed); }, "invalid_ports");

  SourceNode absent;
  absent.format = "benchmark-json/1";
  absent.fields = object({});
  for (const auto *type :
       {"BatchNormAttributes", "SiLUAttributes", "SplitAttributes", "MatMulAttributes"}) {
    absent.type = type;
    expectError([&] { registry.create(absent); }, "operator_unknown");
  }
  changed = conv;
  changed.format = "unrecognized/1";
  expectError([&] { registry.create(changed); }, "operator_unknown");
}
} // namespace sourceBindingCases

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: OpFusionFrontendTests <neutral-fixture.json>\n";
    return 1;
  }
  try {
    std::ifstream input(argv[1], std::ios::binary);
    check(input.is_open(), "cannot open neutral frontend fixture");
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const Json document = Json::parse(text);
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"PreserveGraphAndCompleteSource", [&] { preserveGraph(document); }},
        {"BuiltinSourceBindings", [] { sourceBindingCases::sourceTests(); }},
        {"StrictSourceNumbers", [&] { sourceNumbers(document); }},
        {"CommaDecimalLocaleWhenAvailable", [&] { localizedSourceNumbers(document); }},
        {"GroupedGraphTokenLocale", [&] { groupedGraphIdentity(text); }},
        {"EscapedNulIsSourceData",
         [&] {
           const std::string expected("a\0b", 3);
           auto escaped = document;
           escaped["nodes"][0]["extra_metadata"]["nul"] = expected;
           bool seen = false;
           const auto factories = registry([&](const SourceNode &source) {
             if (source.index == 0) {
               check(source.fields.at("extra_metadata").at("nul").string() == expected,
                     "escaped Unicode NUL was not preserved as string data");
               seen = true;
             }
           });
           (void)ModelImporter(factories).importText(escaped.dump());
           check(seen, "escaped-NUL source never reached its factory");
         }},
        {"MalformedDocuments", [&] { malformedDocuments(document); }},
        {"DelegatedFactoryAndDagFailures", [&] { delegatedFailures(document); }},
        {"FreshOpaqueGraphIdentity",
         [&] {
           const auto factories = registry();
           const ModelImporter importer(factories);
           const auto first = importer.importText(text, ModelFormat::JsonGraph, "same-source");
           const auto second = importer.importText(text, ModelFormat::JsonGraph, "same-source");
           check(first.graph->id() != second.graph->id(), "graph identity was reused");
           for (const auto &id : {first.graph->id(), second.graph->id()})
             check(id.size() == 32 &&
                       std::all_of(id.begin(), id.end(),
                                   [](unsigned char c) { return std::isxdigit(c) != 0; }),
                   "graph ID is not an opaque 128-bit token");
         }},
        {"GenericOperationFallbackAndPrecedence",
         [&] {
           const auto factories = registry([](const SourceNode &source) {
             check(source.operation == "PROJECT", "generic operation selector changed");
           });
           auto fallback = document;
           for (auto &node : fallback["nodes"]) {
             node["inputs"].erase("operation");
             node["resample_mode"] = "PROJECT";
           }
           const ModelImporter importer(factories);
           (void)importer.importText(fallback.dump());
           auto precedence = document;
           for (auto &node : precedence["nodes"])
             node["resample_mode"] = "IGNORED";
           (void)importer.importText(precedence.dump());
         }},
        {"FlattenedSourcesBelongToFactory",
         [&] {
           auto flattened = document;
           for (auto &node : flattened["nodes"]) {
             node["source_refs"] = node["inputs"]["tensors"];
             node["result_refs"] = node["outputs"]["tensors"];
             node["resample_mode"] = "PROJECT";
             node.erase("inputs");
             node.erase("outputs");
           }
           Registry flattenedRegistry;
           flattenedRegistry.add({{"frontend-test", "projection"}});
           flattenedRegistry.bind({{"frontend-test", "projection"},
                                   {{"benchmark-json/1", "FixtureRoute", "PROJECT"}},
                                   [](const SourceNode &source) {
                                     check(source.fields.find("inputs") == nullptr,
                                           "frontend invented grouped source ports");
                                     return std::make_unique<Projection>(
                                         static_cast<OperatorId>(source.index) + 1,
                                         references(source.fields.at("source_refs")),
                                         references(source.fields.at("result_refs")), false);
                                   }});
           flattenedRegistry.freeze("flattened-factory/1");
           const auto result = ModelImporter(flattenedRegistry).importText(flattened.dump());
           check(result.graph->edges().size() == 2,
                 "flattened source failed to reach its factory intact");
         }},
        {"OptionalProvenanceHints",
         [&] {
           auto minimalHints = document;
           for (auto &tensor : minimalHints["tensors"])
             for (const auto &key : {"strides", "alignment", "virtual", "is_runtime_pass_by_value"})
               tensor.erase(key);
           const auto factories = registry();
           const auto result = ModelImporter(factories).importText(minimalHints.dump());
           for (const auto &item : result.tensorOrigins) {
             const auto &origin = item.second;
             check(origin.strides.empty() && !origin.alignment.has_value() &&
                       !origin.virtualTensor.has_value() && !origin.runtimePassByValue.has_value(),
                   "missing hints acquired invented defaults");
           }
         }},
        {"UnsignedProvenanceRange",
         [&] {
           auto maximum = document;
           maximum["tensors"][0]["uid"] = std::numeric_limits<std::uint64_t>::max();
           maximum["tensors"][0]["alignment"] = std::numeric_limits<std::uint64_t>::max();
           const auto factories = registry();
           const auto result = ModelImporter(factories).importText(maximum.dump());
           check(result.tensorOrigins.at(std::numeric_limits<std::uint64_t>::max()).alignment ==
                     std::numeric_limits<std::uint64_t>::max(),
                 "unsigned hint/UID was narrowed");
         }},
        {"FactoryIdentityContract",
         [&] {
           Registry incorrect;
           incorrect.add({{"frontend-test", "projection"}});
           incorrect.bind({{"frontend-test", "projection"},
                           {{"benchmark-json/1", "FixtureRoute", "PROJECT"}},
                           [](const SourceNode &source) {
                             return std::make_unique<Projection>(
                                 static_cast<OperatorId>(source.index) + 10,
                                 references(source.fields.at("inputs").at("tensors")),
                                 references(source.fields.at("outputs").at("tensors")), false);
                           }});
           incorrect.freeze("incorrect-factory/1");
           expectError([&] { (void)ModelImporter(incorrect).importText(text); },
                       "benchmark_factory_id", {"source node[0]", "first"});
         }},
        {"FileEntryPoint",
         [&] {
           const auto factories = registry();
           const auto result = ModelImporter(factories).importFile(argv[1]);
           check(result.graph->tensors().size() == 5 && result.graph->nodes().size() == 2,
                 "file entry point lost the graph");
         }},
        {"FileErrorsAreDomainErrors",
         [&] {
           const auto factories = registry();
           const ModelImporter importer(factories);
           const std::string fixturePath = argv[1];
           expectError([&] { (void)importer.importFile(fixturePath + ".missing-importer-test"); },
                       "model_file_open");
           const auto separator = fixturePath.find_last_of("/\\");
           const auto directory =
               separator == std::string::npos ? "." : fixturePath.substr(0, separator);
           expectError([&] { (void)importer.importFile(directory); }, {}, {directory});
         }},
        {"RegistryMustBeFrozen",
         [&] {
           Registry mutableRegistry;
           expectError([&] { (void)ModelImporter(mutableRegistry); }, "model_registry_unfrozen");
         }},
        {"UnsupportedModelFormat",
         [&] {
           const ModelImporter importer{registry()};
           const auto unknown = static_cast<ModelFormat>(-1);
           expectError([&] { (void)importer.importText(text, unknown, "unknown-format"); },
                       "model_format_unsupported", {"unknown-format"});
           expectError([&] { (void)importer.importFile("missing-format-test.json", unknown); },
                       "model_format_unsupported", {"missing-format-test.json"});
         }},
    };
    std::size_t failures = 0;
    for (const auto &test : tests) {
      try {
        test.second();
        std::cout << "PASS " << test.first << '\n';
      } catch (const std::exception &error) {
        ++failures;
        std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
      }
    }
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL test setup: " << error.what() << '\n';
    return 1;
  }
}
