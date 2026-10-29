/// @file
/// CLI 参数、公开模型入口与 JSON 展示的进程边界；不实现算子或设备执行。
#include "Context.h"
#include "LinkedPlugins.h"
#include "opfusion/common/Error.h"
#include "opfusion/frontend/ModelImporter.h"
#include "opfusion/graph/Axis.h"
#include "opfusion/graph/AxisModel.h"
#include "opfusion/graph/DAG.h"
#include "opfusion/graph/Operator.h"
#include "opfusion/graph/Tensor.h"
#include "opfusion/registry/Registry.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>

namespace {

using nlohmann::json;

/// 命令行文本的局部配置；compact 仅控制 JSON 排版，不改变输出 schema。
struct Arguments {
  std::string command;
  std::string file;
  std::string pluginRoot;
  std::string context;
  bool compact = false;
  bool help = false;
};

/// 保留未知事实为 JSON null，不能把缺失的 trait 或来源提示当作 false。
json optionalBool(const std::optional<bool> &value) {
  return value ? json(*value) : json(nullptr);
}

// 展示可验证的通用索引语言；不根据算子名称拼接关系。
json expression(const opfusion::IndexExpression &e) {
  json terms = json::array(), lookups = json::array();
  for (const auto &t : e.terms) {
    terms.push_back({{"axis", t.axis},
                     {"coefficient", t.coefficient},
                     {"divisor", t.divisor},
                     {"modulus", t.modulus ? json(*t.modulus) : json(nullptr)}});
  }
  for (const auto &lookup : e.lookups) {
    json coordinates = json::array();
    for (const auto &c : lookup.coordinates) {
      coordinates.push_back(expression(c));
    }
    lookups.push_back({{"input_port", lookup.port},
                       {"coordinates", coordinates},
                       {"valid_min", -lookup.extent},
                       {"valid_max_exclusive", lookup.extent},
                       {"negative_index_add", lookup.extent}});
  }
  return {{"offset", e.offset}, {"terms", terms}, {"lookups", lookups}};
}
json coordinates(const std::vector<opfusion::IndexExpression> &expressions) {
  json result = json::array();
  for (const auto &e : expressions) {
    result.push_back(expression(e));
  }
  return result;
}
// 命名语义只是注释；真正的轴定义由坐标范围及可验证的生产者关系给出。
json semanticAxis(const std::optional<opfusion::Axis> &axis) {
  if (!axis) {
    return nullptr;
  }
  return {{"kind", axis->kind == opfusion::AxisKind::Unknown
                       ? json(nullptr)
                       : json(opfusion::toString(axis->kind))},
          {"label", axis->label}};
}
json signature(const opfusion::Tensor &tensor, const opfusion::TensorAxisModel &model) {
  const auto bytes = tensor.logicalBytes();
  json axes = json::array();
  for (const auto &axis : model.axes) {
    json definitions = json::array();
    for (const auto &definition : axis.derivations) {
      json sources = json::array(), metadata = json::array();
      for (const auto &source : definition.sources) {
        sources.push_back({{"tensor", source.tensor},
                           {"input_port", source.inputPort},
                           {"input_access", source.access},
                           {"dimension", source.dimension},
                           {"coordinate", expression(source.read)}});
      }
      for (const auto &read : definition.shapeReads) {
        metadata.push_back({{"input_port", read.port},
                            {"tensor", model.inputs.at(read.port)},
                            {"dimension", expression(read.dimension)}});
      }
      definitions.push_back({{"piece", definition.piece},
                             {"structure", opfusion::toString(definition.structure)},
                             {"coordinate", expression(definition.write)},
                             {"sources", sources},
                             {"shape_reads", metadata}});
    }
    axes.push_back({{"position", axis.position},
                    {"lower", 0},
                    {"upper", axis.extent},
                    {"semantic", semanticAxis(axis.semantic)},
                    {"definitions", definitions}});
  }
  const auto status = model.status == opfusion::SemanticStatus::Known     ? "known"
                      : model.status == opfusion::SemanticStatus::Unknown ? "unknown"
                                                                          : "unsupported";
  return {
      {"uid", tensor.id()},
      {"name", tensor.name()},
      {"shape", tensor.shape()},
      {"dtype", opfusion::toString(tensor.dtype())},
      {"value_kind", opfusion::toString(tensor.kind())},
      {"logical_axes", axes},
      {"axis_model",
       {{"status", status},
        {"reason", model.reason.empty() ? json(nullptr) : json(model.reason)},
        {"origin", opfusion::toString(model.origin)},
        {"producer", model.producer ? json(*model.producer) : json(nullptr)},
        {"output_port", model.outputPort ? json(*model.outputPort) : json(nullptr)},
        {"input_tensors", model.inputs},
        {"relation_pieces", model.pieces},
        {"semantic_annotation_origin",
         model.semanticOrigin ? json(opfusion::toString(*model.semanticOrigin)) : json(nullptr)}}},
      {"logical_bytes", bytes ? json(*bytes) : json(nullptr)}};
}

json accessRelation(const opfusion::IndexSemantics &semantics) {
  const auto status = semantics.status == opfusion::SemanticStatus::Known     ? "known"
                      : semantics.status == opfusion::SemanticStatus::Unknown ? "unknown"
                                                                              : "unsupported";
  json pieces = json::array();
  for (const auto &piece : semantics.pieces) {
    json domain = json::array(), reads = json::array(), writes = json::array(),
         metadata = json::array();
    for (const auto &a : piece.domain) {
      domain.push_back(
          {{"name", a.name}, {"lower", a.lower}, {"upper", a.upper}, {"reduction", a.reduction}});
    }
    for (const auto &a : piece.inputs) {
      std::string fill;
      switch (a.outOfBounds) {
      case opfusion::FillValue::None:
        fill = "none";
        break;
      case opfusion::FillValue::Zero:
        fill = "zero";
        break;
      case opfusion::FillValue::NegativeInfinity:
        fill = "negative_infinity";
        break;
      }
      reads.push_back(
          {{"port", a.port},
           {"coordinates", coordinates(a.coordinates)},
           {"out_of_bounds", fill},
           {"phase",
            a.phase == opfusion::AccessPhase::AfterReduction ? "after_reduction" : "iteration"}});
    }
    for (const auto &a : piece.outputs) {
      writes.push_back({{"port", a.port}, {"coordinates", coordinates(a.coordinates)}});
    }
    for (const auto &a : piece.shapeReads) {
      metadata.push_back({{"port", a.port}, {"dimension", expression(a.dimension)}});
    }
    pieces.push_back({{"domain", domain},
                      {"inputs", reads},
                      {"outputs", writes},
                      {"shape_reads", metadata},
                      {"reduction", opfusion::toString(piece.reduction)},
                      {"reduction_scale", piece.reductionScale},
                      {"after_reduction_scale", piece.afterReductionScale}});
  }
  return {{"status", status},
          {"reason", semantics.reason.empty() ? json(nullptr) : json(semantics.reason)},
          {"pieces", pieces}};
}

/// 仅通过 Operator 公共行为展示有序端口、语义和数值约束，不按算子名分派。
json describeNode(const opfusion::Operator &node, const opfusion::DAG &graph) {
  const auto traits = node.traits();
  json numeric = {{"allow_approximation", node.requirements().allowApproximation},
                  {"allowed_types", json::array()}};
  for (auto type : node.requirements().allowedTypes) {
    numeric["allowed_types"].push_back(opfusion::toString(type));
  }
  numeric["accumulation_type"] =
      node.requirements().accumulationType
          ? json(opfusion::toString(*node.requirements().accumulationType))
          : json(nullptr);
  const auto &req = node.requirements();
  numeric["compute_type"] =
      req.computeType ? json(opfusion::toString(*req.computeType)) : json(nullptr);
  numeric["absolute_error"] = req.absoluteError ? json(*req.absoluteError) : json(nullptr);
  numeric["relative_error"] = req.relativeError ? json(*req.relativeError) : json(nullptr);
  std::vector<opfusion::TensorSignature> inputs;
  for (auto id : node.inputs()) {
    inputs.push_back(graph.tensor(id).signature());
  }
  json attributes = json::object();
  for (const auto &attribute : node.attributes(inputs)) {
    std::visit([&](const auto &value) { attributes[attribute.name] = value; }, attribute.value);
  }
  const auto semantics = node.indexSemantics(inputs);
  const auto broadcast = node.broadcasting(inputs);
  json ports = json::array();
  for (const auto &port : broadcast.inputs) {
    ports.push_back({{"port", port.port}, {"axes", port.axes}});
  }
  return {{"id", node.id()},
          {"type", node.semantic().name},
          {"namespace", node.semantic().nameSpace},
          {"attributes", attributes},
          {"index_semantics", accessRelation(semantics)},
          {"broadcasting", {{"applied", broadcast.applied}, {"inputs", ports}}},
          {"inputs", node.inputs()},
          {"outputs", node.outputs()},
          {"numerical_requirements", numeric},
          {"traits",
           {{"pure", optionalBool(traits.pure)},
            {"elementwise", optionalBool(traits.elementwise)},
            {"reduction", optionalBool(traits.reduction)},
            {"supports_broadcasting", optionalBool(traits.broadcasting)}}}};
}

/// 将成功导入的不可变图与来源提示投影为 inspection/4，不重新推导或执行模型。
/// 输入/终端输出只是图依赖推断；来源物理提示和形状校验不升级为执行证据。
json inspect(const opfusion::frontend::ImportResult &result) {
  const auto &graph = *result.graph;
  const auto axisModels = opfusion::buildAxisModels(graph);
  json nodes = json::array();
  json tensors = json::array();
  json edges = json::array();
  json diagnostics = json::array();
  json sourceNodes = json::array();
  std::map<std::string, std::size_t> counts;
  std::map<std::string, std::size_t> indexCounts{{"known", 0}, {"unknown", 0}, {"unsupported", 0}};
  std::size_t missingCompute = 0, missingAccumulation = 0;
  std::map<std::string, std::size_t> coordinateModels{
      {"known", 0}, {"unknown", 0}, {"unsupported", 0}};
  std::size_t scalars = 0;
  std::map<std::string, std::size_t> axisMetadata{
      {"specified", 0}, {"partial", 0}, {"unspecified", 0}};
  for (const auto &node : graph.nodes()) {
    ++counts[node->semantic().name];
    auto description = describeNode(*node, graph);
    ++indexCounts[description["index_semantics"]["status"].get<std::string>()];
    missingCompute += node->requirements().computeType ? 0 : 1;
    missingAccumulation += node->requirements().accumulationType ? 0 : 1;
    nodes.push_back(std::move(description));
  }
  for (const auto &tensor : graph.tensors()) {
    const auto &axes = tensor.signature().axes;
    if (!axes) {
      ++axisMetadata["unspecified"];
    } else {
      const auto partial = std::any_of(axes->begin(), axes->end(), [](const auto &axis) {
        return axis.kind == opfusion::AxisKind::Unknown;
      });
      ++axisMetadata[partial ? "partial" : "specified"];
    }
    const auto &axisModel = axisModels.at(tensor.id());
    ++coordinateModels[axisModel.status == opfusion::SemanticStatus::Known     ? "known"
                       : axisModel.status == opfusion::SemanticStatus::Unknown ? "unknown"
                                                                               : "unsupported"];
    scalars += axisModel.origin == opfusion::AxisModelOrigin::Scalar ? 1 : 0;
    auto entry = signature(tensor, axisModel);
    if (auto found = result.tensorOrigins.find(tensor.id()); found != result.tensorOrigins.end()) {
      const auto &source = found->second;
      entry["source_layout"] = {
          {"strides", source.strides},
          {"stride_units", "unspecified"},
          {"alignment", source.alignment ? json(*source.alignment) : json(nullptr)},
          {"virtual", optionalBool(source.virtualTensor)},
          {"runtime_pass_by_value", optionalBool(source.runtimePassByValue)}};
    }
    tensors.push_back(std::move(entry));
  }
  for (const auto &edge : graph.edges()) {
    edges.push_back({{"id", edge.id},
                     {"producer", edge.producer},
                     {"output_port", edge.outputPort},
                     {"consumer", edge.consumer},
                     {"input_port", edge.inputPort},
                     {"tensor", edge.tensor}});
  }
  for (const auto &diagnostic : result.diagnostics) {
    diagnostics.push_back({{"code", diagnostic.code}, {"message", diagnostic.message}});
  }
  for (const auto &origin : result.nodeOrigins) {
    sourceNodes.push_back({{"id", origin.id},
                           {"index", origin.index},
                           {"name", origin.name},
                           {"input_names", origin.inputs},
                           {"output_names", origin.outputs},
                           {"normalizations", origin.normalizations}});
  }
  json parameters = json::array();
  for (const auto &p : result.parameters) {
    parameters.push_back({{"tensor", p.id},
                          {"name", p.name},
                          {"sha256", p.contentDigest},
                          {"external_location", p.externalLocation},
                          {"offset", p.offset},
                          {"length", p.length}});
  }
  return {{"schema", "opfusion-inspection/4"},
          {"source",
           {{"sha256", result.sourceDigest.empty() ? json(nullptr) : json(result.sourceDigest)},
            {"content_sha256",
             result.modelContentDigest.empty() ? json(nullptr) : json(result.modelContentDigest)},
            {"ir_version", result.irVersion ? json(*result.irVersion) : json(nullptr)},
            {"opsets", result.opsets},
            {"producer", result.producer},
            {"producer_version", result.producerVersion}}},
          {"parameters", parameters},
          {"model_inputs", graph.modelInputs() ? json(*graph.modelInputs()) : json(nullptr)},
          {"model_outputs", graph.modelOutputs() ? json(*graph.modelOutputs()) : json(nullptr)},
          {"profile", result.profile},
          {"modeling",
           {{"index_relations", indexCounts},
            {"axis_models", coordinateModels},
            {"scalar_tensors", scalars},
            {"named_axis_annotations", axisMetadata},
            {"logical_axes_note", "coordinates and their producer mappings define axes; optional "
                                  "semantic names are annotations, not missing models"},
            {"missing_compute_type", missingCompute},
            {"missing_accumulation_type", missingAccumulation},
            {"numerical_unknown_reason", "source requirements are not inferred from storage dtype"},
            {"physical_model", "requires deployment context and implementation evidence"}}},
          {"graph_id", graph.id()},
          {"name", graph.name()},
          {"registry_id", graph.registryId()},
          {"shape_validation", "passed"},
          {"node_count", graph.nodes().size()},
          {"tensor_count", graph.tensors().size()},
          {"edge_count", graph.edges().size()},
          {"operator_counts", counts},
          {"nodes", nodes},
          {"tensors", tensors},
          {"edges", edges},
          {"source_nodes", sourceNodes},
          {"inferred_external_inputs", graph.externalInputs()},
          {"inferred_terminal_outputs", graph.terminalOutputs()},
          {"model_outputs_complete", graph.modelOutputs().has_value()},
          {"diagnostics", diagnostics},
          {"execution",
           {{"status", "not_established"},
            {"candidate_search", "not_searched"},
            {"platform", "not_provided"},
            {"parameter_contents", result.irVersion ? "verified" : "not_provided"},
            {"boundary", "not_provided"},
            {"cost", nullptr},
            {"reason", "semantic import is not device execution validation"}}}};
}

/// 关联独立的语义与来源绑定视图；无绑定的已登记语义仍必须出现在输出中。
json describeOperators(const opfusion::Registry &registry) {
  json operators = json::array();
  for (const auto &entry : registry.registrations()) {
    json bindings = json::array();
    for (const auto &binding : registry.sourceBindings()) {
      if (binding.semantic.key() != entry.semantic.key()) {
        continue;
      }
      for (const auto &selector : binding.selectors) {
        bindings.push_back(
            {{"format", selector.format},
             {"type", selector.type},
             {"operation", selector.operation ? json(*selector.operation) : json(nullptr)}});
      }
    }
    operators.push_back({{"name", entry.semantic.name},
                         {"namespace", entry.semantic.nameSpace},
                         {"bindings", bindings},
                         {"support", "semantic_model_only"}});
  }
  return {
      {"schema", "opfusion-operators/2"}, {"registry_id", registry.id()}, {"operators", operators}};
}

/// 同步解析借用的 argv 并复制文本；参数缺值或多余位置参数抛 argument_invalid。
/// 命令合法性与 inspect 文件是否给出由进程入口继续校验，帮助选项直接返回。
Arguments parseArguments(int argc, char **argv) {
  Arguments result;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      result.help = true;
      return result;
    }
    if (argument == "--json") {
      result.compact = true;
    } else if (argument == "--context") {
      opfusion::require(++i < argc, "argument_invalid", "--context needs a file");
      result.context = argv[i];
    } else if (argument == "--plugins-root") {
      opfusion::require(++i < argc, "argument_invalid", "--plugins-root needs a directory");
      result.pluginRoot = argv[i];
    } else if (result.command.empty()) {
      result.command = argument;
    } else if (result.command == "inspect" && result.file.empty()) {
      result.file = argument;
    } else {
      throw opfusion::Error("argument_invalid", "unexpected argument: " + argument);
    }
  }
  return result;
}

/// 只展示当前建模命令及能力限制，不暗示尚未实现的设备或优化入口。
void help() {
  std::cout << "OpFusion C++ modeling CLI\n"
               "  opfusion operators [--json] [--plugins-root DIR]\n"
               "  opfusion inspect FILE.onnx|FILE.json [--context scenario.json] [--json] "
               "[--plugins-root DIR]\n"
               "Static ONNX opset 17 and compatible benchmark JSON; no hardware execution.\n";
}

} // namespace

/// 显式装配后调用公开导入器；成功写 stdout，失败写结构化 stderr 并返回 2。
/// Error 保留领域错误码，其余标准异常归为 input_error；不返回部分成功结果。
int main(int argc, char **argv) {
  try {
    const auto arguments = parseArguments(argc, argv);
    if (arguments.help || arguments.command.empty()) {
      help();
      return 0;
    }
    opfusion::require(arguments.command == "operators" || arguments.command == "inspect",
                      "argument_invalid", "unknown command");
    auto registry = opfusion::app::loadRegistry(
        arguments.pluginRoot.empty() ? nullptr : arguments.pluginRoot.c_str());
    json output;
    if (arguments.command == "operators") {
      output = describeOperators(registry);
    } else {
      opfusion::require(!arguments.file.empty(), "argument_invalid",
                        "inspect requires a model file");
      const opfusion::frontend::ModelImporter importer(std::move(registry));
      const auto result = importer.importFile(arguments.file);
      output = inspect(result);
      if (!arguments.context.empty()) {
        output["context"] = opfusion::app::inspectContext(arguments.context, result);
        output["execution"]["platform"] = "declared";
        output["execution"]["boundary"] = "declared";
        output["modeling"]["physical_model"] = "deployment declared; device capabilities and "
                                               "implementation evidence are not established";
      }
    }
    std::cout << output.dump(arguments.compact ? -1 : 2) << '\n';
    return 0;
  } catch (const opfusion::Error &error) {
    std::cerr << json({{"error", {{"code", error.code()}, {"message", error.what()}}}}).dump()
              << '\n';
    return 2;
  } catch (const std::exception &error) {
    std::cerr << json({{"error", {{"code", "input_error"}, {"message", error.what()}}}}).dump()
              << '\n';
    return 2;
  }
}
