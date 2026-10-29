// 原生 ONNX 静态导入：只建立值图、来源和内容摘要，不调设备或预测成本。
#include "OnnxImporter.h"

#include "onnx.pb.h"
#include "opfusion/common/Error.h"
#include "picosha2.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
namespace opfusion::frontend::detail {
namespace {
using ProtoTensor = onnx::TensorProto;
using SV = SourceValue;
std::string digest(const std::string &bytes) {
  return picosha2::hash256_hex_string(bytes);
}
DataType dtype(int type) {
  switch (type) {
  case ProtoTensor::FLOAT:
    return DataType::Float32;
  case ProtoTensor::DOUBLE:
    return DataType::Float64;
  case ProtoTensor::FLOAT16:
    return DataType::Float16;
  case ProtoTensor::INT8:
    return DataType::Int8;
  case ProtoTensor::INT32:
    return DataType::Int32;
  case ProtoTensor::INT64:
    return DataType::Int64;
  case ProtoTensor::BOOL:
    return DataType::Bool;
  default:
    throw Error("onnx_dtype_unsupported",
                "unsupported ONNX tensor data type: " + std::to_string(type));
  }
}
TensorSignature signature(const onnx::ValueInfoProto &value) {
  require(value.has_type() && value.type().has_tensor_type(), "onnx_type_unsupported",
          "only tensor values are supported: " + value.name());
  const auto &t = value.type().tensor_type();
  require(t.has_shape(), "onnx_dynamic_shape", "missing static shape: " + value.name());
  Shape shape;
  for (const auto &d : t.shape().dim()) {
    require(d.has_dim_value() && !d.has_dim_param() && d.dim_value() >= 0, "onnx_dynamic_shape",
            "dynamic dimension: " + value.name());
    shape.push_back(d.dim_value());
  }
  return {shape, dtype(t.elem_type())};
}
std::uint64_t decimal(const std::string &value) {
  require(!value.empty() &&
              std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
          "onnx_external_invalid", "external offset/length must be unsigned decimal");
  std::uint64_t result = 0;
  for (char c : value) {
    const auto d = static_cast<std::uint64_t>(c - '0');
    require(result <= (std::numeric_limits<std::uint64_t>::max() - d) / 10, "onnx_external_invalid",
            "external range overflow");
    result = result * 10 + d;
  }
  return result;
}
// 规范的小端内容用于摘要；不依赖宿主的字节序或 float 指针别名。
void appendBits(std::string &data, std::uint64_t value, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    data.push_back(static_cast<char>((value >> (8 * i)) & 255));
}
std::uint64_t readBits(const std::string &data, std::size_t offset, std::size_t count) {
  std::uint64_t v = 0;
  for (std::size_t i = 0; i < count; ++i)
    v |= static_cast<std::uint64_t>(static_cast<unsigned char>(data.at(offset + i))) << (8 * i);
  return v;
}
struct Payload {
  TensorSignature signature;
  std::string digest;
  std::optional<Shape> integers;
  std::optional<std::vector<double>> floating;
  std::string location;
  std::uint64_t offset = 0, length = 0;
};
Payload payload(const ProtoTensor &t, const std::string &source) {
  Payload p;
  p.signature = {Shape(t.dims().begin(), t.dims().end()), dtype(t.data_type())};
  const auto expected = logicalBytes(p.signature.shape, p.signature.dtype).value();
  require(!t.has_segment(), "onnx_tensor_unsupported", "segmented tensors are unsupported");
  const bool typed = t.float_data_size() || t.double_data_size() || t.int32_data_size() ||
                     t.int64_data_size() || t.uint64_data_size() || t.string_data_size();
  std::string data;
  if (t.data_location() == ProtoTensor::EXTERNAL) {
    require(t.raw_data().empty() && !typed, "onnx_external_invalid",
            "external tensor also has embedded content");
    std::map<std::string, std::string> fields;
    for (const auto &f : t.external_data())
      require(fields.emplace(f.key(), f.value()).second, "onnx_external_invalid",
              "duplicate external data key");
    require(fields.count("location") != 0 && !fields.at("location").empty(),
            "onnx_external_invalid", "external location missing");
    for (const auto &f : fields)
      require(f.first == "location" || f.first == "offset" || f.first == "length",
              "onnx_external_invalid", "unsupported external data field: " + f.first);
    const std::filesystem::path relative(fields.at("location"));
    require(!relative.is_absolute() && !relative.has_root_name(), "onnx_external_invalid",
            "absolute external data reference");
    for (const auto &part : relative)
      require(part != ".." && part != ".", "onnx_external_invalid",
              "external data reference escapes model directory");
    require(source != "<memory>", "onnx_external_invalid",
            "external data requires a model file location");
    const auto root =
        std::filesystem::weakly_canonical(std::filesystem::path(source).parent_path());
    const auto file = std::filesystem::weakly_canonical(root / relative);
    const auto rel = file.lexically_relative(root);
    require(!rel.empty() && *rel.begin() != ".." && std::filesystem::is_regular_file(file),
            "onnx_external_missing", "external file missing or outside model directory");
    p.offset = fields.count("offset") ? decimal(fields.at("offset")) : 0;
    p.length = fields.count("length") ? decimal(fields.at("length")) : expected;
    const auto size = std::filesystem::file_size(file);
    require(p.length == expected && p.offset <= size && p.length <= size - p.offset,
            "onnx_external_range", "external tensor range disagrees with shape or file");
    require(p.length <= static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) &&
                p.offset <= static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
            "onnx_external_range", "external range too large");
    std::ifstream input(file, std::ios::binary);
    require(bool(input), "onnx_external_missing", "cannot open external parameter file");
    input.seekg(static_cast<std::streamoff>(p.offset));
    data.resize(p.length);
    input.read(data.data(), static_cast<std::streamsize>(p.length));
    require(static_cast<std::uint64_t>(input.gcount()) == p.length, "onnx_external_range",
            "external content truncated");
    p.location = relative.generic_string();
  } else {
    require(t.external_data_size() == 0 && t.data_location() == ProtoTensor::DEFAULT,
            "onnx_external_invalid", "invalid embedded data location");
    if (!t.raw_data().empty()) {
      require(!typed, "onnx_tensor_invalid", "tensor mixes raw and typed content");
      data = t.raw_data();
    } else {
      const auto type = p.signature.dtype;
      if (type == DataType::Float32) {
        require(!t.double_data_size() && !t.int32_data_size() && !t.int64_data_size() &&
                    !t.uint64_data_size() && !t.string_data_size(),
                "onnx_tensor_invalid", "wrong typed tensor field");
        for (float f : t.float_data()) {
          std::uint32_t bits;
          std::memcpy(&bits, &f, 4);
          appendBits(data, bits, 4);
        }
      } else if (type == DataType::Float64) {
        require(!t.float_data_size() && !t.int32_data_size() && !t.int64_data_size() &&
                    !t.uint64_data_size() && !t.string_data_size(),
                "onnx_tensor_invalid", "wrong typed tensor field");
        for (double f : t.double_data()) {
          std::uint64_t bits;
          std::memcpy(&bits, &f, 8);
          appendBits(data, bits, 8);
        }
      } else if (type == DataType::Int64) {
        require(!t.float_data_size() && !t.double_data_size() && !t.int32_data_size() &&
                    !t.uint64_data_size() && !t.string_data_size(),
                "onnx_tensor_invalid", "wrong typed tensor field");
        for (auto v : t.int64_data())
          appendBits(data, static_cast<std::uint64_t>(v), 8);
      } else {
        require(!t.float_data_size() && !t.double_data_size() && !t.int64_data_size() &&
                    !t.uint64_data_size() && !t.string_data_size(),
                "onnx_tensor_invalid", "wrong typed tensor field");
        for (auto v : t.int32_data()) {
          if (type == DataType::Bool)
            require(v == 0 || v == 1, "onnx_tensor_invalid", "invalid boolean content");
          if (type == DataType::Int8)
            require(v >= -128 && v <= 127, "onnx_tensor_invalid", "INT8 content out of range");
          if (type == DataType::Float16)
            require(v >= 0 && v <= 65535, "onnx_tensor_invalid", "FLOAT16 bits out of range");
          appendBits(data, static_cast<std::uint64_t>(v), bits(type).value() / 8);
        }
      }
    }
    p.length = data.size();
  }
  require(data.size() == expected, "onnx_parameter_content",
          "tensor content length mismatch: " + t.name());
  p.digest = digest(data);
  const auto count = elements(p.signature.shape);
  const auto width = bits(p.signature.dtype).value() / 8;
  if (p.signature.dtype == DataType::Int64 || p.signature.dtype == DataType::Int32 ||
      p.signature.dtype == DataType::Bool || p.signature.dtype == DataType::Int8) {
    p.integers = Shape{};
    p.integers->reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto v = readBits(data, i * width, width);
      std::int64_t value = 0;
      if (width == 8)
        std::memcpy(&value, &v, 8);
      else if (p.signature.dtype == DataType::Int32)
        value = static_cast<std::int32_t>(static_cast<std::uint32_t>(v));
      else if (p.signature.dtype == DataType::Int8)
        value = static_cast<std::int8_t>(v);
      else {
        require(v <= 1, "onnx_tensor_invalid", "invalid boolean content");
        value = static_cast<std::int64_t>(v);
      }
      p.integers->push_back(value);
    }
  } else if (count <= 4096 &&
             (p.signature.dtype == DataType::Float32 || p.signature.dtype == DataType::Float64)) {
    p.floating = std::vector<double>{};
    for (std::size_t i = 0; i < count; ++i) {
      auto v = readBits(data, i * width, width);
      if (width == 4) {
        const auto u = static_cast<std::uint32_t>(v);
        float f;
        std::memcpy(&f, &u, 4);
        p.floating->push_back(f);
      } else {
        double f;
        std::memcpy(&f, &v, 8);
        p.floating->push_back(f);
      }
    }
  }
  return p;
}
SV integers(const Shape &values) {
  SV::Array a;
  for (auto v : values)
    a.emplace_back(v);
  return SV(std::move(a));
}
SV tensorValue(const Payload &p) {
  SV::Object obj{{"shape", integers(p.signature.shape)},
                 {"dtype", SV(static_cast<std::int64_t>(p.signature.dtype))},
                 {"content_sha256", SV(p.digest)}};
  if (p.integers)
    obj.emplace("integers", integers(*p.integers));
  if (p.floating) {
    SV::Array a;
    for (double v : *p.floating)
      a.emplace_back(v);
    obj.emplace("floating", SV(std::move(a)));
  }
  return SV(std::move(obj));
}
struct Fields {
  SV fields;
  std::optional<Payload> constant;
};
Fields attributes(const onnx::NodeProto &n, const std::string &source) {
  Fields result;
  SV::Object obj;
  for (const auto &a : n.attribute()) {
    require(!a.name().empty() && !obj.count(a.name()) && a.ref_attr_name().empty(),
            "onnx_attribute_invalid", "duplicate or referenced ONNX attribute");
    SV value;
    switch (a.type()) {
    case onnx::AttributeProto::INT:
      value = SV(a.i());
      break;
    case onnx::AttributeProto::FLOAT:
      value = SV(static_cast<double>(a.f()));
      break;
    case onnx::AttributeProto::STRING:
      value = SV(a.s());
      break;
    case onnx::AttributeProto::INTS:
      value = integers(Shape(a.ints().begin(), a.ints().end()));
      break;
    case onnx::AttributeProto::FLOATS: {
      SV::Array values;
      for (auto f : a.floats())
        values.emplace_back(static_cast<double>(f));
      value = SV(std::move(values));
      break;
    }
    case onnx::AttributeProto::TENSOR: {
      const auto p = payload(a.t(), source);
      value = tensorValue(p);
      result.constant = p;
      break;
    }
    default:
      throw Error("onnx_attribute_unsupported", "unsupported ONNX attribute type: " + a.name());
    }
    obj.emplace(a.name(), std::move(value));
  }
  result.fields = SV(std::move(obj));
  return result;
}
} // namespace
ImportResult importOnnx(const std::string &bytes, const Registry &registry,
                        const std::string &path) {
  onnx::ModelProto model;
  require(bytes.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
              model.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())),
          "onnx_parse_invalid", "cannot parse ONNX ModelProto");
  require(model.has_graph() && model.ir_version() >= 7 && model.ir_version() <= 13,
          "onnx_ir_unsupported", "unsupported ONNX IR version");
  require(model.functions_size() == 0 && model.training_info_size() == 0,
          "onnx_feature_unsupported", "ONNX functions or training graphs are unsupported");
  require(model.opset_import_size() == 1 && model.opset_import(0).domain().empty() &&
              model.opset_import(0).version() == 17,
          "onnx_opset_unsupported", "only standard ONNX domain opset 17 is supported");
  const auto &g = model.graph();
  require(g.sparse_initializer_size() == 0 && g.quantization_annotation_size() == 0,
          "onnx_feature_unsupported",
          "sparse parameters or quantization annotations are unsupported");
  ImportResult result;
  result.profile = "onnx-static/1";
  result.sourceDigest = digest(bytes);
  result.irVersion = model.ir_version();
  result.opsets[""] = 17;
  result.producer = model.producer_name();
  result.producerVersion = model.producer_version();
  std::map<std::string, TensorSignature> declared;
  auto declare = [&](const auto &v) {
    require(!v.name().empty(), "onnx_name_invalid", "empty value name");
    const auto sig = signature(v);
    const auto found = declared.find(v.name());
    require(found == declared.end() ||
                (found->second.shape == sig.shape && found->second.dtype == sig.dtype),
            "onnx_type_conflict", "conflicting value declarations: " + v.name());
    declared[v.name()] = sig;
  };
  for (const auto &v : g.input())
    declare(v);
  for (const auto &v : g.output())
    declare(v);
  for (const auto &v : g.value_info())
    declare(v);
  TensorId nextTensor = 1;
  OperatorId nextNode = 1;
  std::map<std::string, TensorId> ids;
  std::map<TensorId, TensorSignature> signatures;
  std::map<TensorId, std::optional<Shape>> integerValues;
  std::map<TensorId, std::optional<std::vector<double>>> floatingValues;
  std::string contentIdentity = result.sourceDigest;
  std::vector<Tensor> tensors;
  std::vector<std::unique_ptr<Operator>> nodes;
  std::vector<TensorId> modelInputs, modelOutputs;
  auto add = [&](const std::string &name, const TensorSignature &sig, ValueKind kind,
                 std::optional<ParameterRef> param = std::nullopt) {
    require(!name.empty() && !ids.count(name), "onnx_value_duplicate",
            "duplicate ONNX value: " + name);
    if (declared.count(name))
      require(declared.at(name).shape == sig.shape && declared.at(name).dtype == sig.dtype,
              "onnx_shape_mismatch", "inferred shape/type disagrees with declaration: " + name);
    const auto id = nextTensor++;
    ids[name] = id;
    signatures[id] = sig;
    tensors.emplace_back(id, name, sig.shape, sig.dtype, std::nullopt, kind, std::move(param));
    return id;
  };
  std::set<std::string> initializerNames;
  for (const auto &t : g.initializer()) {
    require(initializerNames.insert(t.name()).second, "onnx_value_duplicate",
            "duplicate initializer");
    const auto p = payload(t, path);
    const auto id = add(t.name(), p.signature, ValueKind::Parameter,
                        ParameterRef{result.sourceDigest, t.name(), p.digest});
    integerValues[id] = p.integers;
    floatingValues[id] = p.floating;
    contentIdentity += p.digest;
    result.parameters.push_back({id, t.name(), p.digest, p.location, p.offset, p.length});
  }
  for (const auto &input : g.input()) {
    require(!initializerNames.count(input.name()), "onnx_overridable_initializer",
            "overridable initializer inputs are unsupported");
    modelInputs.push_back(add(input.name(), signature(input), ValueKind::Input));
  }
  std::set<std::string> outputs;
  for (const auto &n : g.node()) {
    require(n.domain().empty() && n.overload().empty(), "onnx_domain_unsupported",
            "custom domains and overloads are unsupported");
    for (const auto &out : n.output())
      require(!out.empty() && !ids.count(out) && outputs.insert(out).second, "onnx_value_duplicate",
              "empty or duplicate node output: " + out);
  }
  std::vector<bool> done(g.node_size());
  std::size_t remaining = done.size();
  while (remaining) {
    bool progress = false;
    for (int i = 0; i < g.node_size(); ++i) {
      if (done[i])
        continue;
      const auto &n = g.node(i);
      bool ready = true;
      for (const auto &in : n.input())
        if (!in.empty() && !ids.count(in))
          ready = false;
      if (!ready)
        continue;
      try {
        SourceNode s;
        s.format = "onnx/17";
        s.type = n.op_type();
        s.name = n.name().empty() ? n.op_type() + "/" + std::to_string(i) : n.name();
        s.index = nextNode - 1;
        auto fields = attributes(n, path);
        s.fields = std::move(fields.fields);
        if (fields.constant)
          contentIdentity += fields.constant->digest;
        for (const auto &in : n.input()) {
          const auto id = in.empty() ? 0 : ids.at(in);
          s.inputs.push_back(id);
          s.inputSignatures.push_back(id ? signatures.at(id)
                                         : TensorSignature{{0}, DataType::Float32});
          s.integerInputs.push_back(id ? integerValues[id] : std::nullopt);
          s.floatingInputs.push_back(id ? floatingValues[id] : std::nullopt);
        }
        for (int j = 0; j < n.output_size(); ++j)
          s.outputs.push_back(nextTensor++);
        s.nextTensor = nextTensor;
        auto expansion = registry.expand(s);
        if (expansion.nodes.size() == 1 && expansion.nodes[0]->inputs() != s.inputs)
          expansion.notes.push_back(
              "static source inputs folded into validated typed attributes; not a fusion saving");
        for (auto &t : expansion.tensors) {
          require(t.id() >= s.nextTensor && !signatures.count(t.id()), "factory_invalid",
                  "invalid normalized tensor ID");
          // ONNX 节点名允许重复；内部值名不能覆盖现有或未来的来源值。
          auto normalizedName = t.name();
          while (ids.count(normalizedName) || outputs.count(normalizedName) ||
                 declared.count(normalizedName))
            normalizedName += "/" + std::to_string(t.id());
          if (normalizedName != t.name())
            t = Tensor(t.id(), normalizedName, t.shape(), t.dtype(), t.signature().axes, t.kind(),
                       t.parameter(), t.axesOrigin());
          nextTensor = std::max(nextTensor, t.id() + 1);
          ids[t.name()] = t.id();
          signatures[t.id()] = t.signature();
          tensors.push_back(std::move(t));
        }
        std::map<TensorId, std::string> original;
        for (int j = 0; j < n.output_size(); ++j)
          original.emplace(s.outputs[j], n.output(j));
        for (auto &op : expansion.nodes) {
          require(op && op->id() == nextNode++, "factory_invalid",
                  "normalized node IDs must be consecutive");
          std::vector<TensorSignature> in;
          std::vector<std::optional<Shape>> values;
          for (auto id : op->inputs()) {
            require(signatures.count(id), "factory_invalid", "normalized input unresolved");
            in.push_back(signatures.at(id));
            values.push_back(integerValues[id]);
          }
          const auto inferred = op->infer(in);
          const auto evaluated = op->evaluateIntegers(in, values);
          require(inferred.size() == op->outputs().size() && evaluated.size() == inferred.size(),
                  "factory_invalid", "inference output count mismatch");
          for (std::size_t j = 0; j < inferred.size(); ++j) {
            const auto id = op->outputs()[j];
            const auto &sig = inferred[j];
            if (original.count(id)) {
              const auto &name = original.at(id);
              require(!ids.count(name), "onnx_value_duplicate", "duplicate output");
              if (declared.count(name))
                require(declared.at(name).shape == sig.shape &&
                            declared.at(name).dtype == sig.dtype,
                        "onnx_shape_mismatch", "inferred shape/type mismatch: " + name);
              ids[name] = id;
              signatures[id] = sig;
              tensors.emplace_back(id, name, sig.shape, sig.dtype, std::nullopt,
                                   ValueKind::OperatorResult);
            } else
              require(signatures.count(id) && signatures.at(id).shape == sig.shape &&
                          signatures.at(id).dtype == sig.dtype,
                      "factory_invalid", "normalized signature mismatch");
            integerValues[id] = evaluated[j];
            if (fields.constant && op->inputs().empty())
              floatingValues[id] = fields.constant->floating;
          }
          NodeOrigin origin{op->id(), static_cast<std::size_t>(i), n.name(), {}, {}, {}};
          origin.inputs.assign(n.input().begin(), n.input().end());
          origin.outputs.assign(n.output().begin(), n.output().end());
          origin.normalizations = expansion.notes;
          if (std::any_of(evaluated.begin(), evaluated.end(),
                          [](const auto &v) { return v.has_value(); }))
            origin.normalizations.push_back(
                "static_integer_evaluation: retained in graph; not a fusion saving");
          result.nodeOrigins.push_back(std::move(origin));
          nodes.push_back(std::move(op));
        }
        for (const auto &note : expansion.notes)
          result.diagnostics.push_back({"normalization", note});
      } catch (const Error &e) {
        throw Error(e.code(), "ONNX node " + std::to_string(i) + " " + n.name() + " (" +
                                  n.op_type() + "): " + e.what());
      }
      done[i] = true;
      --remaining;
      progress = true;
    }
    require(progress, "onnx_graph_invalid", "graph is cyclic or references missing input values");
  }
  for (const auto &out : g.output()) {
    require(ids.count(out.name()), "onnx_output_missing", "graph output unresolved: " + out.name());
    modelOutputs.push_back(ids.at(out.name()));
  }
  for (const auto &value : declared)
    require(ids.count(value.first), "onnx_value_missing",
            "declared value not defined: " + value.first);
  result.modelContentDigest = digest(contentIdentity);
  result.graph =
      std::make_shared<DAG>("onnx:" + result.modelContentDigest, g.name(), std::move(tensors),
                            std::move(nodes), modelOutputs, registry.id(), modelInputs);
  result.diagnostics.push_back(
      {"execution_not_established",
       "static semantic import; device capabilities, plan and cost are unknown"});
  return result;
}
} // namespace opfusion::frontend::detail
