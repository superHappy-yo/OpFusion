#pragma once
#include "opfusion/registry/Registry.h"

#include <initializer_list>
namespace opfusion::frontend::onnxGraph {
// 仅处理 ONNX 边界字段，不把来源字段袋传入领域算子。
void known(const SourceNode &, std::initializer_list<const char *>);
std::int64_t integer(const SourceNode &, const char *, std::int64_t);
double number(const SourceNode &, const char *, double);
std::string text(const SourceNode &, const char *, const char *);
Shape integers(const SourceNode &, const char *, Shape fallback = {});
const Shape &constant(const SourceNode &, std::size_t port);
OperatorId id(const SourceNode &);
SourceBinding binding(SemanticIdentity, const char *, OperatorFactory, SourceNormalizer = {});
} // namespace opfusion::frontend::onnxGraph
