"""通过 C++ CLI 校验装配与导入边界；Python 仅作黑盒测试，不承担生产逻辑。"""
import argparse
import collections
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, Optional, Union


JsonObject = Dict[str, Any]

# argparse 的外部选项保留连字符，dest 显式绑定本文件使用的驼峰属性。
parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True, type=pathlib.Path)
parser.add_argument("--benchmark-root", dest="benchmarkRoot", required=True, type=pathlib.Path)
args, rest = parser.parse_known_args()
binary = args.binary.resolve()
benchmarkRoot = args.benchmarkRoot.resolve()


def run(*arguments: Union[str, pathlib.Path]) -> "subprocess.CompletedProcess[str]":
    """调用指定 C++ 二进制并保留两个输出通道；失败由测试断言，不替换实现。"""
    return subprocess.run([str(binary), *map(str, arguments)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def manifestFixture(pluginId: str = "cli-fixture", contributionId: Optional[str] = None) -> str:
    """构造独立的 schema 2 测试清单；没有真实工厂或已链接插件的成功路径。"""
    manifest = f'''manifest_schema_version = "2.0.0"
plugin_id = "{pluginId}"
plugin_version = "1.0.0"
plugin_api_range = ">=5.0.0,<6.0.0"
kind = "operators"
requires = []
external_dependencies = []
artifact_refs = ["description.txt"]
'''
    if contributionId is None:
        return manifest + "contributions = []\n"
    return manifest + f'''
[[contributions]]
type = "operator"
id = "{contributionId}"
description_ref = "description.txt"
factory_ref = "unlinkedCliFixture"
'''


class CliTests(unittest.TestCase):
    """区分纯语义导入、静态插件校验和未建立的执行条件，不产生实机证据。"""

    def success(self, *arguments: Union[str, pathlib.Path]) -> JsonObject:
        """成功响应必须来自 stdout 的完整 JSON，非零退出不能伪装成结果。"""
        process = run(*arguments)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def reject(self, document: JsonObject, code: Optional[str] = None) -> JsonObject:
        """用临时来源输入检查退出码与结构化错误，不修改仓库中的 benchmark。"""
        with tempfile.TemporaryDirectory(prefix="opfusion-input-") as directory:
            path = pathlib.Path(directory) / "input.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            process = run("inspect", path, "--json")
        self.assertEqual(process.returncode, 2, process.stdout)
        error = json.loads(process.stderr)["error"]
        self.assertTrue(error["code"])
        self.assertTrue(error["message"])
        if code:
            self.assertEqual(error["code"], code, error)
        return error

    def rejectManifest(self, manifest: str, code: str, message: str,
                       pluginId: str = "cli-fixture") -> None:
        """只提供最小清单和普通描述文件；同时核对错误码和实际到达的失败阶段。"""
        with tempfile.TemporaryDirectory(prefix="opfusion-plugin-") as directory:
            root = pathlib.Path(directory)
            package = root / "operators" / pluginId
            package.mkdir(parents=True)
            (package / "description.txt").write_text(
                "Manifest validation fixture only; no C++ factory is provided.\n", encoding="utf-8")
            (package / "plugin.toml").write_text(manifest, encoding="utf-8")
            process = run("operators", "--json", "--plugins-root", root)
        self.assertEqual(process.returncode, 2, process.stdout)
        error = json.loads(process.stderr)["error"]
        self.assertEqual(error["code"], code, error)
        self.assertIn(message, error["message"], error)

    def source(self, relative: str = "resnet/resnet50-layer1.json") -> JsonObject:
        """仅跳过依赖缺失文件的当前用例或 subTest，不跳过整套 CLI 检查。"""
        path = benchmarkRoot / relative
        if not path.is_file():
            self.skipTest("local benchmark data is absent: " + relative)
        return json.loads(path.read_text(encoding="utf-8"))

    def assertBuiltins(self, result: JsonObject) -> None:
        """固定当前内置语义/绑定基线；清单只属于测试，不参与生产发现。"""
        self.assertEqual(result["schema"], "opfusion-operators/2")
        self.assertTrue(result["registry_id"].startswith("registry/cpp-static/4:"))
        self.assertEqual(len(result["operators"]), 22)
        self.assertEqual({item["name"] for item in result["operators"]}, {
            "Conv", "Add", "ReLU", "Sigmoid", "Mul", "Slice", "Concat",
            "Resize", "MaxPool", "MatMul", "Constant", "Div", "Sub", "Flatten", "Gather", "Gemm", "GlobalAveragePool", "Reshape", "Shape", "Softmax", "Split", "Transpose"})
        self.assertTrue(all(item["support"] == "semantic_model_only" for item in result["operators"]))
        self.assertTrue(all("semantic" not in item and item["namespace"]=="opfusion" for item in result["operators"]))
        self.assertEqual(sum(len(item["bindings"]) for item in result["operators"]), 31)
        matmul = next(item for item in result["operators"] if item["name"] == "MatMul")
        self.assertEqual([b["format"] for b in matmul["bindings"]], ["onnx/17"])

    def testOperatorObjects(self) -> None:
        """语义身份与可导入来源分离；同一二进制的注册内容身份可重复查询。"""
        result = self.success("operators", "--json")
        self.assertBuiltins(result)
        self.assertEqual(result["registry_id"], self.success("operators", "--json")["registry_id"])

    def testAllBenchmarks(self) -> None:
        """逐图核对逻辑值、来源提示与端口；总计只对完整的八图数据集成立。"""
        expected = {
            "resnet/resnet50-layer1.json": 32, "resnet/resnet50-layer2.json": 42,
            "resnet/resnet50-layer3.json": 62, "resnet/resnet50-layer4.json": 32,
            "yolo/yolo11-c3k2-large.json": 42, "yolo/yolo11-c3k2-small.json": 20,
            "yolo/yolo11-fpn-p4.json": 22, "yolo/yolo11-sppf.json": 12,
        }
        counts = collections.Counter()
        tensors = 0
        for relative, expectedNodes in expected.items():
            with self.subTest(graph=relative):
                source = self.source(relative)
                result = self.success("inspect", benchmarkRoot / relative, "--json")
                self.assertEqual(result["schema"], "opfusion-inspection/4")
                self.assertEqual(result["node_count"], expectedNodes)
                self.assertEqual(result["tensor_count"], len(source["tensors"]))
                self.assertEqual(result["shape_validation"], "passed")
                self.assertEqual(result["profile"], "benchmark-json/1")
                self.assertFalse(result["model_outputs_complete"])
                self.assertEqual(result["execution"]["status"], "not_established")
                self.assertEqual(result["execution"]["candidate_search"], "not_searched")
                self.assertIsNone(result["execution"]["cost"])
                self.assertEqual(result["execution"]["parameter_contents"], "not_provided")
                byUid = {tensor["uid"]: tensor for tensor in source["tensors"]}
                for tensor in result["tensors"]:
                    original = byUid[tensor["uid"]]
                    self.assertEqual(tensor["shape"], original["dims"])
                    self.assertEqual(tensor["dtype"], "float32")
                    self.assertEqual(tensor["source_layout"]["strides"], original["strides"])
                    self.assertEqual(tensor["source_layout"]["stride_units"], "unspecified")
                nodes = {node["id"]: node for node in result["nodes"]}
                for edge in result["edges"]:
                    self.assertEqual(nodes[edge["producer"]]["outputs"][edge["output_port"]], edge["tensor"])
                    self.assertEqual(nodes[edge["consumer"]]["inputs"][edge["input_port"]], edge["tensor"])
                counts.update(result["operator_counts"])
                tensors += result["tensor_count"]
        with self.subTest(graph="aggregate"):
            missing = [name for name in expected if not (benchmarkRoot / name).is_file()]
            if missing:
                self.skipTest("aggregate counts require all benchmark files: " + ", ".join(missing))
            self.assertEqual(dict(counts), {"Conv": 71, "Add": 91, "ReLU": 48, "Sigmoid": 19,
                                          "Mul": 19, "Slice": 6, "Concat": 6, "Resize": 1, "MaxPool": 3})
            self.assertEqual(sum(counts.values()), 264)
            self.assertEqual(tensors, 415)

    def testDeclaredOutputIsNotAnInferenceOracle(self) -> None:
        """篡改声明输出必须被输入驱动的推导发现，不能把声明当成答案。"""
        source = self.source()
        output = source["nodes"][0]["outputs"]["y_tensor_uid"]
        next(tensor for tensor in source["tensors"] if tensor["uid"] == output)["dims"][2] += 1
        self.reject(source, "shape_mismatch")

    def testUnknownSourceOperator(self) -> None:
        """未知来源选择器明确拒绝，不借用默认算子或设备实现。"""
        source = self.source()
        source["nodes"][0]["type"] = "UnregisteredOperator"
        self.reject(source, "operator_unknown")

    def testMissingTensor(self) -> None:
        """端口引用必须指向已声明的逻辑值。"""
        source = self.source()
        source["nodes"][0]["inputs"]["x_tensor_uid"] = 999999
        self.reject(source, "tensor_missing")

    def testCycle(self) -> None:
        """节点读取自身输出必须被判为图环，而不是按来源数组顺序执行。"""
        source = self.source()
        source["nodes"][0]["inputs"]["x_tensor_uid"] = source["nodes"][0]["outputs"]["y_tensor_uid"]
        self.reject(source, "graph_cycle")

    def testDuplicateUid(self) -> None:
        """逻辑值身份不可因内容相同而重复登记或覆盖。"""
        source = self.source()
        source["tensors"].append(copy.deepcopy(source["tensors"][0]))
        self.reject(source, "benchmark_duplicate_uid")

    def testUnknownSemanticAttribute(self) -> None:
        """未解释的语义字段不得被静默丢弃，诊断需保留字段上下文。"""
        source = self.source()
        source["nodes"][0]["parameters"]["unrecognized_semantics"] = 7
        error = self.reject(source)
        self.assertIn("unrecognized_semantics", error["message"])

    def testNonstandardRelu(self) -> None:
        """来源子集不把带上限的激活伪装成现有 ReLU 语义。"""
        source = self.source()
        relu = next(node for node in source["nodes"] if node.get("inputs", {}).get("operation") == "relu_fwd")
        relu["inputs"]["relu_upper_clip"] = 6.0
        self.reject(source)

    def testNoGroupGuess(self) -> None:
        """权重通道不匹配时拒绝，不能靠形状猜测未声明的卷积分组。"""
        source = self.source()
        weight = source["nodes"][0]["inputs"]["w_tensor_uid"]
        next(tensor for tensor in source["tensors"] if tensor["uid"] == weight)["dims"][1] //= 2
        self.reject(source)

    def testSliceBounds(self) -> None:
        """切片边界必须符合输入形状，不能截断非法上界掩盖错误。"""
        source = self.source("yolo/yolo11-c3k2-large.json")
        node = next(node for node in source["nodes"] if node["type"] == "SliceAttributes")
        node["parameters"]["limit_indices"][1] = 1000000
        self.reject(source)

    def testResizeSubset(self) -> None:
        """未支持的坐标变换模式明确拒绝，不扩大现有 Resize 子集。"""
        source = self.source("yolo/yolo11-fpn-p4.json")
        source["nodes"][0]["parameters"]["coordinate_transformation_mode"] = "half_pixel"
        self.reject(source)

    def testOversizeIntegerIsNotRounded(self) -> None:
        """超范围整数字面量不能先舍入成浮点再冒充合法 UID。"""
        source = self.source()
        source["tensors"][0]["uid"] = 2 ** 100
        self.reject(source)

    def testRegistryMetadataValidation(self) -> None:
        """根据实际校验先后构造可达负例，不依赖删除的包或伪造已链接工厂。"""
        manifest = manifestFixture()
        operatorManifest = manifestFixture(contributionId="cli.fixture:Identity")
        builtinSemantic = "opfusion:Add"
        # 包描述先校验贡献的链接状态；依赖负例用空贡献清单才能到达依赖阶段。
        # 同理，重复贡献使用已登记的内置身份，在未链接错误之前即可被拒绝。
        variants = [
            ("oldIndexApi",manifest.replace('plugin_api_range = ">=5.0.0,<6.0.0"','plugin_api_range = ">=4.0.0,<5.0.0"'),
             "api_incompatible","requires the current Operator protocol"),
            ("oldOperatorApi",manifest.replace('plugin_api_range = ">=5.0.0,<6.0.0"','plugin_api_range = ">=3.0.0,<4.0.0"'),
             "api_incompatible","requires the current Operator protocol"),
            ("oldManifestSchema",manifest.replace('manifest_schema_version = "2.0.0"','manifest_schema_version = "1.0.0"'),
             "manifest_invalid","unknown manifest schema"),
            ("incompatibleApi", manifest.replace(
                'plugin_api_range = ">=5.0.0,<6.0.0"', 'plugin_api_range = ">=9.0.0,<10.0.0"'),
             "api_incompatible", "requires the current Operator protocol"),
            ("contractIsNotFactoryApi", manifest.replace(
                'plugin_api_range = ">=5.0.0,<6.0.0"', 'plugin_api_range = "=6.0.0"'),
             "api_incompatible", "requires the current Operator protocol"),
            ("invalidPluginVersion", manifest.replace(
                'plugin_version = "1.0.0"', 'plugin_version = "01.0.0"'),
             "manifest_invalid", "expected three-part version"),
            ("invalidApiRange", manifest.replace(
                'plugin_api_range = ">=5.0.0,<6.0.0"', 'plugin_api_range = ">=3.0.0,"'),
             "manifest_invalid", "empty or malformed version range"),
            ("removedSemanticVersion", operatorManifest + 'semantic_version = "1.0.0"\n',
             "manifest_invalid", "semantic_version was removed"),
            ("duplicateContribution", manifestFixture(contributionId=builtinSemantic),
             "duplicate_contribution", "duplicate contribution ID"),
            ("artifactEscape", manifest.replace(
                'artifact_refs = ["description.txt"]', 'artifact_refs = ["../outside.txt"]'),
             "manifest_invalid", "package reference escapes package"),
            ("descriptionEscape", operatorManifest.replace(
                'description_ref = "description.txt"', 'description_ref = "../outside.txt"'),
             "manifest_invalid", "package reference escapes package"),
            ("missingDependency", manifest.replace('requires = []',
                'requires = [{id = "missing.provider", version_range = "=1.0.0"}]'),
             "dependency_missing", "missing or incompatible provider: missing.provider"),
            ("oldContractVersion", manifest.replace('requires = []',
                'requires = [{id = "contracts", version_range = "=3.0.0"}]'),
             "dependency_missing", "missing or incompatible provider: contracts"),
            ("currentProviderVersions", manifest.replace('requires = []',
                'requires = [{id = "contracts", version_range = "=6.1.0"}, '
                '{id = "builtin-operators", version_range = "=5.0.0"}, '
                '{id = "builtin-source-bindings", version_range = "=5.0.0"}]'),
             "provider_unavailable", "package is not linked"),
            ("invalidRequiresType", manifest.replace('requires = []', 'requires = 7'),
             "manifest_invalid", "missing/array field: requires"),
            ("invalidExternalDependenciesType", manifest.replace(
                'external_dependencies = []', 'external_dependencies = "bad"'),
             "manifest_invalid", "missing/array field: external_dependencies"),
            ("dependencyCycle", manifest.replace('requires = []',
                'requires = [{id = "cli-fixture", version_range = "=1.0.0"}]'),
             "dependency_cycle", "plugin dependency cycle"),
            ("unlinkedOperator", operatorManifest,
             "provider_unavailable", "operator package is not linked; rebuild required"),
            ("unlinkedPackage", manifest, "provider_unavailable", "package is not linked"),
        ]
        for name, changed, code, message in variants:
            with self.subTest(case=name):
                self.rejectManifest(changed, code, message)
        with self.subTest(case="duplicateProviderId"):
            self.rejectManifest(manifestFixture(pluginId="contracts"), "duplicate_contribution",
                                "duplicate plugin ID", pluginId="contracts")

    def testEmptyDiscoveryHasNoHiddenStandardFallback(self) -> None:
        """空可选插件目录保留十个内置语义和九个绑定，不重新依赖已删除的包。"""
        with tempfile.TemporaryDirectory(prefix="opfusion-plugin-") as directory:
            root = pathlib.Path(directory)
            (root / "operators").mkdir()
            result = self.success("operators", "--json", "--plugins-root", root)
            self.assertBuiltins(result)


if __name__ == "__main__":
    # 缺失 benchmark 的跳过决定留在 source/subTest，装配与 manifest 测试始终运行。
    unittest.main(argv=[sys.argv[0], *rest])
