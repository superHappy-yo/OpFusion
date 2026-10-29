"""ONNX、部署 context 与 shell 入口黑盒回归；Python 仅创建夹具和独立参考。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
args, rest = parser.parse_known_args()
binary = args.binary.resolve()
root = args.root.resolve()

def value(name, shape, dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)

def model(nodes, inputs, outputs, parameters=()):
    return h.make_model(h.make_graph(nodes, "fixture", inputs, outputs, initializer=list(parameters)),
                        opset_imports=[h.make_opsetid("", 17)], ir_version=8)

class OnnxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="opfusion-onnx-")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def inspect(self, doc, expected=None, context=None, filename="model.onnx"):
        path = self.dir / filename
        onnx.save(doc, path)
        command = [str(binary), "inspect", str(path), "--json"]
        if context is not None:
            cp = self.dir / "scenario.json"
            cp.write_text(json.dumps(context))
            command += ["--context", str(cp)]
        p = subprocess.run(command, text=True, capture_output=True)
        if expected is None:
            self.assertEqual(p.returncode, 0, p.stderr)
            return json.loads(p.stdout)
        self.assertEqual(p.returncode, 2, p.stdout)
        err = json.loads(p.stderr)["error"]
        self.assertEqual(err["code"], expected, err)
        return err

    def convolution(self, dimensions=2):
        spatial = [5] * dimensions
        weights = np.arange(4 * 2 * (2 ** dimensions), dtype=np.float32).reshape([4, 2] + [2] * dimensions)
        node = h.make_node("Conv", ["images", "weights", "bias"], ["out"], name="conv", group=2,
                           strides=[1] * dimensions, dilations=[1] * dimensions,
                           pads=[1] * dimensions + [0] * dimensions)
        return model([node], [value("images", [1, 4] + spatial)], [value("out", [1, 4] + spatial)],
                     [nh.from_array(weights, "weights"), nh.from_array(np.arange(4, dtype=np.float32), "bias")])

    def simple(self):
        return model([h.make_node("Mul", ["images", "images"], ["out"])],
                     [value("images", [1, 3, 4])], [value("out", [1, 3, 4])])

    def scenario(self):
        return {"schema": "opfusion-context/1", "id": "scenario",
                "platform": {"id": "platform", "devices": [{"id": "cpu", "node_id": "host", "kind": "cpu"}],
                             "pools": [{"id": "ram", "physical_identity": "host-ram", "physical_capacity_bytes": 4096,
                                        "accessible_devices": ["cpu"]}], "resources": [], "links": []},
                "representations": [{"id": "input", "tensor": "images", "pool": "ram", "dtype": "float32", "layout": "row_major", "packing": "dense", "actual_bytes": 48},
                                    {"id": "output", "tensor": "out", "pool": "ram", "dtype": "float32", "layout": "row_major", "packing": "dense", "actual_bytes": 48}],
                "inputs": [{"id": "source", "tensor": "images", "representation": "input", "data_ref": "caller-input", "ready_us": 0}],
                "outputs": [{"id": "target", "tensor": "out", "representation": "output"}],
                "available_bytes_by_pool": {"ram": 2048},
                "timing": {"start": "inputs_available", "finish": "outputs_delivered", "includes_input_transfers": True, "includes_output_transfers": True}}

    def testStructuralAxisProvenance(self):
        target = nh.from_array(np.array([1, 2, 4, 4], dtype=np.int64), "target")
        doc = model([h.make_node("Reshape", ["input", "target"], ["out"])],
                    [value("input", [1, 8, 2, 2])], [value("out", [1, 2, 4, 4])], [target])
        result = self.inspect(doc)
        self.assertEqual(result["schema"], "opfusion-inspection/4")
        self.assertEqual(result["modeling"]["axis_models"]["unknown"], 0)
        by_name = {t["name"]: t for t in result["tensors"]}
        out = by_name["out"]
        self.assertNotIn("axes_origin", out)
        self.assertEqual(out["axis_model"]["input_tensors"],
                         [by_name["input"]["uid"], by_name["target"]["uid"]])
        axes = out["logical_axes"]
        self.assertEqual([a["definitions"][0]["structure"] for a in axes],
                         ["singleton", "split", "split", "merged"])
        self.assertTrue(all(a["semantic"] is None for a in axes))
        self.assertEqual([s["dimension"] for s in axes[3]["definitions"][0]["sources"]], [2, 3])
        for axis in axes[1:3]:
            source = axis["definitions"][0]["sources"][0]
            self.assertEqual(source["tensor"], by_name["input"]["uid"])
            self.assertEqual(source["dimension"], 1)

    def testGeneratedMetadataAndIndirectAxes(self):
        nodes = [h.make_node("Constant", [], ["literal"],
                             value=nh.from_array(np.array([4, 5], dtype=np.int64))),
                 h.make_node("Constant", [], ["scalar"],
                             value=nh.from_array(np.array(2, dtype=np.int64))),
                 h.make_node("Shape", ["data"], ["shape"]),
                 h.make_node("Gather", ["data", "indices"], ["selected"], axis=0)]
        doc = model(nodes, [value("data", [2, 3]), value("indices", [2], T.INT64)],
                    [value("literal", [2], T.INT64), value("scalar", [], T.INT64),
                     value("shape", [2], T.INT64), value("selected", [2, 3])])
        result = self.inspect(doc)
        tensors = {t["name"]: t for t in result["tensors"]}
        self.assertEqual(tensors["literal"]["logical_axes"][0]["definitions"][0]["structure"], "generated")
        self.assertEqual(tensors["scalar"]["logical_axes"], [])
        self.assertEqual(tensors["scalar"]["axis_model"]["origin"], "scalar_no_axes")
        metadata = tensors["shape"]["logical_axes"][0]["definitions"][0]
        self.assertEqual(metadata["structure"], "shape_metadata")
        self.assertEqual(metadata["shape_reads"][0]["tensor"], tensors["data"]["uid"])
        self.assertEqual(tensors["selected"]["logical_axes"][0]["definitions"][0]["structure"], "indirect")
        self.assertEqual(result["modeling"]["axis_models"], {"known": 6, "unknown": 0, "unsupported": 0})

    def testNdConvAndBiasNormalization(self):
        for d in (1, 2, 3):
            with self.subTest(dimensions=d):
                result = self.inspect(self.convolution(d))
                self.assertEqual([n["type"] for n in result["nodes"]], ["Conv", "Reshape", "Add"])
                attrs = result["nodes"][0]["attributes"]
                self.assertEqual(attrs["spatial_dimensions"], d)
                self.assertEqual(attrs["kernel_shape"], [2] * d)
                self.assertEqual(attrs["groups"], 2)
                self.assertEqual(len(result["source_nodes"]), 3)
                self.assertTrue(all(o["index"] == 0 for o in result["source_nodes"]))
                self.assertEqual(len(result["model_inputs"]), 1)
                self.assertEqual(len(result["parameters"]), 2)
                bias = next(t for t in result["tensors"] if t["name"] == "bias")
                self.assertEqual(bias["logical_axes"][0]["semantic"], {"kind": "channel", "label": ""})
                self.assertEqual(bias["logical_axes"][0]["upper"], 4)
                self.assertEqual(bias["axis_model"]["semantic_annotation_origin"], "derived")
                for node in result["nodes"]:
                    self.assertNotIn("semantic", node)
                    for key in ("compute_type", "accumulation_type", "absolute_error", "relative_error"):
                        self.assertIsNone(node["numerical_requirements"][key])

    def testActualBroadcasting(self):
        r = self.inspect(self.simple())
        self.assertTrue(r["nodes"][0]["traits"]["supports_broadcasting"])
        self.assertFalse(r["nodes"][0]["broadcasting"]["applied"])
        m = model([h.make_node("MatMul", ["a", "b"], ["out"])],
                  [value("a", [2, 2, 3]), value("b", [1, 3, 4])], [value("out", [2, 2, 4])])
        r = self.inspect(m)
        self.assertTrue(r["nodes"][0]["broadcasting"]["applied"])
        self.assertEqual(r["nodes"][0]["broadcasting"]["inputs"][1], {"port": 1, "axes": [0]})

    def testMultipleOutputsAndRepeatedConsumers(self):
        m = model([h.make_node("Split", ["x", "parts"], ["a", "b"], axis=1),
                   h.make_node("Add", ["a", "a"], ["c"])], [value("x", [2, 6])],
                  [value("a", [2, 2]), value("b", [2, 4]), value("c", [2, 2])],
                  [nh.from_array(np.array([2, 4], np.int64), "parts")])
        r = self.inspect(m)
        self.assertEqual(len(r["model_outputs"]), 3)
        self.assertEqual(r["edge_count"], 2)
        self.assertEqual([e["input_port"] for e in r["edges"]], [0, 1])
        self.assertEqual(r["nodes"][1]["inputs"][0], r["nodes"][1]["inputs"][1])

    def testShapeEvaluationWithoutValueInfo(self):
        nodes = [h.make_node("Shape", ["x"], ["shape"]),
                 h.make_node("Gather", ["shape", "index"], ["size"], axis=0),
                 h.make_node("Mul", ["size", "two"], ["double"]),
                 h.make_node("Div", ["double", "two"], ["original"]),
                 h.make_node("Reshape", ["x", "shape"], ["out"])]
        m = model(nodes, [value("x", [2, 3, 4])], [value("out", [2, 3, 4]), value("original", [1], T.INT64)],
                  [nh.from_array(np.array([0], np.int64), "index"), nh.from_array(np.array([2], np.int64), "two")])
        r = self.inspect(m)
        self.assertEqual(r["node_count"], 5)
        self.assertEqual([t["shape"] for t in r["tensors"] if t["name"] == "out"], [[2, 3, 4]])

    def testRepeatedOnnxNodeNames(self):
        m = self.convolution(1)
        m.graph.node.append(h.make_node("Conv", ["out", "weights", "bias"], ["second"], name="conv", group=2, pads=[1, 0]))
        m.graph.output.append(value("second", [1, 4, 5]))
        result = self.inspect(m)
        names = [t["name"] for t in result["tensors"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(result["node_count"], 6)

    def testKnownGatherBounds(self):
        m = model([h.make_node("Gather", ["x", "index"], ["out"], axis=0)],
                  [value("x", [2, 3])], [value("out", [1, 3])],
                  [nh.from_array(np.array([2], np.int64), "index")])
        self.inspect(m, "index_out_of_bounds")

    def testDynamicGatherRelationship(self):
        m = model([h.make_node("Gather", ["x", "indices"], ["out"], axis=1)],
                  [value("x", [2, 3]), value("indices", [2], T.INT64)], [value("out", [2, 2])])
        r = self.inspect(m)
        semantics = r["nodes"][0]["index_semantics"]
        self.assertEqual(semantics["status"], "known")
        lookup = semantics["pieces"][0]["inputs"][1]["coordinates"][1]["lookups"][0]
        self.assertEqual((lookup["input_port"], lookup["valid_min"], lookup["valid_max_exclusive"]), (1, -3, 3))
        self.assertIsNone(r["nodes"][0]["numerical_requirements"]["compute_type"])

    def testStaticShapeReordering(self):
        nodes = [h.make_node("Shape", ["x"], ["shape"]),
                 h.make_node("Transpose", ["shape"], ["same"], perm=[0]),
                 h.make_node("Split", ["same", "parts"], ["first", "remaining"], axis=0),
                 h.make_node("Concat", ["first", "remaining"], ["target"], axis=0),
                 h.make_node("Reshape", ["x", "target"], ["out"])]
        m = model(nodes, [value("x", [2, 3, 4])], [value("out", [2, 3, 4])],
                  [nh.from_array(np.array([1, 2], np.int64), "parts")])
        r = self.inspect(m)
        self.assertEqual(r["modeling"]["index_relations"], {"known": 5, "unknown": 0, "unsupported": 0})
        self.assertTrue(all(n["index_semantics"]["pieces"] for n in r["nodes"]))
        shape = next(n for n in r["nodes"] if n["type"] == "Shape")
        self.assertEqual(shape["attributes"]["values"], [2, 3, 4])

    def testLargeInt64IsExact(self):
        big = 2**53 + 1
        m = model([h.make_node("Constant", [], ["out"], value=nh.from_array(np.array([big], np.int64)))],
                  [], [value("out", [1], T.INT64)])
        r = self.inspect(m)
        self.assertEqual(r["nodes"][0]["attributes"]["integers"], [big])
        self.assertEqual(r["nodes"][0]["attributes"]["content_sha256"], hashlib.sha256(np.array([big],np.int64).tobytes()).hexdigest())
        self.assertIn("static_integer_evaluation", r["source_nodes"][0]["normalizations"][0])

    def testExternalDataContentAndRanges(self):
        m = self.convolution(1)
        t = m.graph.initializer[0]
        raw = t.raw_data
        t.ClearField("raw_data")
        t.data_location = T.EXTERNAL
        for key, val in (("location", "weights.bin"), ("offset", "16"), ("length", str(len(raw)))):
            pair = t.external_data.add(); pair.key = key; pair.value = val
        file = self.dir / "weights.bin"
        file.write_bytes(b"prefix-padding!!" + raw)
        r = self.inspect(m)
        p = next(p for p in r["parameters"] if p["name"] == "weights")
        self.assertEqual(p["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(p["offset"], 16)
        self.assertEqual(p["length"], len(raw))
        changed = bytearray(file.read_bytes()); changed[-1] ^= 1; file.write_bytes(changed)
        changed_result = self.inspect(m)
        self.assertEqual(changed_result["source"]["sha256"], r["source"]["sha256"])
        self.assertNotEqual(changed_result["graph_id"], r["graph_id"])
        file.write_bytes(file.read_bytes()[:-1])
        self.inspect(m, "onnx_external_range")
        file.unlink()
        self.inspect(m, "onnx_external_missing")
        for location in ("../outside.bin", "/tmp/outside.bin"):
            t.external_data[0].value = location
            self.inspect(m, "onnx_external_invalid")
        t.external_data[0].value = "weights.bin"
        file.symlink_to(binary)
        self.inspect(m, "onnx_external_missing")

    def testMalformedModelBoundaries(self):
        variants = []
        m = self.simple(); m.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value"); m.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"; variants.append((m, "onnx_dynamic_shape"))
        m = self.simple(); m.opset_import[0].version = 18; variants.append((m, "onnx_opset_unsupported"))
        m = self.simple(); m.graph.node[0].domain = "custom"; variants.append((m, "onnx_domain_unsupported"))
        m = self.simple(); m.graph.node[0].op_type = "If"; variants.append((m, "operator_unknown"))
        m = self.convolution(); m.graph.input.append(value("weights", [4, 2, 2, 2])); variants.append((m, "onnx_overridable_initializer"))
        m = self.convolution(); m.graph.initializer[0].raw_data = m.graph.initializer[0].raw_data[:-1]; variants.append((m, "onnx_parameter_content"))
        m = self.convolution(); m.graph.node[0].attribute.append(h.make_attribute("strides", [1])); variants.append((m, "onnx_attribute_invalid"))
        m = self.convolution(); m.graph.node[0].attribute.append(h.make_attribute("auto_pad", "SAME_UPPER")); variants.append((m, "unsupported_attribute"))
        m = self.convolution(); m.graph.node[0].attribute.append(h.make_attribute("unknown_policy", 1)); variants.append((m, "unsupported_attribute"))
        m = self.convolution(); m.graph.value_info.append(value("out", [1, 4, 99, 99])); variants.append((m, "onnx_type_conflict"))
        m = self.convolution(); m.graph.initializer.pop(); variants.append((m, "onnx_graph_invalid"))
        m = self.simple(); m.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 5; variants.append((m, "onnx_shape_mismatch"))
        m = self.convolution(); m.graph.initializer[0].ClearField("dims"); variants.append((m, "onnx_parameter_content"))
        for m, code in variants:
            with self.subTest(code=code): self.inspect(m, code)
        path = self.dir / "broken.onnx"; path.write_bytes(b"not protobuf")
        p = subprocess.run([str(binary), "inspect", str(path), "--json"], capture_output=True, text=True)
        self.assertEqual(json.loads(p.stderr)["error"]["code"], "onnx_parse_invalid")

    def testContext(self):
        scenario = self.scenario()
        r = self.inspect(self.simple(), context=scenario)
        self.assertTrue(r["context"]["declarations_validated"])
        self.assertFalse(r["context"]["execution_validated"])
        self.assertEqual(r["context"]["timing_event_binding"], "pending_plan")
        codes = {issue["code"] for issue in r["context"]["issues"]}
        self.assertIn("timing_events_pending_plan", codes)
        self.assertNotIn("timing_scope_unknown", codes)
        self.assertIsNone(r["execution"]["cost"])
        for alter, code in ((lambda s: s["inputs"][0].update(tensor="missing"), "context_tensor_missing"),
                            (lambda s: s["available_bytes_by_pool"].update(ram=8192), "available_exceeds_physical"),
                            (lambda s: s["timing"].update(start_event="invented"), "context_invalid"),
                            (lambda s: s["timing"].update(finish="region_outputs_ready"), "timing_output_coverage"),
                            (lambda s: s["representations"][0].update(actual_bytes=1), "representation_size")):
            changed = copy.deepcopy(scenario); alter(changed)
            with self.subTest(code=code): self.inspect(self.simple(), code, changed)

    def testRunScript(self):
        self.inspect(self.simple(), filename="a model.onnx")
        context = self.dir / "scenario.json"; context.write_text(json.dumps(self.scenario()))
        # 使用临时入口目录保证脚本指向本次实际构建。
        env = dict(os.environ, OPFUSION_BUILD_DIR=str(binary.parent.parent))
        runsh = root / "run.sh"
        command = [str(runsh), "--context", str(context), str(self.dir)]
        process = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(process.stdout.splitlines()), 1)
        self.assertEqual(json.loads(process.stdout)["context"]["id"], "scenario")
        (self.dir / "bad.onnx").write_bytes(b"bad")
        process = subprocess.run([str(runsh), str(self.dir)], env=env, text=True, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(len(process.stdout.splitlines()), 1)
        # shell 展开的通配符和含空格路径必须逐文件运行。
        (self.dir / "bad.onnx").unlink()
        onnx.save(self.simple(), self.dir / "b.onnx")
        process = subprocess.run(["bash", "-c", '"$1" "$2"/*.onnx', "test", str(runsh), str(self.dir)], env=env, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(process.stdout.splitlines()), 2)
        compatible = root / "benchmark/yolo/yolo11-sppf.json"
        shutil.copyfile(compatible, self.dir / "compat.json")
        context.unlink()
        process = subprocess.run([str(runsh), str(self.dir)], env=env, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(process.stdout.splitlines()), 3)
        sppf = next(json.loads(line) for line in process.stdout.splitlines() if json.loads(line)["profile"] == "benchmark-json/1")
        self.assertEqual((sppf["node_count"], sppf["tensor_count"], sppf["edge_count"]), (12, 17, 16))

if __name__ == "__main__":
    unittest.main(argv=[__file__, *rest])
