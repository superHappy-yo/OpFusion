"""完整模型验收：独立运行时中间签名、原始参数摘要和一对多规范化来源。缺模型直接失败。"""
import argparse
import collections
import copy
import hashlib
import json
import pathlib
import subprocess
import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=pathlib.Path, required=True)
parser.add_argument("--cache", type=pathlib.Path, required=True)
args = parser.parse_args()
for name, expected_input, expected_output in (("resnet50", [1, 3, 224, 224], [1, 1000]),
                                               ("yolo11n", [1, 3, 640, 640], [1, 84, 8400])):
    path = args.cache / (name + ".onnx")
    provenance = path.with_suffix(".provenance.json")
    assert path.is_file() and provenance.is_file(), "required acceptance model/provenance missing; run PrepareModels.py"
    info = json.loads(provenance.read_text())
    assert info["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key, version in {"torch": "2.5.1", "torchvision": "0.20.1", "onnx": "1.20.0", "ultralytics": "8.3.0"}.items():
        assert info[key].split("+")[0] == version
    assert info["options"]["opset"] == 17 and info["options"]["batch"] == 1
    assert not any(info["options"][k] for k in ("dynamic", "half", "nms", "simplify"))
    assert info["input_shape"] == expected_input and info["dtype"] == "float32"
    assert info["classes"] == (1000 if name=="resnet50" else 80)
    assert info["weights"], "weight digests are required"
    for entry in info["weights"]:
        weight = (args.cache / entry["name"]).resolve()
        assert weight.is_relative_to(args.cache.resolve()) and weight.is_file()
        assert hashlib.sha256(weight.read_bytes()).hexdigest() == entry["sha256"]
    process = subprocess.run([str(args.binary), "inspect", str(path), "--json"], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["source"]["sha256"] == info["model_sha256"]
    assert result["source"]["opsets"] == {"": 17}
    assert result["execution"]["cost"] is None and result["execution"]["status"] == "not_established"
    actual = {t["name"]: t for t in result["tensors"]}
    by_id = {t["uid"]: t for t in result["tensors"]}
    assert result["modeling"]["axis_models"] == {"known": len(by_id), "unknown": 0, "unsupported": 0}
    for tensor in by_id.values():
        assert tensor["axis_model"]["status"] == "known"
        assert len(tensor["logical_axes"]) == len(tensor["shape"])
        for position, axis in enumerate(tensor["logical_axes"]):
            assert axis["position"] == position and axis["lower"] == 0 and axis["upper"] == tensor["shape"][position]
            if tensor["axis_model"]["producer"] is not None:
                assert axis["definitions"], (tensor["name"], position)
        if not tensor["shape"]:
            assert tensor["axis_model"]["origin"] == "scalar_no_axes"
    nodes = {n["id"]: n for n in result["nodes"]}
    assert result["schema"] == "opfusion-inspection/4"
    assert result["modeling"]["index_relations"] == {"known": len(nodes), "unknown": 0, "unsupported": 0}
    for node in nodes.values():
        semantics = node["index_semantics"]
        assert semantics["status"] == "known" and semantics["pieces"]
        assert semantics["reason"] is None
        if node["type"] == "Softmax":
            assert semantics["pieces"][0]["reduction"] == "normalized_exponential"
        if node["type"] == "Shape":
            assert semantics["pieces"][0]["shape_reads"]
        if node["type"] == "Gemm" and len(node["inputs"]) == 3:
            assert semantics["pieces"][0]["inputs"][-1]["phase"] == "after_reduction"
    if name == "yolo11n":
        attention = actual["/model.10/m/m.0/attn/Reshape_output_0"]
        assert attention["shape"] == [1, 2, 128, 400]
        axes = attention["logical_axes"]
        assert [a["definitions"][0]["structure"] for a in axes] == ["singleton", "split", "split", "merged"]
        for axis in axes[1:3]:
            source = axis["definitions"][0]["sources"]
            assert len(source) == 1 and source[0]["dimension"] == 1
            assert by_id[source[0]["tensor"]]["shape"] == [1, 256, 20, 20]
        assert [s["dimension"] for s in axes[3]["definitions"][0]["sources"]] == [2, 3]
    for node in nodes.values():
        for output in node["outputs"]:
            for axis in by_id[output]["logical_axes"]:
                for definition in axis["definitions"]:
                    piece = node["index_semantics"]["pieces"][definition["piece"]]
                    assert definition["piece"] in by_id[output]["axis_model"]["relation_pieces"]
                    for source in definition["sources"]:
                        assert source["tensor"] == node["inputs"][source["input_port"]]
                        read = piece["inputs"][source["input_access"]]
                        assert source["coordinate"] == read["coordinates"][source["dimension"]]
    model = onnx.load(path)
    assert len(model.graph.input) == len(model.graph.output) == 1
    assert by_id[result["model_inputs"][0]]["shape"] == expected_input
    assert by_id[result["model_outputs"][0]]["shape"] == expected_output
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    declarations = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    outputs = {v.name for v in inferred.graph.output}
    for node in model.graph.node:
        for out in node.output:
            assert out in declarations, "reference type unavailable: " + out
            if out not in outputs:
                inferred.graph.output.append(copy.deepcopy(declarations[out]))
                outputs.add(out)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(inferred.SerializeToString(), options, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    values = session.run(None, {inp.name: np.zeros(expected_input, np.float32)})
    checked = 0
    for out, data in zip(session.get_outputs(), values):
        assert actual[out.name]["shape"] == list(data.shape), (out.name, actual[out.name], data.shape)
        assert actual[out.name]["dtype"] == str(data.dtype)
        checked += 1
    origins = collections.defaultdict(list)
    for origin in result["source_nodes"]:
        origins[origin["index"]].append(origin)
    for index, source in enumerate(model.graph.node):
        group = origins[index]
        biased = source.op_type == "Conv" and len(source.input) == 3
        assert len(group) == (3 if biased else 1), (index, source.op_type, group)
        for item in group:
            assert item["input_names"] == list(source.input) and item["output_names"] == list(source.output)
        if biased:
            lowered = [nodes[item["id"]] for item in group]
            assert [n["type"] for n in lowered] == ["Conv", "Reshape", "Add"]
            conv, reshape, add = lowered
            assert len(conv["inputs"]) == 2 and len(add["inputs"]) == 2
            expected = actual[source.output[0]]["shape"]
            assert by_id[conv["outputs"][0]]["shape"] == expected
            assert by_id[reshape["outputs"][0]]["shape"] == [1, expected[1]] + [1] * (len(expected) - 2)
            assert add["outputs"] == [actual[source.output[0]]["uid"]]
    parameters = {p["name"]: p for p in result["parameters"]}
    assert len(parameters) == len(model.graph.initializer)
    for parameter in model.graph.initializer:
        array = onnx.numpy_helper.to_array(parameter)
        assert actual[parameter.name]["shape"] == list(array.shape)
        assert actual[parameter.name]["dtype"] == str(array.dtype)
        assert actual[parameter.name]["value_kind"] == "parameter"
        assert parameters[parameter.name]["sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    print(f"{name}: {checked} original node outputs and all parameters matched; normalization mappings passed")
