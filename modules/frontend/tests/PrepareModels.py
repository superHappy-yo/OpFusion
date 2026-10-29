"""仅用于锁版验收模型准备；模型、权重、来源清单全部保存在构建缓存。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, required=True)
args = parser.parse_args()
cache = args.cache.resolve()
cache.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(cache / "torch")
os.environ["YOLO_CONFIG_DIR"] = str(cache / "ultralytics-config")
import torch
import torchvision
import onnx
import ultralytics
from ultralytics import YOLO

versions = {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "onnx": onnx.__version__, "ultralytics": ultralytics.__version__}
for key, expected in {"torch": "2.5.1", "torchvision": "0.20.1", "onnx": "1.20.0", "ultralytics": "8.3.0"}.items():
    if versions[key].split("+")[0] != expected:
        raise RuntimeError(f"{key} must be {expected}, got {versions[key]}")
torch.set_num_threads(4)
torch.manual_seed(0)
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval()
torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(cache / "resnet50.onnx"),
                  opset_version=17, input_names=["images"], output_names=["logits"],
                  dynamic_axes=None, do_constant_folding=True)
os.chdir(cache)
YOLO("yolo11n.pt").export(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False,
                         opset=17, half=False, nms=False, device="cpu")
weights = [{"name": str(p.relative_to(cache)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(cache.rglob("*")) if p.suffix in (".pt", ".pth")]
for name, size in (("resnet50", 224), ("yolo11n", 640)):
    file = cache / (name + ".onnx")
    info = {**versions, "model_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "weights": weights, "input_shape": [1, 3, size, size], "dtype": "float32", "classes": 1000 if name=="resnet50" else 80,
            "options": {"opset": 17, "batch": 1, "dynamic": False, "half": False,
                        "nms": False, "simplify": False, "device": "cpu",
                        "do_constant_folding": True}}
    file.with_suffix(".provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(name, info["model_sha256"], flush=True)
