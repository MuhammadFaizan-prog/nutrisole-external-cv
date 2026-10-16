#!/usr/bin/env python3
"""Reproduce the two ONNX files bundled in native/android/app/src/main/assets/models/.

Requirements (versions used for the hashes in SHA256SUMS): Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
ultralytics 8.4.175, onnx 1.23.2, onnxslim 0.1.98, onnxruntime 1.31.0, numpy 2.5.2 (+ gdown for the YOLOv8m weights).
The YOLO11n file is not stored in git (its upstream repository has no licence); CI runs this script with
--only mark360_yolo11n to fetch the original weights from GitHub and export them at build time.
Steps: download the original .pt weights from their public sources, check their SHA-256, export to ONNX
(opset 17, 640x640, static shape, no NMS in graph), then store FP32 weights as FP16 + Cast (fp16_storage.py) so
each file stays well under GitHub's 100 MB limit while inference still runs in FP32.
"""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "native/android/app/src/main/assets/models"
WORK = pathlib.Path(os.environ.get("EXTCV_WORK", "extcv-model-work")); WORK.mkdir(parents=True, exist_ok=True)
MODELS = [
    {"name": "henning_yolov8m", "pt_sha256": "a18624d446e2fea08384949115de0b49446f0be2487fc954a37a9d962f71e5d3",
     "get": ["gdown", "https://drive.google.com/uc?id=1XALsPwM4850LkmqsXIfmGhPj8s_nhuiK"]},  # yolo_fruits_and_vegetables_v1.pt
    {"name": "mark360_yolo11n", "pt_sha256": "4f663749fe9192b60a34d2a71fa26d9c9f17c7ad997a557c50879c095456ed4e",
     "get": ["curl", "-sL", "https://github.com/mark-360/Fruit-Detection/raw/main/Models/Yolo11.pt"]},
]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=[m["name"] for m in MODELS]); args = ap.parse_args()
    from ultralytics import YOLO
    for m in MODELS:
        if args.only and m["name"] != args.only:
            continue
        pt = WORK / f"{m['name']}.pt"
        if not pt.exists():
            cmd = m["get"] + (["-O", str(pt)] if m["get"][0] == "gdown" else ["-o", str(pt)])
            subprocess.run(cmd, check=True)
        assert sha(pt) == m["pt_sha256"], f"{pt} hash mismatch"
        model = YOLO(str(pt))
        exported = model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, half=False, nms=False)
        fp32 = WORK / f"{m['name']}_fp32.onnx"; shutil.move(exported, fp32)
        json.dump([model.names[i] for i in range(len(model.names))], open(WORK / f"{m['name']}_names.json", "w"))
        dst = OUT / f"{m['name']}_640_w16.onnx"
        subprocess.run([sys.executable, str(HERE / "fp16_storage.py"), str(fp32), str(dst)], check=True)
        print(m["name"], "->", dst.name, sha(dst))

if __name__ == "__main__":
    main()
