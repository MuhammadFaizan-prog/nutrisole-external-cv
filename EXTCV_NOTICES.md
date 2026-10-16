# Third-party notices — NutriSole External-CV

This repository is a research build of NutriSole that bundles third-party detection models. The NutriSole app
code itself is unchanged from the NutriSole frontend `main` snapshot except for the files under `native/extcv/`,
the Android model-file module, the build workflow and these notices.

## Models

### YOLOv8m — Fruits and Vegetables Detection (henningheyen)
- Source: https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset (MIT, © 2024 Henning Heyen, see
  `LICENSES/henningheyen-MIT.txt`).
- Weights: `yolo_fruits_and_vegetables_v1.pt` from the Google Drive folder linked in that README
  (https://drive.google.com/drive/folders/1I4mtQK11C3p41pO9raR0trgPVj0eQ2yb), sha256
  `a18624d446e2fea08384949115de0b49446f0be2487fc954a37a9d962f71e5d3`. The Hugging Face card for the same work states
  Apache-2.0.
- Training data: a fruit/vegetable subset of LVIS v1 (https://www.lvisdataset.org) — annotations CC BY 4.0; images from
  COCO (Flickr), each under its own licence.
- Modified here: exported to ONNX and weights stored as FP16 + Cast (`scripts/extcv/`).

### YOLO11n — Fruit Detection (mark-360)
- Source: https://github.com/mark-360/Fruit-Detection, file `Models/Yolo11.pt`, sha256
  `4f663749fe9192b60a34d2a71fa26d9c9f17c7ad997a557c50879c095456ed4e`.
- **No licence is published in that repository** (GitHub reports none), so no reuse rights are granted by default.
  Its weights and the converted ONNX file are therefore **not committed to this repository**. CI fetches the original
  file from the author's GitHub at build time for an experimental speed comparison in the QA APK only. Drop it from
  any public or commercial distribution unless the author grants a licence.

## Ultralytics (AGPL-3.0)
Both models were produced with Ultralytics YOLO (https://github.com/ultralytics/ultralytics), licensed under the GNU
Affero General Public License v3.0 (`LICENSES/AGPL-3.0.txt`). Ultralytics treats trained weights as covered by the
AGPL. Distributing an app that contains them requires making the complete corresponding source available under the
AGPL-3.0 to its users, or obtaining an Ultralytics Enterprise licence (https://www.ultralytics.com/license).

## Runtime libraries (MIT unless stated)
onnxruntime-react-native / ONNX Runtime (MIT, Microsoft) · react-native-vision-camera, -resizer, -worklets (MIT,
Marc Rousavy / Margelo) · react-native-nitro-modules, react-native-nitro-image (MIT, Margelo) · react-native-worklets
(MIT, Software Mansion) · React Native (MIT, Meta) · lucide-react-native (ISC).
