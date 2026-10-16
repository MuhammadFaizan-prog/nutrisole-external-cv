# NutriSole External-CV

A standalone experiment: the NutriSole Android app with its **Scan** screen driven only by external, open-source
fruit and vegetable detectors downloaded from their public sources. No NutriSole-trained model is included.

## Flow (from the FYP proposal's camera feature)

Open app → **Scan Food** → live camera. Detection runs automatically on camera frames:

1. VisionCamera frame output (preview-sized buffers, native pixel format) → GPU resizer to a 640×640 planar RGB float
   tensor ("contain" letterbox).
2. Exactly one inference is in flight. While it runs, new frames are skipped on the camera thread (counted in the HUD).
3. ONNX Runtime (CPU, 4 threads) runs the selected model; the output is decoded (best class per anchor, per-class NMS,
   IoU 0.45) and mapped back to the frame, rotated upright from the frame's orientation, and drawn over the preview.
4. Boxes ≥ 50 % are green with the label; 25–50 % are amber and marked "?" (low confidence). The bottom banner shows
   the top result, "Low confidence" when the top score is under 50 % or a different food overlaps within 10 points,
   or "No food detected".
5. Tap a box or the banner → result screen: name, confidence, model; nutrition per 100 g and portion from the app's
   existing catalogue when the food exists in NutriSole (Apple, Banana, Tomato), with **Log this food**,
   **Nutrition details** and **Market reference** (the app has no verified prices yet, as on main).
   Low-confidence results are never used automatically: the screen offers **Choose food manually** instead.
6. **Quality grade: "Grading not available in this build"** — these models have no freshness/quality head.

Top bar: close, **model picker** (YOLOv8m / YOLO11n), debug sheet (manual rotation correction, HUD toggle) and
**Models & licences**. The perf HUD shows model, inference ms, end-to-end ms (frame captured → boxes), inferences/s,
frames skipped, camera-dropped frames, frame size/orientation and whether the loaded model's SHA-256 matched.

Everything else in the app is unchanged from `main`. The APK installs as **NutriSole ExtCV**
(`com.muhammadfaizan.nutrisole.extcv`), beside any other NutriSole build.

## Models

| | YOLOv8m (default) | YOLO11n (fast toggle) |
|---|---|---|
| Source | [henningheyen/Fruits-And-Vegetables-Detection-Dataset](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset) | [mark-360/Fruit-Detection](https://github.com/mark-360/Fruit-Detection) |
| Weights | `yolo_fruits_and_vegetables_v1.pt` from the repo's public Google Drive folder, sha256 `a18624d4…71e5d3` | `Models/Yolo11.pt`, sha256 `4f663749…456ed4e` |
| Classes | 63 LVIS fruit/vegetable classes | apple, mango, banana, orange |
| Bundled file | `henning_yolov8m_640_w16.onnx`, 51,953,747 B, sha256 `698f7b1e64b2eac475f4d84d491d57e41f683bbcfc902c50f7761bbf970de37c` | `mark360_yolo11n_640_w16.onnx`, 5,377,616 B, sha256 `c8f468937bf146483dd70e3e6c106e4ad789a01a2a94e6fa9d27043157bfa426` |

Both were exported with Ultralytics 8.4.175 (opset 17, 640, static, no NMS). FP32 weights are stored as FP16
initializers with Cast-to-FP32 nodes (`scripts/extcv/fp16_storage.py`): the files are half size and stay under
GitHub's 100 MB limit, while ONNX Runtime folds the casts at load and computes in FP32. The export-time `date`
metadata is removed so a re-export gives the identical file.

**YOLO11n is not stored in this repository**: its upstream repository has no licence, so CI downloads the original
`Yolo11.pt` from mark-360's GitHub at build time (hash-checked), exports it with pinned versions and must reproduce
the exact SHA-256 above before it is packed into the APK. Only the YOLOv8m ONNX (MIT/AGPL/CC BY sources) is committed. Measured in the sandbox against
the FP32 export: max class-score difference 0.010, max box-coordinate difference 2.2 px (of 640). Rebuild with
`scripts/extcv/build_models.py`; CI checks `native/android/app/src/main/assets/models/SHA256SUMS`.

## Class → NutriSole food mapping

| NutriSole food | YOLOv8m classes | YOLO11n classes |
|---|---|---|
| Apple | `apple` (1) | `apple` (0) |
| Banana | `banana` (6) | `banana` (2) |
| Tomato | `Tomato` (35), `tomato` (59) | — |

Every other class is shown with its readable name (first alias, e.g. "bell pepper/capsicum" → "Bell pepper") and the
result screen says it is not in NutriSole's food library yet, without nutrition or price.

## Code

- `native/extcv/models.ts` model registry, `classes.ts` mapping, `geometry.ts` letterbox/rotation/preview mapping,
  `decode.ts` YOLO decode + NMS, `pipeline.ts` thresholds and result state — all pure TypeScript.
- `native/extcv/detector.ts` ONNX Runtime session; `ExtCvScan.tsx` camera, overlay, HUD, picker, result and licences.
- `native/android/app/src/main/java/.../ExtCvModelFiles.kt` copies a model asset to app storage (ONNX Runtime needs a
  file path) and returns its SHA-256.
- `native/scripts/ort-gradle9-fix.cjs` (postinstall) rewrites one Gradle-9-incompatible version check in
  onnxruntime-react-native's build script.
- Tests: `native/tests/extcv.test.ts` (class mapping, decode/NMS, letterbox, orientation, preview mapping, states).

## Build

`.github/workflows/extcv-android-apk.yml` builds a release-signed QA APK (debug keystore, Hermes bundle embedded, no
Metro) for arm64-v8a, verifies the models and bundle inside the APK and uploads it as an artifact.

## Licences

See `EXTCV_NOTICES.md`. In short: both models were trained with Ultralytics YOLO (AGPL-3.0, `LICENSES/AGPL-3.0.txt`);
distributing this app requires offering its full source under the AGPL or an Ultralytics Enterprise licence. The
YOLOv8m repository is MIT (`LICENSES/henningheyen-MIT.txt`); its LVIS annotations are CC BY 4.0 and the COCO images
keep their own licences. The YOLO11n repository has **no licence**, so its weights are used only for this private
research demo.

## Limitations

- Sandbox CPU latency (x86, not a phone): YOLOv8m ≈ 638 ms, YOLO11n ≈ 77 ms per 640×640 inference. Phone numbers come
  from the HUD.
- Thresholds (25 % draw, 50 % confident, 10-point ambiguity margin) are untuned heuristics.
- Box placement assumes the preview shows the same field of view as the analysed frames (preview "cover").
- Only Apple, Banana and Tomato have nutrition data in the app; prices are not verified anywhere in the app.
- No freshness or quality grading.
