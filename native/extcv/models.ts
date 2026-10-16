// NutriSole External-CV: registry of the third-party detectors bundled in this build.
// Both are Ultralytics YOLO detectors exported to ONNX (opset 17, 640x640, NCHW float32 input "images",
// output [1, 4 + classes, 8400] = cx, cy, w, h in input pixels followed by per-class sigmoid scores; no NMS in graph).
// Weights are stored as FP16 initializers + Cast-to-FP32 nodes, so the files are half size but inference runs in FP32.
// The SHA-256 values below are verified by CI before the files are packed into the APK. The YOLO11n file is not in git
// (no upstream licence); CI downloads the original weights and re-exports them reproducibly (scripts/extcv/build_models.py).

export type ModelId = 'henning-yolov8m' | 'mark360-yolo11n';

export type ModelSpec = {
  id: ModelId;
  title: string;
  short: string;
  file: string;
  bytes: number;
  sha256: string;
  inputSize: number;
  names: readonly string[];
  source: string;
  weightsUrl: string;
  weightsSha256: string;
  licence: string;
};

export const HENNING_NAMES = ['almond', 'apple', 'apricot', 'artichoke', 'asparagus', 'avocado', 'banana', 'bean curd/tofu', 'bell pepper/capsicum', 'blackberry', 'blueberry', 'broccoli', 'brussels sprouts', 'cantaloup/cantaloupe', 'carrot', 'cauliflower', 'cayenne/cayenne spice/cayenne pepper/cayenne pepper spice/red pepper/red pepper', 'celery', 'cherry', 'chickpea/garbanzo', 'chili/chili vegetable/chili pepper/chili pepper vegetable/chilli/chilli vegetable/chilly/chilly', 'clementine', 'coconut/cocoanut', 'edible corn/corn/maize', 'cucumber/cuke', 'date/date fruit', 'eggplant/aubergine', 'fig/fig fruit', 'garlic/ail', 'ginger/gingerroot', 'Strawberry', 'gourd', 'grape', 'green bean', 'green onion/spring onion/scallion', 'Tomato', 'kiwi fruit', 'lemon', 'lettuce', 'lime', 'mandarin orange', 'melon', 'mushroom', 'onion', 'orange/orange fruit', 'papaya', 'pea/pea food', 'peach', 'pear', 'persimmon', 'pickle', 'pineapple', 'potato', 'prune', 'pumpkin', 'radish/daikon', 'raspberry', 'strawberry', 'sweet potato', 'tomato', 'turnip', 'watermelon', 'zucchini/courgette'] as const;

export const MARK360_NAMES = ['apple', 'mango', 'banana', 'orange'] as const;

export const MODELS: Record<ModelId, ModelSpec> = {
  'henning-yolov8m': {
    id: 'henning-yolov8m',
    title: 'YOLOv8m · 63 fruits & vegetables',
    short: 'YOLOv8m',
    file: 'henning_yolov8m_640_w16.onnx',
    bytes: 51953747,
    sha256: '698f7b1e64b2eac475f4d84d491d57e41f683bbcfc902c50f7761bbf970de37c',
    inputSize: 640,
    names: HENNING_NAMES,
    source: 'https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset',
    weightsUrl: 'https://drive.google.com/drive/folders/1I4mtQK11C3p41pO9raR0trgPVj0eQ2yb (yolo_fruits_and_vegetables_v1.pt)',
    weightsSha256: 'a18624d446e2fea08384949115de0b49446f0be2487fc954a37a9d962f71e5d3',
    licence: 'Repository MIT; dataset from LVIS (annotations CC BY 4.0, COCO images under their own Flickr licences); trained with Ultralytics YOLOv8 (AGPL-3.0).',
  },
  'mark360-yolo11n': {
    id: 'mark360-yolo11n',
    title: 'YOLO11n · 4 fruits (fast)',
    short: 'YOLO11n',
    file: 'mark360_yolo11n_640_w16.onnx',
    bytes: 5377616,
    sha256: 'c8f468937bf146483dd70e3e6c106e4ad789a01a2a94e6fa9d27043157bfa426',
    inputSize: 640,
    names: MARK360_NAMES,
    source: 'https://github.com/mark-360/Fruit-Detection',
    weightsUrl: 'https://github.com/mark-360/Fruit-Detection/blob/main/Models/Yolo11.pt',
    weightsSha256: '4f663749fe9192b60a34d2a71fa26d9c9f17c7ad997a557c50879c095456ed4e',
    licence: 'No licence file in the repository (all rights reserved by default); trained with Ultralytics YOLO11 (AGPL-3.0). Not committed to git: fetched from the author at build time for this experimental QA build only.',
  },
};

export const DEFAULT_MODEL: ModelId = 'henning-yolov8m';
export const MODEL_ORDER: ModelId[] = ['henning-yolov8m', 'mark360-yolo11n'];
