// Decoding of Ultralytics YOLOv8 / YOLO11 detection heads exported to ONNX without NMS.
// Output layout: [1, 4 + numClasses, numAnchors], row-major, i.e. value(channel c, anchor i) = out[c * numAnchors + i].
// Channels 0..3 are cx, cy, w, h in model-input pixels; channels 4.. are per-class scores in [0, 1].

import { iou, type Box } from './geometry.ts';

export type RawDetection = Box & { classIndex: number; score: number };

export type DecodeOptions = {
  numClasses: number;
  numAnchors: number;
  scoreThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  /** Max candidates kept before NMS (highest scores first). */
  preNmsTopK?: number;
};

export const DEFAULT_DECODE: Omit<DecodeOptions, 'numClasses' | 'numAnchors'> = {
  scoreThreshold: 0.25,
  iouThreshold: 0.45,
  maxDetections: 30,
  preNmsTopK: 3000,
};

/** Best class per anchor above the threshold, as xyxy boxes in model-input pixels. */
export function decodeCandidates(out: ArrayLike<number>, o: Pick<DecodeOptions, 'numClasses' | 'numAnchors' | 'scoreThreshold'>): RawDetection[] {
  const { numClasses, numAnchors: n, scoreThreshold } = o;
  if (out.length < (4 + numClasses) * n) throw new Error(`YOLO output too short: ${out.length} < ${(4 + numClasses) * n}`);
  const res: RawDetection[] = [];
  for (let i = 0; i < n; i++) {
    let best = -1; let score = scoreThreshold;
    for (let c = 0; c < numClasses; c++) {
      const v = out[(4 + c) * n + i];
      if (v > score) { score = v; best = c; }
    }
    if (best < 0) continue;
    const cx = out[i]; const cy = out[n + i]; const w = out[2 * n + i]; const h = out[3 * n + i];
    res.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, classIndex: best, score });
  }
  return res;
}

/** Greedy per-class NMS (Ultralytics' default, agnostic=false). Input order does not matter. */
export function nms(dets: RawDetection[], iouThreshold: number, maxDetections: number, preNmsTopK = Infinity): RawDetection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score).slice(0, preNmsTopK);
  const keep: RawDetection[] = [];
  for (const d of sorted) {
    if (keep.length >= maxDetections) break;
    if (keep.some(k => k.classIndex === d.classIndex && iou(k, d) > iouThreshold)) continue;
    keep.push(d);
  }
  return keep;
}

export function decodeYolo(out: ArrayLike<number>, o: DecodeOptions): RawDetection[] {
  return nms(decodeCandidates(out, o), o.iouThreshold, o.maxDetections, o.preNmsTopK);
}
