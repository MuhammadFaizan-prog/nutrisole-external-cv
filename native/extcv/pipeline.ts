// From raw ONNX output to what the scan screen shows. Pure (no React Native), unit-tested.

import { appFoodFor, displayLabel, type AppFood } from './classes.ts';
import { decodeYolo, DEFAULT_DECODE } from './decode.ts';
import { letterboxFor, unletterbox, uprightSize, type Box, type Orientation } from './geometry.ts';
import type { ModelSpec } from './models.ts';

/** Boxes below DRAW_MIN are dropped; DRAW_MIN..CONFIDENT_MIN are shown as "low confidence"; untuned heuristics. */
export const DRAW_MIN = 0.25;
export const CONFIDENT_MIN = 0.5;
/** A confident top result is still flagged when the runner-up box of another class is this close in score. */
export const AMBIGUITY_MARGIN = 0.1;

export type Confidence = 'confident' | 'low';

export type Detection = {
  id: string;
  rawLabel: string;
  label: string;
  food: AppFood | null;
  score: number;
  confidence: Confidence;
  /** Upright frame pixels. */
  box: Box;
};

export type ScanResult = {
  detections: Detection[];
  frameWidth: number;
  frameHeight: number;
  state: 'none' | 'low' | 'confident';
  top: Detection | null;
};

export function confidenceOf(score: number): Confidence {
  return score >= CONFIDENT_MIN ? 'confident' : 'low';
}

export function interpret(out: ArrayLike<number>, spec: ModelSpec, bufferW: number, bufferH: number, orientation: Orientation, maxDetections = DEFAULT_DECODE.maxDetections): ScanResult {
  const numClasses = spec.names.length;
  const numAnchors = out.length / (4 + numClasses);
  if (!Number.isInteger(numAnchors)) throw new Error(`Unexpected output size ${out.length} for ${numClasses} classes`);
  const raw = decodeYolo(out, { ...DEFAULT_DECODE, maxDetections, scoreThreshold: DRAW_MIN, numClasses, numAnchors });
  const { width, height } = uprightSize(bufferW, bufferH, orientation);
  const lb = letterboxFor(width, height, spec.inputSize);
  const detections = raw.map((d, i): Detection => {
    const rawLabel = spec.names[d.classIndex] ?? `class ${d.classIndex}`;
    return { id: `${i}-${d.classIndex}`, rawLabel, label: displayLabel(rawLabel), food: appFoodFor(rawLabel), score: d.score, confidence: confidenceOf(d.score), box: unletterbox(d, lb) };
  }).filter(d => d.box.x2 - d.box.x1 >= 1 && d.box.y2 - d.box.y1 >= 1);
  const top = detections[0] ?? null;
  let state: ScanResult['state'] = top ? top.confidence : 'none';
  if (top && state === 'confident') {
    const rival = detections.find(d => d.label !== top.label);
    if (rival && top.score - rival.score < AMBIGUITY_MARGIN && overlaps(top.box, rival.box)) state = 'low';
  }
  return { detections, frameWidth: width, frameHeight: height, state, top };
}

function overlaps(a: Box, b: Box) {
  return Math.min(a.x2, b.x2) > Math.max(a.x1, b.x1) && Math.min(a.y2, b.y2) > Math.max(a.y1, b.y1);
}

/** Rolling inference-rate meter: results per second over the last `windowMs`. */
export class RateMeter {
  private stamps: number[] = [];
  private windowMs: number;
  constructor(windowMs = 3000) { this.windowMs = windowMs; }
  tick(now: number): number {
    this.stamps.push(now);
    while (this.stamps.length && now - this.stamps[0] > this.windowMs) this.stamps.shift();
    if (this.stamps.length < 2) return 0;
    return (this.stamps.length - 1) * 1000 / (this.stamps[this.stamps.length - 1] - this.stamps[0]);
  }
}
