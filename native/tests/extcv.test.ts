import assert from 'node:assert/strict';
import { test } from 'node:test';
import { APP_FOODS, appFoodFor, appFoodIndex, displayLabel } from '../extcv/classes.ts';
import { decodeCandidates, decodeYolo, nms } from '../extcv/decode.ts';
import { addQuarterTurns, frameToViewCover, iou, letterboxFor, rotatePlanarToUpright, toLetterbox, unletterbox, uprightSize } from '../extcv/geometry.ts';
import { MODELS, HENNING_NAMES, MARK360_NAMES } from '../extcv/models.ts';
import { interpret, RateMeter, CONFIDENT_MIN, DRAW_MIN } from '../extcv/pipeline.ts';
import { foodInfo } from '../../src/nutrisole/extensions/model.ts';

const close = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// ---------- class mapping ----------
test('every app food has a catalogue entry and is reachable from the 63-class model', () => {
  for (const food of APP_FOODS) assert.notEqual(foodInfo(food), undefined);
  const idx = appFoodIndex(HENNING_NAMES);
  assert.deepEqual(idx.Apple, [1]);
  assert.deepEqual(idx.Banana, [6]);
  assert.deepEqual(idx.Tomato, [35, 59], 'the model lists both "Tomato" and "tomato"; both map to Tomato');
  assert.equal(HENNING_NAMES.length, 63);
});

test('the fast 4-class model maps apple and banana; mango and orange are not in the app library', () => {
  assert.equal(MARK360_NAMES.length, 4);
  assert.deepEqual(MARK360_NAMES.map(appFoodFor), ['Apple', null, 'Banana', null]);
});

test('unsupported classes map to null and get readable labels', () => {
  assert.equal(appFoodFor('bell pepper/capsicum'), null);
  assert.equal(appFoodFor('Strawberry'), null);
  assert.equal(appFoodFor('pineapple'), null, 'substring match must not turn pineapple into apple');
  assert.equal(displayLabel('bell pepper/capsicum'), 'Bell pepper');
  assert.equal(displayLabel('orange/orange fruit'), 'Orange');
  assert.equal(displayLabel('Tomato'), 'Tomato');
  const mapped = HENNING_NAMES.filter(n => appFoodFor(n)).length;
  assert.equal(mapped, 4);
});

test('model registry is consistent', () => {
  for (const m of Object.values(MODELS)) {
    assert.match(m.sha256, /^[0-9a-f]{64}$/);
    assert.match(m.file, /\.onnx$/);
    assert.equal(m.inputSize, 640);
    assert.ok(m.bytes > 1e6 && m.bytes < 100e6, 'each file must stay under GitHub\'s 100 MB limit');
  }
});

// ---------- decode / NMS ----------
function tensor(numClasses: number, anchors: { cx: number; cy: number; w: number; h: number; scores: Record<number, number> }[]) {
  const n = anchors.length; const out = new Float32Array((4 + numClasses) * n);
  anchors.forEach((a, i) => { out[i] = a.cx; out[n + i] = a.cy; out[2 * n + i] = a.w; out[3 * n + i] = a.h; for (const [c, s] of Object.entries(a.scores)) out[(4 + Number(c)) * n + i] = s; });
  return out;
}

test('decode picks the best class per anchor and converts cxcywh to xyxy', () => {
  const out = tensor(3, [
    { cx: 100, cy: 200, w: 40, h: 60, scores: { 0: 0.3, 2: 0.9 } },
    { cx: 10, cy: 10, w: 4, h: 4, scores: { 1: 0.1 } },
  ]);
  const d = decodeCandidates(out, { numClasses: 3, numAnchors: 2, scoreThreshold: 0.25 });
  assert.equal(d.length, 1);
  assert.equal(d[0].classIndex, 2); close(d[0].score, 0.9);
  assert.deepEqual([d[0].x1, d[0].y1, d[0].x2, d[0].y2], [80, 170, 120, 230]);
});

test('NMS suppresses same-class overlaps but keeps other classes and separate objects', () => {
  const out = tensor(2, [
    { cx: 100, cy: 100, w: 50, h: 50, scores: { 0: 0.8 } },
    { cx: 102, cy: 101, w: 50, h: 50, scores: { 0: 0.7 } },  // duplicate -> removed
    { cx: 101, cy: 100, w: 50, h: 50, scores: { 1: 0.6 } },  // other class, same place -> kept
    { cx: 300, cy: 300, w: 50, h: 50, scores: { 0: 0.5 } },  // separate apple -> kept
  ]);
  const d = decodeYolo(out, { numClasses: 2, numAnchors: 4, scoreThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10 });
  assert.deepEqual(d.map(x => [x.classIndex, x.score.toFixed(1)]), [[0, '0.8'], [1, '0.6'], [0, '0.5']]);
  assert.equal(nms(d, 0.45, 2).length, 2, 'maxDetections caps the result');
});

test('decode rejects a tensor that does not match the class count', () => {
  assert.throws(() => decodeCandidates(new Float32Array(10), { numClasses: 63, numAnchors: 8400, scoreThreshold: 0.25 }));
});

// ---------- letterbox / orientation / preview mapping ----------
test('letterbox round-trips a portrait 720x1280 frame into 640x640 and back', () => {
  const lb = letterboxFor(720, 1280, 640);
  close(lb.scale, 0.5); close(lb.padX, 140); close(lb.padY, 0);
  const b = { x1: 100, y1: 300, x2: 500, y2: 900 };
  const there = toLetterbox(b, lb);
  assert.deepEqual(there, { x1: 190, y1: 150, x2: 390, y2: 450 });
  assert.deepEqual(unletterbox(there, lb), b);
  const clamped = unletterbox({ x1: 0, y1: -5, x2: 640, y2: 700 }, lb);
  assert.deepEqual(clamped, { x1: 0, y1: 0, x2: 720, y2: 1280 }, 'boxes reaching into the padding are clamped to the frame');
});

test('landscape buffer with orientation right/left is treated as portrait upright', () => {
  assert.deepEqual(uprightSize(1280, 720, 'right'), { width: 720, height: 1280 });
  assert.deepEqual(uprightSize(1280, 720, 'up'), { width: 1280, height: 720 });
  assert.equal(addQuarterTurns('right', 1), 'down');
  assert.equal(addQuarterTurns('left', 1), 'up');
  assert.equal(addQuarterTurns('up', -1), 'left');
});

test('planar rotation undoes the frame orientation', () => {
  // Upright 3x3 single-channel image with a marker on the top edge (row 0, col 1).
  const S = 3; const upright = new Float32Array(S * S); upright[0 * S + 1] = 1;
  const rotCW = (a: Float32Array) => { const o = new Float32Array(a.length); for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) o[x * S + (S - 1 - y)] = a[y * S + x]; return o; };
  const right = rotCW(upright);            // data rotated +90 deg: top edge is now the right edge
  assert.equal(right[1 * S + 2], 1);
  assert.deepEqual([...rotatePlanarToUpright(right, S, 1, 'right')], [...upright]);
  const down = rotCW(right); assert.deepEqual([...rotatePlanarToUpright(down, S, 1, 'down')], [...upright]);
  const left = rotCW(down); assert.deepEqual([...rotatePlanarToUpright(left, S, 1, 'left')], [...upright]);
  assert.equal(rotatePlanarToUpright(upright, S, 1, 'up'), upright);
  // Channels are rotated independently.
  const two = new Float32Array(2 * S * S); two.set(right, 0); two.set(right, S * S);
  const back = rotatePlanarToUpright(two, S, 2, 'right');
  assert.deepEqual([...back.subarray(S * S)], [...upright]);
});

test('cover mapping onto a preview with a different aspect ratio', () => {
  // 720x1280 frame shown in a 360x800 view: scale = max(0.5, 0.625) = 0.625, x overflow (450-360)/2 = 45 cropped.
  const v = frameToViewCover({ x1: 72, y1: 0, x2: 648, y2: 1280 }, 720, 1280, 360, 800);
  close(v.x1, 0); close(v.x2, 360); close(v.y1, 0); close(v.y2, 800);
});

test('iou basics', () => {
  close(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 }), 50 / 150);
  assert.equal(iou({ x1: 0, y1: 0, x2: 1, y2: 1 }, { x1: 2, y1: 2, x2: 3, y2: 3 }), 0);
});

// ---------- end-to-end interpretation ----------
test('interpret maps a synthetic YOLOv8m output to an Apple detection in frame pixels', () => {
  const spec = MODELS['henning-yolov8m']; const n = 4;
  const out = new Float32Array((4 + 63) * n);
  const put = (i: number, cx: number, cy: number, w: number, h: number, cls: number, s: number) => { out[i] = cx; out[n + i] = cy; out[2 * n + i] = w; out[3 * n + i] = h; out[(4 + cls) * n + i] = s; };
  put(0, 320, 320, 200, 200, 1, 0.92);   // apple, centre of a portrait frame
  put(1, 500, 100, 60, 60, 30, 0.31);    // "Strawberry", low confidence
  put(2, 100, 100, 10, 10, 6, 0.05);     // below DRAW_MIN
  const r = interpret(out, spec, 1280, 720, 'right');
  assert.equal(r.frameWidth, 720); assert.equal(r.frameHeight, 1280);
  assert.equal(r.detections.length, 2);
  assert.equal(r.top?.food, 'Apple'); assert.equal(r.top?.label, 'Apple'); assert.equal(r.top?.confidence, 'confident');
  assert.equal(r.state, 'confident');
  // 640 -> 720x1280: scale 0.5, padX 140 -> box (220..420) maps to (160..560) x, (440..840) y.
  const b = r.top!.box; close(b.x1, 160); close(b.x2, 560); close(b.y1, 440); close(b.y2, 840);
  const low = r.detections[1]; assert.equal(low.confidence, 'low'); assert.equal(low.food, null); assert.equal(low.label, 'Strawberry');
  assert.ok(DRAW_MIN < CONFIDENT_MIN);
});

test('interpret reports low confidence for a weak or ambiguous top result and none for an empty frame', () => {
  const spec = MODELS['mark360-yolo11n']; const n = 2;
  const out = new Float32Array(8 * n);
  out[0] = 320; out[n] = 320; out[2 * n] = 100; out[3 * n] = 100; out[4 * n + 0] = 0.4;  // apple 0.40
  assert.equal(interpret(out, spec, 640, 640, 'up').state, 'low');
  out[4 * n] = 0.8; out[1] = 330; out[n + 1] = 320; out[2 * n + 1] = 100; out[3 * n + 1] = 100; out[5 * n + 1] = 0.75; // mango 0.75 on the same spot
  assert.equal(interpret(out, spec, 640, 640, 'up').state, 'low', 'apple vs mango within 0.1 on the same object is not a confident answer');
  assert.equal(interpret(new Float32Array(8 * n), spec, 640, 640, 'up').state, 'none');
});

test('rate meter', () => {
  const m = new RateMeter(3000); m.tick(0); m.tick(500); close(m.tick(1000), 2);
});
