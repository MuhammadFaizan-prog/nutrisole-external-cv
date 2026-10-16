// Pure geometry helpers for the live scan: letterbox, frame orientation and preview mapping.
// No React Native imports, so the unit tests run under plain Node.

export type Box = { x1: number; y1: number; x2: number; y2: number };
export type Orientation = 'up' | 'right' | 'down' | 'left';
export type Letterbox = { scale: number; padX: number; padY: number; srcW: number; srcH: number; size: number };

/** Aspect-preserving fit of srcW x srcH into a size x size square, content centred (the resizer's "contain" mode). */
export function letterboxFor(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH);
  return { scale, padX: (size - srcW * scale) / 2, padY: (size - srcH * scale) / 2, srcW, srcH, size };
}

/** Model-input box (square, letterboxed) -> source image pixels, clamped to the image. */
export function unletterbox(b: Box, lb: Letterbox): Box {
  const fx = (v: number) => Math.min(lb.srcW, Math.max(0, (v - lb.padX) / lb.scale));
  const fy = (v: number) => Math.min(lb.srcH, Math.max(0, (v - lb.padY) / lb.scale));
  return { x1: fx(b.x1), y1: fy(b.y1), x2: fx(b.x2), y2: fy(b.y2) };
}

/** Source image pixels -> model-input box (inverse of unletterbox, unclamped). */
export function toLetterbox(b: Box, lb: Letterbox): Box {
  return { x1: b.x1 * lb.scale + lb.padX, y1: b.y1 * lb.scale + lb.padY, x2: b.x2 * lb.scale + lb.padX, y2: b.y2 * lb.scale + lb.padY };
}

const ORDER: Orientation[] = ['up', 'right', 'down', 'left'];
/** Compose a frame orientation with a manual quarter-turn correction (debug setting, 0..3). */
export function addQuarterTurns(o: Orientation, turns: number): Orientation {
  return ORDER[(ORDER.indexOf(o) + ((turns % 4) + 4)) % 4];
}

/** Upright size of a buffer whose pixel data is rotated by `o` (VisionCamera Frame.orientation semantics). */
export function uprightSize(w: number, h: number, o: Orientation): { width: number; height: number } {
  return o === 'left' || o === 'right' ? { width: h, height: w } : { width: w, height: h };
}

/**
 * Rotate a square planar (CHW) tensor so it is upright.
 * `o` follows VisionCamera's Frame.orientation: 'right' = pixel data is rotated +90 deg (clockwise) relative to upright,
 * so it is undone by a 90 deg counter-clockwise turn; 'left' by a clockwise turn; 'down' by 180 deg.
 * Returns the input unchanged for 'up'.
 */
export function rotatePlanarToUpright(src: Float32Array, size: number, channels: number, o: Orientation, out?: Float32Array): Float32Array {
  if (o === 'up') return src;
  const dst = out && out.length === src.length ? out : new Float32Array(src.length);
  const plane = size * size; const last = size - 1;
  for (let ch = 0; ch < channels; ch++) {
    const base = ch * plane;
    for (let r = 0; r < size; r++) {
      const row = base + r * size;
      for (let c = 0; c < size; c++) {
        let sr: number; let sc: number;
        if (o === 'right') { sr = c; sc = last - r; }        // CCW: out[r][c] = in[c][S-1-r]
        else if (o === 'left') { sr = last - c; sc = r; }    // CW:  out[r][c] = in[S-1-c][r]
        else { sr = last - r; sc = last - c; }              // 180
        dst[row + c] = src[base + sr * size + sc];
      }
    }
  }
  return dst;
}

/**
 * Map a box in upright frame pixels onto a preview view that shows the same frame scaled with "cover"
 * (fills the view, centred, overflow cropped). Returns view coordinates (dp).
 */
export function frameToViewCover(b: Box, frameW: number, frameH: number, viewW: number, viewH: number): Box {
  const s = Math.max(viewW / frameW, viewH / frameH);
  const ox = (viewW - frameW * s) / 2; const oy = (viewH - frameH * s) / 2;
  return { x1: b.x1 * s + ox, y1: b.y1 * s + oy, x2: b.x2 * s + ox, y2: b.y2 * s + oy };
}

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}
