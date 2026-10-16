// Camera start-up state for the External-CV scan screen. Pure (no React Native), unit-tested.
//
// Start-up is staged so the preview never depends on frame processing:
//   1. permission  -> the Camera is only mounted once access is authorised (denied/restricted get their own UI)
//   2. preview     -> the session is configured with the preview alone; we wait for the first preview frame
//   3. frames      -> the frame output is attached, trying FRAME_LADDER in order. A format whose session errors or that
//                     delivers no frame within the timeout is abandoned for the next one; 'off' keeps the preview with
//                     detection disabled and the reason on screen.
// Every stage has a timeout so the screen reports what is stuck instead of spinning.

/** Mirrors VisionCamera 5's PermissionStatus. */
export type CamPermission = 'not-determined' | 'authorized' | 'denied' | 'restricted';

export type PermissionGate =
  | { kind: 'granted' }
  | { kind: 'request'; message: string }
  | { kind: 'settings'; message: string }
  | { kind: 'restricted'; message: string };

export function permissionGate(status: CamPermission | string | undefined, canRequest: boolean): PermissionGate {
  if (status === 'authorized') return { kind: 'granted' };
  if (status === 'restricted') return { kind: 'restricted', message: 'Camera access is restricted on this phone (device policy).' };
  if (status === 'not-determined' && canRequest) return { kind: 'request', message: 'Camera permission is needed to scan.' };
  if (status === 'denied') return { kind: 'settings', message: 'Camera access was denied. Allow it in Settings → Apps → NutriSole → Permissions.' };
  return { kind: 'settings', message: `Camera permission is ${status ?? 'unknown'}. Allow it in Settings.` };
}

/**
 * Frame-output pixel formats, most compatible first. 'yuv' is CameraX ImageAnalysis YUV_420_888, which every Android
 * camera must support alongside a preview stream; 'rgb' is RGBA_8888 converted by CameraX. The GPU-only 'native'
 * (ImageFormat.PRIVATE) analysis stream is deliberately not used: it is optional for the camera HAL and is the suspected
 * cause of the black preview on Xiaomi (the whole session fails to bind when the stream combination is rejected).
 */
export const FRAME_LADDER = ['yuv', 'rgb', 'off'] as const;
export type FrameFormat = (typeof FRAME_LADDER)[number];

export const PREVIEW_TIMEOUT_MS = 8000;
export const FIRST_FRAME_TIMEOUT_MS = 8000;

export type CamTimes = {
  /** When the Camera was asked to run (isActive true, permission + device present). */
  activeAt: number | null;
  configuredAt: number | null;
  startedAt: number | null;
  previewAt: number | null;
  /** When the current frame format was attached (set when the preview has started). */
  framesAttachedAt: number | null;
  firstFrameAt: number | null;
};

export const NO_TIMES: CamTimes = { activeAt: null, configuredAt: null, startedAt: null, previewAt: null, framesAttachedAt: null, firstFrameAt: null };

export type CamPhase =
  | { kind: 'idle' }                                   // not active (paused, backgrounded, no permission/device)
  | { kind: 'starting'; stage: string }                // waiting for configure/start/preview, within timeout
  | { kind: 'preview-timeout'; message: string }       // no preview within PREVIEW_TIMEOUT_MS
  | { kind: 'frames-wait' }                            // preview on, frame output attached, waiting for frame 1
  | { kind: 'frames-timeout' }                         // preview on, no frame within FIRST_FRAME_TIMEOUT_MS -> next format
  | { kind: 'running' }                                // preview on and frames arriving
  | { kind: 'preview-only'; message: string };         // preview on, every frame format failed

/** Where the camera start-up is, given the timestamps the Camera callbacks recorded. */
export function cameraPhase(t: CamTimes, format: FrameFormat, now: number, lastError: string): CamPhase {
  if (t.activeAt == null) return { kind: 'idle' };
  if (t.previewAt == null) {
    const stage = t.startedAt != null ? 'session started, waiting for first preview frame'
      : t.configuredAt != null ? 'session configured, starting'
      : 'configuring camera session';
    if (now - t.activeAt >= PREVIEW_TIMEOUT_MS) {
      const secs = Math.round(PREVIEW_TIMEOUT_MS / 1000);
      return { kind: 'preview-timeout', message: `Camera preview did not start within ${secs} s (stuck at: ${stage})${lastError ? ` · error: ${lastError}` : ' · no error reported by the camera'}` };
    }
    return { kind: 'starting', stage };
  }
  if (format === 'off') return { kind: 'preview-only', message: `Live detection is off: no frame format worked on this phone${lastError ? ` (${lastError})` : ''}. The preview still works.` };
  if (t.firstFrameAt != null && t.framesAttachedAt != null && t.firstFrameAt >= t.framesAttachedAt) return { kind: 'running' };
  if (t.framesAttachedAt != null && now - t.framesAttachedAt >= FIRST_FRAME_TIMEOUT_MS) return { kind: 'frames-timeout' };
  return { kind: 'frames-wait' };
}

/** The next frame format after `format` failed (or 'off' when the ladder is exhausted). */
export function nextFormat(format: FrameFormat): FrameFormat {
  const i = FRAME_LADDER.indexOf(format);
  return FRAME_LADDER[Math.min(i + 1, FRAME_LADDER.length - 1)];
}

const ms = (from: number | null, to: number | null) => (from == null || to == null ? '-' : `${Math.max(0, Math.round(to - from))}`);

/** One-line camera diagnostic for the HUD, so a phone screenshot tells us where start-up is. */
export function cameraDiagLine(p: { permission: string; deviceId: string | null; format: FrameFormat; times: CamTimes; frameInfo: string }): string {
  const t = p.times;
  return [
    `perm ${p.permission}`,
    `dev ${p.deviceId ?? 'none'}`,
    `cfg ${t.configuredAt != null ? 'y' : 'n'}`,
    `run ${t.startedAt != null ? 'y' : 'n'}`,
    `prev ${ms(t.activeAt, t.previewAt)} ms`,
    `fmt ${p.format}`,
    `1st frame ${ms(t.framesAttachedAt, t.firstFrameAt)} ms`,
    p.frameInfo ? p.frameInfo : '',
  ].filter(Boolean).join(' · ');
}

/** Error text from whatever VisionCamera / Nitro throws (Error, string, object with message). */
export function errorText(e: unknown): string {
  if (e == null) return 'unknown error';
  const m = typeof e === 'string' ? e : (e as { message?: unknown }).message;
  const s = typeof m === 'string' && m ? m : String(e);
  return s.length > 240 ? `${s.slice(0, 237)}…` : s;
}
