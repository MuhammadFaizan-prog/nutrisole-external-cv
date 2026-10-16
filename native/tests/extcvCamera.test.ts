import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { cameraDiagLine, cameraPhase, errorText, FIRST_FRAME_TIMEOUT_MS, FRAME_LADDER, nextFormat, NO_TIMES, permissionGate, PREVIEW_TIMEOUT_MS, type CamTimes } from '../extcv/cameraState.ts';

const at = (p: Partial<CamTimes>): CamTimes => ({ ...NO_TIMES, ...p });

// ---------- permission ----------
test('permission gate: authorised mounts the camera; undetermined requests; denied and restricted never spin', () => {
  assert.equal(permissionGate('authorized', false).kind, 'granted');
  assert.equal(permissionGate('not-determined', true).kind, 'request');
  assert.equal(permissionGate('not-determined', false).kind, 'settings', 'cannot ask again -> Settings, not a request loop');
  assert.equal(permissionGate('denied', false).kind, 'settings');
  assert.equal(permissionGate('restricted', false).kind, 'restricted');
  assert.equal(permissionGate(undefined, false).kind, 'settings');
  for (const s of ['not-determined', 'denied', 'restricted']) {
    const g = permissionGate(s, s === 'not-determined');
    assert.ok('message' in g && g.message.length > 10, `${s} has a visible message`);
  }
});

// ---------- start-up phases ----------
test('idle until the camera is activated', () => {
  assert.equal(cameraPhase(NO_TIMES, 'yuv', 99_999, '').kind, 'idle');
});

test('starting reports the stuck stage, then times out with the stage and the camera error', () => {
  const t0 = 1_000;
  const p1 = cameraPhase(at({ activeAt: t0 }), 'yuv', t0 + 500, '');
  assert.deepEqual(p1, { kind: 'starting', stage: 'configuring camera session' });
  assert.equal((cameraPhase(at({ activeAt: t0, configuredAt: t0 + 100 }), 'yuv', t0 + 500, '') as { stage: string }).stage, 'session configured, starting');
  assert.equal((cameraPhase(at({ activeAt: t0, configuredAt: t0 + 100, startedAt: t0 + 200 }), 'yuv', t0 + 500, '') as { stage: string }).stage, 'session started, waiting for first preview frame');
  const late = cameraPhase(at({ activeAt: t0, configuredAt: t0 + 100 }), 'yuv', t0 + PREVIEW_TIMEOUT_MS, 'No supported surface combination');
  assert.equal(late.kind, 'preview-timeout');
  assert.match((late as { message: string }).message, /within 8 s/);
  assert.match((late as { message: string }).message, /session configured, starting/);
  assert.match((late as { message: string }).message, /No supported surface combination/);
  const silent = cameraPhase(at({ activeAt: t0 }), 'yuv', t0 + PREVIEW_TIMEOUT_MS + 1, '');
  assert.match((silent as { message: string }).message, /no error reported/);
});

test('after the preview starts: waiting for frames, running, frame timeout, preview-only', () => {
  const t0 = 1_000; const prev = t0 + 700; const att = prev + 50;
  assert.equal(cameraPhase(at({ activeAt: t0, previewAt: prev }), 'yuv', prev + 10, '').kind, 'frames-wait');
  assert.equal(cameraPhase(at({ activeAt: t0, previewAt: prev, framesAttachedAt: att }), 'yuv', att + 1000, '').kind, 'frames-wait');
  assert.equal(cameraPhase(at({ activeAt: t0, previewAt: prev, framesAttachedAt: att, firstFrameAt: att + 120 }), 'yuv', att + 60_000, '').kind, 'running');
  assert.equal(cameraPhase(at({ activeAt: t0, previewAt: prev, framesAttachedAt: att }), 'yuv', att + FIRST_FRAME_TIMEOUT_MS, '').kind, 'frames-timeout');
  // a first frame from a previous format (before the current attach) does not count
  assert.equal(cameraPhase(at({ activeAt: t0, previewAt: prev, framesAttachedAt: att, firstFrameAt: att - 1 }), 'rgb', att + 100, '').kind, 'frames-wait');
  // the preview never times out once it has started, even long after
  assert.notEqual(cameraPhase(at({ activeAt: t0, previewAt: prev }), 'yuv', t0 + 10 * PREVIEW_TIMEOUT_MS, '').kind, 'preview-timeout');
  const off = cameraPhase(at({ activeAt: t0, previewAt: prev }), 'off', prev + 5, 'rgb: boom');
  assert.equal(off.kind, 'preview-only');
  assert.match((off as { message: string }).message, /preview still works/);
});

test('frame-format ladder: CPU YUV first, no GPU-only PRIVATE stream, ends at preview-only', () => {
  assert.deepEqual([...FRAME_LADDER], ['yuv', 'rgb', 'off']);
  assert.ok(!(FRAME_LADDER as readonly string[]).includes('native'));
  assert.equal(nextFormat('yuv'), 'rgb');
  assert.equal(nextFormat('rgb'), 'off');
  assert.equal(nextFormat('off'), 'off');
});

test('HUD camera line names permission, device, stages and first-frame latency', () => {
  const line = cameraDiagLine({ permission: 'authorized', deviceId: '0', format: 'yuv', times: at({ activeAt: 0, configuredAt: 50, startedAt: 90, previewAt: 420, framesAttachedAt: 430, firstFrameAt: 530 }), frameInfo: '1280×720 yuv-420-8-bit-full' });
  assert.equal(line, 'perm authorized · dev 0 · cfg y · run y · prev 420 ms · fmt yuv · 1st frame 100 ms · 1280×720 yuv-420-8-bit-full');
  assert.equal(cameraDiagLine({ permission: 'denied', deviceId: null, format: 'yuv', times: NO_TIMES, frameInfo: '' }), 'perm denied · dev none · cfg n · run n · prev - ms · fmt yuv · 1st frame - ms');
});

test('errorText handles Error, string, objects and very long messages', () => {
  assert.equal(errorText(new Error('x')), 'x');
  assert.equal(errorText('y'), 'y');
  assert.equal(errorText(null), 'unknown error');
  assert.equal(errorText({ message: 'z' }), 'z');
  assert.ok(errorText('a'.repeat(1000)).length <= 240);
});

// ---------- wiring in the scan screen (source checks; the screen itself needs a device) ----------
test('scan screen: staged outputs, all camera callbacks wired, no forced GPU-only frame format', () => {
  const src = readFileSync(decodeURIComponent(new URL('../extcv/ExtCvScan.tsx', import.meta.url).pathname), 'utf8');
  assert.ok(!/pixelFormat:\s*'native'/.test(src), 'frame output must not request the PRIVATE (native) format');
  assert.ok(!/enablePreviewSizedOutputBuffers/.test(src));
  assert.match(src, /outputs=\{outputs\}/, 'Camera gets the staged outputs, not a fixed [frameOutput]');
  assert.match(src, /attachFrames \? \[frameOutput\] : \[\]/, 'frame output only after the preview streams');
  for (const cb of ['onError={onCamError}', 'onPreviewStarted={onPreviewStarted}', 'onConfigured={onConfigured}', 'onStarted={onStarted}', 'onInterruptionStarted={onInterruption}']) assert.ok(src.includes(cb), cb);
  assert.match(src, /Linking\.openSettings\(\)/, 'denied state offers Settings');
  assert.match(src, /Retry camera/, 'preview timeout offers Retry');
  assert.match(src, /cameraDiagLine\(/, 'HUD shows the camera diagnostic line');
  assert.match(src, /AppState\.addEventListener/, 'camera pauses in the background');
});
