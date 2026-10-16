// NutriSole External-CV live scan: VisionCamera preview + frame output, GPU resizer to a 640x640 planar RGB float tensor,
// ONNX Runtime inference on the JS side with exactly one inference in flight (frames arriving meanwhile are dropped
// on the camera thread), boxes + labels over the preview, a perf HUD, a model picker, and a tap-through result screen.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AppState, BackHandler, Linking, Pressable, StyleSheet, View } from 'react-native';
import { Camera, CommonResolutions, useCameraDevice, useCameraPermission, useFrameOutput, type Frame } from 'react-native-vision-camera';
import { useResizer } from 'react-native-vision-camera-resizer';
import { createSynchronizable, scheduleOnRN } from 'react-native-worklets';
import { Button, Header, Panel, Pic, Screen, T, Tap, s } from '../responsive/components';
import { colors, foodInfo } from '../../src/nutrisole/extensions/model';
import type { AppRoute, Sheet } from '../../src/nutrisole/types';
import Icons, { type ExtIconName } from './icons';
import { loadModel, releaseModel, runModel, type LoadedModel } from './detector';
import { DEFAULT_MODEL, MODEL_ORDER, MODELS, type ModelId } from './models';
import { addQuarterTurns, frameToViewCover, rotatePlanarToUpright, type Orientation } from './geometry';
import { interpret, RateMeter, CONFIDENT_MIN, type Detection, type ScanResult } from './pipeline';
import { cameraDiagLine, cameraPhase, errorText, FRAME_LADDER, nextFormat, NO_TIMES, permissionGate, type CamTimes, type FrameFormat } from './cameraState';
import type { AppFood } from './classes';

export type ExtCvProps = { go: (r: AppRoute) => void; back: () => void; open: (sheet: Sheet) => void; chooseFood: (food: string) => void };

type Perf = { inferMs: number; totalMs: number; fps: number; orientation: string; frame: string; dropped: number; skipped: number };
type View3 = { kind: 'camera' } | { kind: 'result'; detection: Detection | null; modelId: ModelId } | { kind: 'about' };

const SIZE = 640;
const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function ExtCvScan(props: ExtCvProps) {
  const [view, setView] = useState<View3>({ kind: 'camera' });
  const [modelId, setModelId] = useState<ModelId>(DEFAULT_MODEL);
  const [loaded, setLoaded] = useState<LoadedModel | null>(null);
  const [modelError, setModelError] = useState('');
  const modelRef = useRef<LoadedModel | null>(null);

  // Load / switch model. Only one session is kept in memory.
  useEffect(() => {
    let cancelled = false;
    setLoaded(null); setModelError('');
    const previous = modelRef.current; modelRef.current = null;
    (async () => {
      await releaseModel(previous);
      try {
        const m = await loadModel(modelId);
        if (cancelled) { await releaseModel(m); return; }
        modelRef.current = m; setLoaded(m);
      } catch (e) { if (!cancelled) setModelError(String((e as Error)?.message ?? e)); }
    })();
    return () => { cancelled = true; };
  }, [modelId]);
  useEffect(() => () => { releaseModel(modelRef.current); modelRef.current = null; }, []);

  // Hardware back inside the scan flow returns to the camera first.
  useEffect(() => {
    if (view.kind === 'camera') return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => { setView({ kind: 'camera' }); return true; });
    return () => sub.remove();
  }, [view.kind]);

  if (view.kind === 'result') return <ExtCvResult {...props} detection={view.detection} modelId={view.modelId} close={() => setView({ kind: 'camera' })} />;
  if (view.kind === 'about') return <ExtCvAbout loaded={loaded} close={() => setView({ kind: 'camera' })} />;
  return <LiveCamera {...props} modelId={modelId} setModelId={setModelId} loaded={loaded} modelError={modelError}
    openResult={d => setView({ kind: 'result', detection: d, modelId })} openAbout={() => setView({ kind: 'about' })} />;
}

function LiveCamera({ back, open, modelId, setModelId, loaded, modelError, openResult, openAbout }: ExtCvProps & {
  modelId: ModelId; setModelId: (m: ModelId) => void; loaded: LoadedModel | null; modelError: string;
  openResult: (d: Detection | null) => void; openAbout: () => void;
}) {
  const { status: permStatus, requestPermission, canRequestPermission } = useCameraPermission();
  const gate = permissionGate(permStatus, canRequestPermission);
  const hasPermission = gate.kind === 'granted';
  const device = useCameraDevice('back');
  // Camera start-up diagnostics (see cameraState.ts): timestamps from the Camera callbacks, the frame format being tried,
  // the last camera error, and a remount key for Retry.
  const [format, setFormat] = useState<FrameFormat>(FRAME_LADDER[0]);
  const [times, setTimes] = useState<CamTimes>(NO_TIMES);
  const [camError, setCamError] = useState('');
  const [frameInfo, setFrameInfo] = useState('');
  const [camKey, setCamKey] = useState(0);
  const [now, setNow] = useState(() => Date.now());
  const [appActive, setAppActive] = useState(AppState.currentState !== 'background');
  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const [result, setResult] = useState<ScanResult | null>(null);
  const [perf, setPerf] = useState<Perf>({ inferMs: 0, totalMs: 0, fps: 0, orientation: '-', frame: '-', dropped: 0, skipped: 0 });
  const [hud, setHud] = useState(true);
  const [paused, setPaused] = useState(false);
  const [turns, setTurns] = useState(0);
  const [runError, setRunError] = useState('');
  const meter = useMemo(() => new RateMeter(3000), []);
  const dropped = useRef(0);
  const scratch = useRef<Float32Array | null>(null);
  const loadedRef = useRef(loaded); loadedRef.current = loaded;
  const turnsRef = useRef(turns); turnsRef.current = turns;

  // Ask for CAMERA once when the status is still undetermined; a denial switches to the Settings UI (no request loop).
  const asked = useRef(false);
  useEffect(() => {
    if (gate.kind !== 'request' || asked.current) return;
    asked.current = true;
    requestPermission().catch(e => setCamError(`permission request failed: ${errorText(e)}`));
  }, [gate.kind, requestPermission]);
  useEffect(() => { const sub = AppState.addEventListener('change', st => setAppActive(st === 'active')); return () => sub.remove(); }, []);

  const resizer = useResizer({ width: SIZE, height: SIZE, channelOrder: 'rgb', dataType: 'float32', scaleMode: 'contain', pixelLayout: 'planar' });
  // true while a frame is being converted or an inference is running; read on the camera thread, cleared on the JS thread.
  const busy = useMemo(() => createSynchronizable(false), []);
  // frames skipped on the camera thread because an inference was still in flight
  const skipped = useMemo(() => createSynchronizable(0), []);

  const onTensor = useCallback(async (buffer: ArrayBuffer, width: number, height: number, orientation: string, capturedAt: number) => {
    const m = loadedRef.current;
    try {
      if (!m) return;
      const o = addQuarterTurns(orientation as Orientation, turnsRef.current);
      let input: Float32Array = new Float32Array(buffer);
      if (o !== 'up') { input = rotatePlanarToUpright(input, SIZE, 3, o, scratch.current ?? undefined); scratch.current = input; }
      const { output, ms } = await runModel(m, input);
      if (loadedRef.current !== m) return; // model switched while running
      const r = interpret(output, MODELS[m.id], width, height, o);
      const now = Date.now();
      setResult(r);
      setPerf({ inferMs: ms, totalMs: now - capturedAt, fps: meter.tick(now), orientation: o === orientation ? orientation : `${orientation}→${o}`, frame: `${width}×${height}`, dropped: dropped.current, skipped: skipped.getDirty() });
      setRunError('');
    } catch (e) {
      setRunError(String((e as Error)?.message ?? e));
    } finally {
      busy.setBlocking(false);
    }
  }, [busy, skipped, meter]);
  const onFrameError = useCallback((message: string) => setRunError(message), []);
  const countDrop = useCallback(() => { dropped.current += 1; }, []);

  const active = hasPermission && !!device && !paused && appActive;
  // Start-up clock: restarts whenever the camera is (re)activated or remounted.
  useEffect(() => { setTimes(active ? { ...NO_TIMES, activeAt: Date.now() } : NO_TIMES); if (active) setNow(Date.now()); }, [active, camKey]);
  const phase = cameraPhase(times, format, now, camError);
  // Re-evaluate timeouts once a second while something is still pending.
  const pending = phase.kind === 'starting' || phase.kind === 'frames-wait';
  useEffect(() => { if (!pending) return; const id = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(id); }, [pending]);
  const attachFrames = times.previewAt != null && format !== 'off';
  // Asked on the camera thread: report the next frame (first frame of the current format) back to JS.
  const needFirst = useMemo(() => createSynchronizable(false), []);
  useEffect(() => {
    if (!attachFrames) return;
    needFirst.setBlocking(true);
    const t = Date.now(); setTimes(x => ({ ...x, framesAttachedAt: t, firstFrameAt: null })); setNow(t);
  }, [attachFrames, format, needFirst]);
  // Abandon `failed` (only if it is still the current format, so a burst of errors skips one step, not several) and
  // restart the first-frame clock in the same render so the old timestamp cannot time out the next format too.
  const failFormat = useCallback((failed: FrameFormat, why: string) => {
    setCamError(why);
    setFormat(f => (f === failed ? nextFormat(f) : f));
    const t = Date.now(); setTimes(x => ({ ...x, framesAttachedAt: t, firstFrameAt: null }));
  }, []);
  useEffect(() => { if (phase.kind === 'frames-timeout') failFormat(format, `no frames within 8 s with ${format} frames`); }, [phase.kind, format, failFormat]);
  const onFirstFrame = useCallback((w: number, h: number, pixelFormat: string) => {
    setTimes(x => ({ ...x, firstFrameAt: Date.now() }));
    setFrameInfo(`${w}×${h} ${pixelFormat}`);
  }, []);
  const onCamError = useCallback((e: Error) => {
    const msg = errorText(e);
    // An error after the frame output was attached: drop that format and go on with the next one (preview stays).
    if (attachFrames) failFormat(format, `${format}: ${msg}`); else setCamError(msg);
  }, [attachFrames, format, failFormat]);
  const onPreviewStarted = useCallback(() => setTimes(x => (x.activeAt == null || x.previewAt != null ? x : { ...x, previewAt: Date.now() })), []);
  // A preview stop is only counted: attaching the frame output re-binds the session, which briefly stops the preview, and
  // treating that as "no preview" would detach the frames again in a loop. A real stall shows up as frames-timeout.
  const [previewStops, setPreviewStops] = useState(0);
  const onPreviewStopped = useCallback(() => setPreviewStops(n => n + 1), []);
  const onConfigured = useCallback(() => setTimes(x => (x.activeAt == null ? x : { ...x, configuredAt: x.configuredAt ?? Date.now() })), []);
  const onStarted = useCallback(() => setTimes(x => (x.activeAt == null ? x : { ...x, startedAt: x.startedAt ?? Date.now() })), []);
  const onInterruption = useCallback((reason: string) => setCamError(`camera interrupted: ${reason}`), []);
  const retryCamera = () => { setCamError(''); setFrameInfo(''); setFormat(FRAME_LADDER[0]); setCamKey(k => k + 1); };
  const ready = resizer.state === 'ready' && !!loaded;
  const r0 = resizer.resizer;

  const onFrame = useCallback((frame: Frame) => {
    'worklet';
    if (needFirst.getDirty()) { needFirst.setBlocking(false); scheduleOnRN(onFirstFrame, frame.width, frame.height, String(frame.pixelFormat)); }
    if (r0 == null || !ready) { frame.dispose(); return; }
    if (busy.getDirty()) { skipped.setBlocking(n => n + 1); frame.dispose(); return; }
    busy.setBlocking(true);
    try {
      const w = frame.width; const h = frame.height; const o = frame.orientation;
      const gpu = r0.resize(frame);
      const copy = gpu.getPixelBuffer().slice(0);
      gpu.dispose(); frame.dispose();
      scheduleOnRN(onTensor, copy, w, h, o, Date.now());
    } catch (e) {
      frame.dispose(); busy.setBlocking(false);
      scheduleOnRN(onFrameError, String(e));
    }
  }, [r0, ready, busy, skipped, onTensor, onFrameError, needFirst, onFirstFrame]);

  // CPU YUV (or RGBA) ImageAnalysis frames: always combinable with a preview stream. The resizer accepts them on Android.
  const frameOutput = useFrameOutput({
    targetResolution: CommonResolutions.HD_16_9,
    pixelFormat: format === 'rgb' ? 'rgb' : 'yuv',
    dropFramesWhileBusy: true,
    onFrame,
    onFrameDropped: countDrop,
  });
  // Preview first, frames only once the preview is streaming, so frame processing can never keep the preview black.
  const outputs = useMemo(() => (attachFrames ? [frameOutput] : []), [attachFrames, frameOutput]);

  const pickModel = () => open({
    title: 'Detection model',
    description: 'External open-source detectors, used as-is. Switching reloads the model.',
    choices: MODEL_ORDER.map(id => ({ label: `${MODELS[id].title}`, selected: id === modelId, action: () => { setResult(null); setModelId(id); } })),
  });
  const debugSheet = () => open({
    title: 'Scan debug',
    description: `Frame orientation reported by the camera: ${perf.orientation}. If boxes or labels look rotated, add a quarter-turn correction.`,
    choices: [0, 1, 2, 3].map(t => ({ label: t === 0 ? 'Rotation: automatic' : `Rotation: automatic + ${t * 90}°`, selected: turns === t, action: () => { setTurns(t); setResult(null); } }))
      .concat([{ label: hud ? 'Hide performance HUD' : 'Show performance HUD', selected: false, action: () => setHud(v => !v) }]),
  });

  const status = gate.kind !== 'granted' ? gate.message
    : !device ? 'No back camera found on this device.'
    : phase.kind === 'preview-timeout' ? phase.message
    : resizer.state === 'error' ? `Frame resizer unavailable on this phone (needs Android 9+ with Vulkan): ${resizer.error.message}`
    : modelError ? `Model failed to load: ${modelError}`
    : !loaded ? `Loading ${MODELS[modelId].short}…`
    : runError ? `Scan error: ${runError}` : '';

  const top = result?.top ?? null;
  const banner = !ready || !result ? null
    : result.state === 'none' ? { title: 'No food detected', sub: 'Point at one fruit or vegetable, close and well lit', tone: 'none' as const }
    : result.state === 'low' ? { title: 'Low confidence', sub: top ? `Maybe ${top.label} (${pct(top.score)}) · move closer or improve light` : 'Move closer or improve light', tone: 'low' as const }
    : { title: top!.label, sub: `Confidence ${pct(top!.score)} · ${top!.food ? 'Tap for nutrition' : 'Not in NutriSole’s food library'}`, tone: 'ok' as const };

  return <View testID="screen-scan" style={{ flex: 1, backgroundColor: '#111' }}>
    <View style={{ flex: 1 }} onLayout={e => { const { width, height } = e.nativeEvent.layout; setLayout(l => l.width === width && l.height === height ? l : { width, height }); }}>
      {hasPermission && device ? <Camera key={camKey} style={StyleSheet.absoluteFill} device={device} isActive={active} outputs={outputs} orientationSource="interface" resizeMode="cover"
          onConfigured={onConfigured} onStarted={onStarted} onError={onCamError} onPreviewStarted={onPreviewStarted} onPreviewStopped={onPreviewStopped} onInterruptionStarted={onInterruption} />
        : <View style={[StyleSheet.absoluteFill, { alignItems: 'center', justifyContent: 'center', padding: 30, gap: 16 }]}>
          <T size={17} color="#fff" style={s.center}>{status}</T>
          {gate.kind === 'request' && <Button label="Allow camera" text="Allow camera" primary onPress={() => { requestPermission().catch(e => setCamError(errorText(e))); }} />}
          {gate.kind === 'settings' && <Button label="Open settings" text="Open settings" primary onPress={() => Linking.openSettings()} />}
        </View>}
      {/* Detection overlay */}
      {result && layout.width > 0 && result.detections.map(d => {
        const b = frameToViewCover(d.box, result.frameWidth, result.frameHeight, layout.width, layout.height);
        const ok = d.confidence === 'confident'; const color = ok ? '#7ee08a' : '#f2c946';
        return <Pressable key={d.id} accessibilityRole="button" accessibilityLabel={`${ok ? d.label : 'Low confidence ' + d.label}, ${pct(d.score)}`} onPress={() => openResult(d)}
          style={{ position: 'absolute', left: b.x1, top: b.y1, width: Math.max(8, b.x2 - b.x1), height: Math.max(8, b.y2 - b.y1), borderWidth: 2.5, borderColor: color, borderRadius: 8 }}>
          <View style={{ position: 'absolute', left: -2, top: -24, backgroundColor: color, borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2, maxWidth: 220 }}>
            <T size={12} bold color="#10210f">{ok ? `${d.label} ${pct(d.score)}` : `? ${d.label} ${pct(d.score)}`}</T>
          </View>
        </Pressable>;
      })}
      {/* Top bar */}
      <View style={[s.row, { position: 'absolute', top: 10, left: 10, right: 10, justifyContent: 'space-between' }]}>
        <RoundButton label="Close camera" icon="X" onPress={back} />
        <Tap label="Choose detection model" onPress={pickModel} style={[s.row, { gap: 6, backgroundColor: '#000a', borderRadius: 22, paddingHorizontal: 14, minHeight: 44 }]}>
          <Icons name="Cpu" size={18} color="#fff" /><T size={15} bold color="#fff">{MODELS[modelId].short}</T><Icons name="ChevronDown" size={16} color="#fff" />
        </Tap>
        <View style={[s.row, { gap: 8 }]}>
          <RoundButton label="Scan debug settings" icon="Settings" onPress={debugSheet} />
          <RoundButton label="About models and licences" icon="Info" onPress={openAbout} />
        </View>
      </View>
      {/* Perf HUD */}
      {hud && <View pointerEvents="none" style={{ position: 'absolute', top: 64, left: 10, backgroundColor: '#000b', borderRadius: 10, padding: 8, gap: 2 }}>
        <T size={11} color="#9fe8a8" style={{ fontFamily: 'monospace' }}>{`${MODELS[modelId].short} @${SIZE} · CPU`}</T>
        <T size={11} color="#fff" style={{ fontFamily: 'monospace' }}>{`infer ${perf.inferMs} ms · e2e ${perf.totalMs} ms`}</T>
        <T size={11} color="#fff" style={{ fontFamily: 'monospace' }}>{`${perf.fps.toFixed(1)} inf/s · skipped ${perf.skipped} · cam-dropped ${perf.dropped}`}</T>
        <T size={11} color="#ccc" style={{ fontFamily: 'monospace' }}>{`frame ${perf.frame} ${perf.orientation}`}</T>
        {loaded && <T size={11} color={loaded.hashOk ? '#9fe8a8' : '#ff8a80'} style={{ fontFamily: 'monospace' }}>{`model ${loaded.hashOk ? 'sha256 ok' : 'SHA MISMATCH'} · load ${loaded.loadMs} ms`}</T>}
        <T size={11} color="#9ecbff" style={{ fontFamily: 'monospace', maxWidth: 330 }}>{`cam ${cameraDiagLine({ permission: String(permStatus), deviceId: device?.id ?? null, format, times, frameInfo })}`}</T>
        <T size={11} color="#9ecbff" style={{ fontFamily: 'monospace' }}>{`cam phase ${phase.kind}${phase.kind === 'starting' ? `: ${phase.stage}` : ''} · prev-stops ${previewStops} · resizer ${resizer.state}`}</T>
        {!!camError && <T size={11} color="#ff8a80" style={{ fontFamily: 'monospace', maxWidth: 330 }}>{`cam error ${camError}`}</T>}
      </View>}
      {!!status && hasPermission && device && <View style={{ position: 'absolute', left: 20, right: 20, top: '45%', backgroundColor: '#000c', borderRadius: 16, padding: 14, gap: 10 }}>
        <T size={15} color="#fff" style={s.center}>{status}</T>
        {phase.kind === 'preview-timeout' && <Button label="Retry camera" text="Retry camera" primary onPress={retryCamera} />}
      </View>}
      {phase.kind === 'preview-only' && <View style={{ position: 'absolute', left: 20, right: 20, top: '45%', backgroundColor: '#000c', borderRadius: 16, padding: 14, gap: 10 }}>
        <T size={15} color="#fff" style={s.center}>{phase.message}</T>
        <Button label="Retry live detection" text="Retry live detection" onPress={retryCamera} />
      </View>}
    </View>
    {/* Bottom: result banner + controls */}
    <View testID="footer-scan" style={{ backgroundColor: '#11110f', paddingHorizontal: 14, paddingTop: 12, paddingBottom: 16, gap: 12 }}>
      <Tap label="Open scan result" disabled={!banner || banner.tone === 'none'} onPress={() => openResult(top)}
        style={[s.row, { minHeight: 64, borderRadius: 20, paddingHorizontal: 16, paddingVertical: 10, backgroundColor: banner?.tone === 'ok' ? '#173e27' : banner?.tone === 'low' ? '#4a3a12' : '#262622' }]}>
        <Icons name={banner?.tone === 'ok' ? 'ScanSearch' : banner?.tone === 'low' ? 'CircleAlert' : 'Scan'} size={26} color="#fff" />
        <View style={[s.flex, { gap: 3 }]}>
          <T size={18} bold color="#fff">{banner?.title ?? (phase.kind === 'preview-timeout' ? 'Camera did not start' : phase.kind === 'preview-only' ? 'Preview only' : !ready ? 'Getting ready…' : phase.kind === 'running' ? 'Scanning…' : 'Starting camera…')}</T>
          <T size={13} color="#ddd8cf">{banner?.sub ?? (phase.kind === 'starting' ? `Camera: ${phase.stage}` : phase.kind === 'frames-wait' ? 'Preview on · waiting for the first camera frame' : phase.kind === 'preview-timeout' ? 'See the message above · tap Retry camera' : 'Detection runs automatically on live frames')}</T>
        </View>
        {banner && banner.tone !== 'none' && <Icons name="ChevronRight" size={20} color="#fff" />}
      </Tap>
      <View style={[s.row, { justifyContent: 'space-between' }]}>
        <T size={12} color="#aaa" style={s.flex}>Grading not available in this build · external models detect foods only</T>
        <Tap label={paused ? 'Resume live detection' : 'Pause live detection'} onPress={() => setPaused(p => !p)} style={[s.row, { gap: 6, minHeight: 40, paddingHorizontal: 12, borderRadius: 18, backgroundColor: '#2c2c28' }]}>
          <Icons name={paused ? 'Play' : 'Pause'} size={16} color="#fff" /><T size={14} color="#fff">{paused ? 'Resume' : 'Pause'}</T>
        </Tap>
      </View>
    </View>
  </View>;
}

function RoundButton({ label, icon, onPress }: { label: string; icon: ExtIconName; onPress: () => void }) {
  return <Tap label={label} onPress={onPress} style={{ width: 44, height: 44, borderRadius: 22, backgroundColor: '#000a', alignItems: 'center', justifyContent: 'center' }}><Icons name={icon} size={22} color="#fff" /></Tap>;
}

function ExtCvResult({ detection, modelId, close, go, chooseFood }: ExtCvProps & { detection: Detection | null; modelId: ModelId; close: () => void }) {
  const spec = MODELS[modelId];
  const low = !detection || detection.score < CONFIDENT_MIN;
  const food: AppFood | null = detection?.food ?? null;
  const info = food ? foodInfo(food) : null;
  const openWithFood = (route: AppRoute) => { if (!food) return; chooseFood(food); go(route); };
  return <Screen route="extcv-result" gutter={16} header={<Header title="Scan result" back={close} />} gap={14}>
    <Panel style={{ gap: 10 }}>
      <View style={s.row}>
        {info ? <Pic asset={info.asset} size={72} radius={14} /> : <View style={{ width: 72, height: 72, borderRadius: 14, backgroundColor: colors.pale, alignItems: 'center', justifyContent: 'center' }}><Icons name="ScanSearch" size={30} /></View>}
        <View style={[s.flex, { gap: 4 }]}>
          <T size={24} bold serif>{low ? 'Not sure' : detection!.label}</T>
          <T size={15} color={colors.muted}>{detection ? `${low ? 'Best guess ' + detection.label + ' · ' : ''}Confidence ${pct(detection.score)}` : 'Nothing detected'}</T>
          <T size={13} color={colors.muted}>{`Model: ${spec.title}`}</T>
        </View>
      </View>
    </Panel>
    {low && <Panel fill="#fff6dc" texture="" style={{ gap: 8 }}>
      <View style={[s.row, { gap: 8 }]}><Icons name="CircleAlert" size={22} color="#8a6100" /><T size={17} bold color="#5c4100">Low confidence</T></View>
      <T size={15} color="#5c4100">This result may be wrong, so NutriSole will not use it automatically. Move closer, keep one item in frame in good light, or choose the food yourself.</T>
      <Button label="Choose food manually" text="Choose food manually" onPress={() => go('food-selector')} />
    </Panel>}
    {!low && detection && !food && <Panel fill={colors.pale} texture="" style={{ gap: 6 }}>
      <T size={17} bold>{`${detection.label} isn’t in NutriSole’s food library yet`}</T>
      <T size={15} color={colors.muted}>The external model recognised it, but the app has no nutrition or price entry for it. Supported foods in this build: Apple, Banana, Tomato.</T>
      <Button label="Choose a supported food" text="Choose a supported food" onPress={() => go('food-selector')} />
    </Panel>}
    {!low && info && <Panel style={{ gap: 8 }}>
      <T size={18} bold>Nutrition · per 100 g</T>
      {[['Energy', `${info.calories} kcal`], ['Carbohydrate', `${info.carbs} g`], ['Protein', `${info.protein} g`], ['Fat', `${info.fat} g`], ['Fibre', `${info.fibre} g`]].map(([k, v]) =>
        <View key={k} style={[s.row, { justifyContent: 'space-between' }]}><T size={15} color={colors.muted}>{k}</T><T size={15} bold>{v}</T></View>)}
      <T size={13} color={colors.muted}>{`Typical portion ${info.grams} g · sample catalogue values (same as the rest of the app)`}</T>
      <Button label="Log this food" text="Log this food" primary onPress={() => openWithFood('portion-confirmation')} />
      <Button label="Nutrition details" text="Nutrition details" onPress={() => openWithFood('nutrition-details')} />
    </Panel>}
    {!low && info && <Tap label="Market reference" onPress={() => go('market-reference')}><Panel style={[s.row, { gap: 12 }]}>
      <Icons name="ShoppingCart" size={22} /><View style={[s.flex, { gap: 3 }]}><T size={16} bold>Market reference</T><T size={13} color={colors.muted}>No verified price yet · choose a market in the app</T></View><Icons name="ChevronRight" size={18} />
    </Panel></Tap>}
    <Panel fill="#f3f1ec" texture="" style={{ gap: 4 }}>
      <T size={15} bold>Quality grade</T>
      <T size={14} color={colors.muted}>Grading not available in this build. These external models only detect which food is in view; they have no freshness or quality output.</T>
    </Panel>
    <Button label="Back to camera" text="Back to camera" onPress={close} />
  </Screen>;
}

function ExtCvAbout({ loaded, close }: { loaded: LoadedModel | null; close: () => void }) {
  return <Screen route="extcv-about" gutter={16} header={<Header title="Models & licences" back={close} />} gap={14}>
    <T size={15} color={colors.muted}>This experimental build of NutriSole uses external, third-party detection models without any NutriSole-trained model. Results are not medical or food-safety advice.</T>
    {MODEL_ORDER.map(id => { const m = MODELS[id]; return <Panel key={id} style={{ gap: 6 }}>
      <T size={17} bold>{m.title}</T>
      <T size={13} color={colors.muted} selectable>{`Source: ${m.source}`}</T>
      <T size={13} color={colors.muted} selectable>{`Weights: ${m.weightsUrl}`}</T>
      <T size={13} color={colors.muted}>{`Licence: ${m.licence}`}</T>
      <T size={12} color={colors.muted} selectable>{`ONNX ${m.file}\nsha256 ${m.sha256}`}</T>
      {loaded?.id === id && <T size={12} color={loaded.hashOk ? colors.green : '#a52d22'}>{loaded.hashOk ? 'Loaded file hash verified on this device' : `Loaded file hash differs: ${loaded.file.sha256}`}</T>}
      <Tap label={`Open ${m.short} source`} onPress={() => Linking.openURL(m.source)}><T size={14} bold color={colors.green}>Open source page</T></Tap>
    </Panel>; })}
    <Panel style={{ gap: 6 }}>
      <T size={17} bold>Ultralytics · AGPL-3.0</T>
      <T size={13} color={colors.muted}>Both models were trained with Ultralytics YOLO and are distributed under the GNU Affero General Public License v3.0. If you distribute this app, its complete corresponding source must be offered under the AGPL, or an Ultralytics Enterprise licence obtained. https://github.com/ultralytics/ultralytics · https://www.gnu.org/licenses/agpl-3.0.html</T>
    </Panel>
    <Panel style={{ gap: 6 }}>
      <T size={17} bold>Data</T>
      <T size={13} color={colors.muted}>YOLOv8m was trained on a fruit and vegetable subset of LVIS (annotations CC BY 4.0, images from COCO/Flickr under their own licences). The YOLO11n repository has no licence file, so its weights are not stored in the NutriSole source repository; they are fetched from the author at build time for this experimental QA build only.</T>
    </Panel>
    <Panel style={{ gap: 6 }}>
      <T size={17} bold>Runtime</T>
      <T size={13} color={colors.muted}>ONNX Runtime (MIT), react-native-vision-camera and its resizer/worklets plugins (MIT), react-native-worklets (MIT), React Native (MIT).</T>
    </Panel>
  </Screen>;
}
