// ONNX Runtime session management for the bundled external detectors.
// The .onnx files ship inside the APK (android/app/src/main/assets/models/). ONNX Runtime needs a file path, so the
// small native module ExtCvModelFiles copies the asset into app storage once and returns its path and SHA-256.

import { NativeModules, TurboModuleRegistry } from 'react-native';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { MODELS, type ModelId } from './models';

type ModelFile = { path: string; sha256: string; bytes: number; copied: boolean };
type ModelFilesModule = { ensure(name: string, expectedBytes: number): Promise<ModelFile> };

const files = (TurboModuleRegistry.get('ExtCvModelFiles') ?? NativeModules.ExtCvModelFiles) as unknown as ModelFilesModule | undefined;

export type LoadedModel = {
  id: ModelId;
  session: InferenceSession;
  inputName: string;
  outputName: string;
  file: ModelFile;
  hashOk: boolean;
  loadMs: number;
};

export async function loadModel(id: ModelId): Promise<LoadedModel> {
  const spec = MODELS[id];
  if (!files) throw new Error('ExtCvModelFiles native module is missing from this build');
  const t0 = Date.now();
  const file = await files.ensure(spec.file, spec.bytes);
  const session = await InferenceSession.create(file.path, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    intraOpNumThreads: 4,
  });
  return { id, session, inputName: session.inputNames[0], outputName: session.outputNames[0], file, hashOk: file.sha256 === spec.sha256, loadMs: Date.now() - t0 };
}

/** One inference on a 1x3xSxS float32 RGB tensor in [0, 1]. Returns the raw [1, 4+C, 8400] output. */
export async function runModel(m: LoadedModel, input: Float32Array): Promise<{ output: Float32Array; ms: number }> {
  const size = MODELS[m.id].inputSize;
  const t0 = Date.now();
  const result = await m.session.run({ [m.inputName]: new Tensor('float32', input, [1, 3, size, size]) });
  const ms = Date.now() - t0;
  return { output: result[m.outputName].data as Float32Array, ms };
}

export async function releaseModel(m: LoadedModel | null) {
  try { await m?.session.release(); } catch { /* already released */ }
}
