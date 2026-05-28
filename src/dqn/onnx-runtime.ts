/**
 * ONNX runtime wrapper for browser-side DQN inference.
 * Lazily loads onnxruntime-web; implements an LRU session cache (max 50 models).
 *
 * The training script exports, alongside each ONNX checkpoint, a companion
 * q_grid.json with pre-computed Q-values for all states. The heatmap (Panel B)
 * uses these pre-computed values for zero-latency scrubbing. ONNX inference is
 * used only for click-to-inspect (Panel D) to produce Q-values at any step.
 */

export interface DQNTrace {
  config: string;
  env: string;
  seed: number;
  nEpisodes: number;
  /** Per-step scalars */
  lossPerStep: number[];
  paramNormPerStep: number[];
  bellmanErrorPerStep: number[];
  /** Q(s, a) for selected states over training time, shape [nStates][nSteps] */
  qTracesPerState: number[][];
  /** Per-checkpoint: index of training step */
  checkpointSteps: number[];
  /** Per-checkpoint: full Q-grid [nStates × nActions], flattened row-major */
  qGridPerCheckpoint: number[][];
}

export interface SessionCache {
  sessions: Map<string, Promise<unknown>>;
  lruKeys: string[];
  maxSize: number;
}

const _cache: SessionCache = { sessions: new Map(), lruKeys: [], maxSize: 50 };

function makeCacheKey(config: string, env: string, seed: number, step: number): string {
  return `${config}/${env}/${seed}/step${step}`;
}

/** Lazily load onnxruntime-web and create an InferenceSession. */
export async function getSession(config: string, env: string, seed: number, step: number) {
  const key = makeCacheKey(config, env, seed, step);
  if (_cache.sessions.has(key)) {
    // Move to end of LRU list
    _cache.lruKeys = _cache.lruKeys.filter((k) => k !== key);
    _cache.lruKeys.push(key);
    return _cache.sessions.get(key)!;
  }

  // Evict if over capacity
  while (_cache.lruKeys.length >= _cache.maxSize) {
    const evict = _cache.lruKeys.shift()!;
    _cache.sessions.delete(evict);
  }

  const modelPath = `/models/dqn/${config}/${env}/seed${seed}/step${step}.onnx`;
  const sessionPromise = import("onnxruntime-web").then(async (ort) => {
    const session = await ort.InferenceSession.create(modelPath);
    return { session, ort };
  });

  _cache.sessions.set(key, sessionPromise);
  _cache.lruKeys.push(key);
  return sessionPromise;
}

/** Run inference for one state: returns Q-values for all actions. */
export async function qValuesForState(
  config: string,
  env: string,
  seed: number,
  step: number,
  stateIdx: number,
  nStates: number,
): Promise<Float32Array> {
  const result = await getSession(config, env, seed, step) as { session: any; ort: any };
  const { session, ort } = result;
  const input = new Float32Array(nStates);
  input[stateIdx] = 1;
  const tensor = new ort.Tensor("float32", input, [1, nStates]);
  const out = await session.run({ state: tensor });
  return out.q_values.data as Float32Array;
}

/** Load a training trace JSON. */
export async function loadTrace(config: string, env: string, seed: number): Promise<DQNTrace> {
  const url = `/data/dqn/${config}/${env}/seed${seed}.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load trace: ${url}`);
  return res.json() as Promise<DQNTrace>;
}

/** Load pre-computed Q-grid for a checkpoint (from trace). */
export function qGridAtCheckpoint(trace: DQNTrace, step: number): number[][] | null {
  const idx = trace.checkpointSteps.indexOf(step);
  if (idx < 0) {
    // Find nearest
    const nearest = trace.checkpointSteps.reduce((prev, curr) =>
      Math.abs(curr - step) < Math.abs(prev - step) ? curr : prev,
    );
    const nearIdx = trace.checkpointSteps.indexOf(nearest);
    return nearIdx >= 0 ? chunkGrid(trace.qGridPerCheckpoint[nearIdx], 4) : null;
  }
  return chunkGrid(trace.qGridPerCheckpoint[idx], 4);
}

function chunkGrid(flat: number[], nA: number): number[][] {
  const nS = flat.length / nA;
  return Array.from({ length: nS }, (_, s) => flat.slice(s * nA, (s + 1) * nA));
}
