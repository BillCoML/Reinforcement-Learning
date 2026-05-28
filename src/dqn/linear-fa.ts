/**
 * Linear function-approximation TD(0) for V-learning (semi-gradient).
 * Update: θ ← θ + α·δ·φ(s)  where δ = r + γ·Vθ(s') − Vθ(s)
 * The bootstrap target r + γ·Vθ(s') treats Vθ(s') as constant w.r.t. θ.
 */
import type { MDP, Policy } from "../mdp/types";
import { sampleFromRow } from "../td/helpers";
import type { RNG } from "../td/helpers";

const START_STATE = 0;

function dot(theta: Float64Array, phi: Float64Array): number {
  let v = 0;
  for (let i = 0; i < theta.length; i++) v += theta[i] * phi[i];
  return v;
}

export function linearTDPrediction(
  mdp: MDP,
  policy: Policy,
  featurize: (s: number) => Float64Array,
  nEpisodes: number,
  alpha: number,
  options: { rng?: RNG; randomStart?: boolean } = {},
): { theta: Float64Array; history: Float64Array[] } {
  const rng = options.rng ?? Math.random;
  const randomStart = options.randomStart ?? false;
  const nonTerminals = Array.from({ length: mdp.nS }, (_, s) => s).filter(
    (s) => !mdp.terminals[s],
  );
  const d = featurize(0).length;
  const theta = new Float64Array(d);
  const history: Float64Array[] = [];

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = randomStart
      ? nonTerminals[Math.floor(rng() * nonTerminals.length)]
      : START_STATE;
    history.push(new Float64Array(theta));

    for (let step = 0; step < 500; step++) {
      if (mdp.terminals[s]) break;
      const a = sampleFromRow(policy.pi[s], rng);
      const sp = sampleFromRow(mdp.P[s][a], rng);
      const r = mdp.r[s][a];
      const done = mdp.terminals[sp];

      const phi_s = featurize(s);
      const vSp = done ? 0 : dot(theta, featurize(sp));
      const delta = r + mdp.gamma * vSp - dot(theta, phi_s);

      for (let i = 0; i < d; i++) theta[i] += alpha * delta * phi_s[i];
      s = sp;
      if (done) break;
    }
  }

  return { theta, history };
}

/** φ(s) = (r, c, 1) — plane in row-col space. Generalizes but approximates V^π. */
export function featurizeRowColBias(gridSize: number, s: number): Float64Array {
  const r = Math.floor(s / gridSize);
  const c = s % gridSize;
  return new Float64Array([r, c, 1]);
}

/** φ(s) = one-hot — equivalent to tabular; recovers exact V^π. */
export function featurizeOneHot(nStates: number, s: number): Float64Array {
  const phi = new Float64Array(nStates);
  phi[s] = 1;
  return phi;
}

/** φ(s) = (r, c, r², c², rc, 1) — quadratic in row-col. Better approximation. */
export function featurizeQuadratic(gridSize: number, s: number): Float64Array {
  const r = Math.floor(s / gridSize);
  const c = s % gridSize;
  return new Float64Array([r, c, r * r, c * c, r * c, 1]);
}
