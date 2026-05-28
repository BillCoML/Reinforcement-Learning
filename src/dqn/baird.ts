/**
 * Baird's counterexample (Baird 1995; Sutton & Barto §11.1).
 * 7-state MDP with 2 actions demonstrating that the deadly triad
 * (function approximation + bootstrapping + off-policy) causes divergence.
 *
 * States 0–5: φ(i) = 2·eᵢ + e₆   (2 in pos i, 1 in pos 6)
 * State 6:    φ(6) = 2·e₇          (2 in pos 7)
 * θ ∈ ℝ⁸. True V* = 0 for all states (all rewards are 0).
 * With θ₀ = [1,1,1,1,1,1,10,1], ‖θ₀‖ ≈ 10.3.
 */
import { mulberry32 } from "../importance-sampling/gaussian";

export type { RNG } from "../td/helpers";

const N_STATES = 7;
const N_DIM = 8;
const GAMMA = 0.99;

const DASHED = 0;
const SOLID = 1;

/**
 * 8-dimensional feature vector for Baird's 7-state MDP (S&B 2nd ed §11.1).
 * States 0–5: 2 in their unique dim, 1 in shared dim 6.
 * State 6:    2 in shared dim 6, 1 in unique dim 7.
 * This coupling is the source of the deadly-triad divergence.
 */
function phi(s: number): Float64Array {
  const f = new Float64Array(N_DIM);
  if (s < 6) {
    f[s] = 2;
    f[6] = 1;
  } else {
    // State 6 (state 7 in 1-indexed): shared dim 6 gets 2, unique dim 7 gets 1.
    f[6] = 2;
    f[7] = 1;
  }
  return f;
}

function dot(theta: Float64Array, f: Float64Array): number {
  let v = 0;
  for (let i = 0; i < N_DIM; i++) v += theta[i] * f[i];
  return v;
}

function norm(theta: Float64Array): number {
  let s = 0;
  for (let i = 0; i < N_DIM; i++) s += theta[i] * theta[i];
  return Math.sqrt(s);
}

export interface BairdResult {
  /** ‖θ‖ recorded at each step (index = step number, 0-indexed). */
  normHistory: number[];
  /** θ sampled at every 100 steps. */
  thetaHistory: Float64Array[];
}

export interface BairdOptions {
  alpha?: number;
  /** θ initial value (length 8). Default: [1,1,1,1,1,1,10,1]. */
  initTheta?: Float64Array;
  /** If true, behavior = target (no off-policy component) → bounded. */
  onPolicy?: boolean;
  /** If false, use MC return (G=0, since all r=0) instead of TD target. */
  bootstrap?: boolean;
  rng?: () => number;
}

/**
 * Run Baird's counterexample for nSteps iterations.
 * Each step: sample one state uniformly, sample one action from behavior policy.
 */
export function bairdCounterexample(
  nSteps: number,
  options: BairdOptions = {},
): BairdResult {
  const {
    alpha = 0.01,
    initTheta,
    onPolicy = false,
    bootstrap = true,
    rng = Math.random,
  } = options;

  const theta = initTheta
    ? new Float64Array(initTheta)
    : new Float64Array([1, 1, 1, 1, 1, 1, 10, 1]);

  const normHistory: number[] = new Array(nSteps + 1);
  const thetaHistory: Float64Array[] = [];

  normHistory[0] = norm(theta);
  thetaHistory.push(new Float64Array(theta));

  for (let step = 0; step < nSteps; step++) {
    // Sample state uniformly
    const s = Math.floor(rng() * N_STATES);

    // Sample action from behavior (50/50 dashed/solid)
    const a = rng() < 0.5 ? DASHED : SOLID;

    // Importance sampling ratio
    let rho: number;
    if (onPolicy) {
      rho = 1; // behavior = target, no IS needed
    } else {
      // π_target(solid) = 1, π_target(dashed) = 0
      // b(solid) = 0.5, b(dashed) = 0.5
      rho = a === SOLID ? 2 : 0;
    }

    if (rho === 0) {
      // dashed action under off-policy: no update
      normHistory[step + 1] = norm(theta);
      continue;
    }

    // Next state
    let sp: number;
    if (a === SOLID) {
      sp = 6; // solid always → state 6
    } else {
      // dashed → uniform over states 0..5
      sp = Math.floor(rng() * 6);
    }

    const phi_s = phi(s);
    const vs = dot(theta, phi_s);
    const vsp = dot(theta, phi(sp));

    let delta: number;
    if (bootstrap) {
      // Semi-gradient TD(0): target = γ·V(s')
      delta = GAMMA * vsp - vs;
    } else {
      // MC: true return = 0 (all rewards 0; target policy always reaches state 6 quickly)
      delta = 0 - vs;
    }

    for (let i = 0; i < N_DIM; i++) {
      theta[i] += alpha * rho * delta * phi_s[i];
    }

    normHistory[step + 1] = norm(theta);
    if ((step + 1) % 100 === 0) thetaHistory.push(new Float64Array(theta));
  }

  return { normHistory, thetaHistory };
}

export { mulberry32 };
