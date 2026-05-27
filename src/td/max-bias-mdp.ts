/**
 * The Sutton-Barto fig 6.5 maximization-bias MDP.
 *
 * States: A (0), B (1), Terminal (2).
 * Actions at A: 0=right → Terminal, r=0; 1=left → B, r=0.
 * Actions at B: 0–9, all → Terminal with reward ~ N(−0.1, 1).
 *
 * Q-learning systematically overestimates max_{a'} Q(B, a') because
 * noisy Q estimates have positive deviations that the max selects.
 * This produces a left-from-A fraction well above the optimal ~5% (ε/2).
 *
 * nA is 10 globally (the max across all states). epsGreedy at A must
 * only sample from actions 0–1 (nActionsAt[A] = 2).
 */
import type { RNG } from "./helpers";

export const A_STATE = 0;
export const B_STATE = 1;
export const TERMINAL_STATE = 2;
export const N_ACTIONS_A = 2;
export const N_ACTIONS_B = 10;

export interface MaxBiasMDP {
  readonly nS: number;
  readonly nA: number; // 10 — global max, used for Q array sizing
  readonly nActionsAt: readonly number[]; // per-state action counts
  readonly terminals: readonly boolean[];
  readonly gamma: number;
}

export function makeMaxBiasMDP(): MaxBiasMDP {
  return {
    nS: 3,
    nA: N_ACTIONS_B,
    nActionsAt: [N_ACTIONS_A, N_ACTIONS_B, 1],
    terminals: [false, false, true],
    gamma: 1.0,
  };
}

// Box-Muller: one N(mean, std) sample using two uniform draws.
function sampleGaussian(mean: number, std: number, rng: RNG): number {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1 < 1e-10 ? 1e-10 : u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

/** Sample one step of the max-bias MDP. Gaussian rewards at B use the RNG. */
export function sampleMaxBiasStep(
  _mdp: MaxBiasMDP,
  s: number,
  a: number,
  rng: RNG,
): { sp: number; r: number; done: boolean } {
  if (s === A_STATE) {
    if (a === 0) return { sp: TERMINAL_STATE, r: 0, done: true }; // right
    return { sp: B_STATE, r: 0, done: false }; // left
  }
  if (s === B_STATE) {
    const r = sampleGaussian(-0.1, 1.0, rng); // N(-0.1, 1)
    return { sp: TERMINAL_STATE, r, done: true };
  }
  return { sp: TERMINAL_STATE, r: 0, done: true };
}

function argmaxQAt(Q: Float64Array, s: number, nA: number, nActionsAt: readonly number[]): number {
  const n = nActionsAt[s];
  const base = s * nA;
  let bestA = 0;
  let bestVal = Q[base];
  for (let a = 1; a < n; a++) {
    if (Q[base + a] > bestVal) {
      bestVal = Q[base + a];
      bestA = a;
    }
  }
  return bestA;
}

function maxQAt(Q: Float64Array, s: number, nA: number, nActionsAt: readonly number[]): number {
  return Q[s * nA + argmaxQAt(Q, s, nA, nActionsAt)];
}

function epsGreedyAt(
  Q: Float64Array,
  s: number,
  nA: number,
  nActionsAt: readonly number[],
  eps: number,
  rng: RNG,
): number {
  const nAs = nActionsAt[s];
  if (rng() < eps) return Math.floor(rng() * nAs);
  return argmaxQAt(Q, s, nA, nActionsAt);
}

export interface MaxBiasQResult {
  Q: Float64Array;
  /** Fraction of episodes where "left" was chosen from A (over all nEpisodes). */
  leftFraction: number;
  /** Running fraction of left-from-A choices, one entry per episode. */
  leftHistory: Float64Array;
}

/** Q-learning on the max-bias MDP, logging left-from-A choice frequency. */
export function qLearningMaxBias(
  mdp: MaxBiasMDP,
  nEpisodes: number,
  epsilon: number,
  alpha: number,
  options: { rng?: RNG } = {},
): MaxBiasQResult {
  const { nA, nActionsAt, terminals } = mdp;
  const Q = new Float64Array(mdp.nS * nA);
  const rng = options.rng ?? Math.random;
  const leftHistory = new Float64Array(nEpisodes);
  let leftCount = 0;

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = A_STATE;
    while (!terminals[s]) {
      const a = epsGreedyAt(Q, s, nA, nActionsAt, epsilon, rng);
      if (s === A_STATE && a === 1) leftCount++;
      const { sp, r, done } = sampleMaxBiasStep(mdp, s, a, rng);
      const target = done ? r : r + mdp.gamma * maxQAt(Q, sp, nA, nActionsAt);
      Q[s * nA + a] += alpha * (target - Q[s * nA + a]);
      if (done) break;
      s = sp;
    }
    leftHistory[ep] = leftCount / (ep + 1);
  }

  return { Q, leftFraction: leftCount / nEpisodes, leftHistory };
}
