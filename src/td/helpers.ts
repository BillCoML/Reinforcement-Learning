/**
 * Shared utilities for the TD learning module.
 * Provides RNG, step sampling, and Q-value helpers used by all TD algorithms.
 */
import { mulberry32 } from "../importance-sampling/gaussian";
import type { MDP, Policy } from "../mdp/types";

export type RNG = () => number;
export { mulberry32 };

/** State index of the gridworld start cell (0,0). */
export const START_STATE = 0;

/** Sample index from a probability row using one uniform draw. */
export function sampleFromRow(row: number[], rng: RNG): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < row.length; i++) {
    acc += row[i];
    if (u <= acc) return i;
  }
  return row.length - 1;
}

/** Sample action from policy.pi[s]. */
export function samplePolicyAction(policy: Policy, s: number, rng: RNG): number {
  return sampleFromRow(policy.pi[s], rng);
}

/** Sample one transition (s, a) → (sp, r, done). */
export function sampleStep(
  mdp: MDP,
  s: number,
  a: number,
  rng: RNG,
): { sp: number; r: number; done: boolean } {
  const sp = sampleFromRow(mdp.P[s][a], rng);
  return { sp, r: mdp.r[s][a], done: mdp.terminals[sp] };
}

/** Index of the action with the highest Q value at state s. Tie-breaks to lowest index. */
export function argmaxQ(Q: Float64Array, s: number, nA: number): number {
  const base = s * nA;
  let bestA = 0;
  let bestVal = Q[base];
  for (let a = 1; a < nA; a++) {
    if (Q[base + a] > bestVal) {
      bestVal = Q[base + a];
      bestA = a;
    }
  }
  return bestA;
}

/** Maximum Q value at state s. */
export function maxQ(Q: Float64Array, s: number, nA: number): number {
  return Q[s * nA + argmaxQ(Q, s, nA)];
}

/**
 * ε-greedy action at state s.
 * Consumes 1 RNG draw for the eps check; if exploring, 1 more for the random action.
 */
export function epsGreedyAction(
  Q: Float64Array,
  s: number,
  nA: number,
  eps: number,
  rng: RNG,
): number {
  if (rng() < eps) return Math.floor(rng() * nA);
  return argmaxQ(Q, s, nA);
}

/** Expected value of state s under the ε-greedy policy derived from Q. */
export function epsSoftValue(Q: Float64Array, s: number, nA: number, eps: number): number {
  const base = s * nA;
  const bestA = argmaxQ(Q, s, nA);
  const pExplore = eps / nA;
  const pGreedy = 1 - eps + pExplore;
  let v = pGreedy * Q[base + bestA];
  for (let a = 0; a < nA; a++) {
    if (a !== bestA) v += pExplore * Q[base + a];
  }
  return v;
}
