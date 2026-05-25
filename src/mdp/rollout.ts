/**
 * Sample one trajectory under a policy: a sequence of (s, a, r, s') steps.
 * Stops when a terminal state is entered or `maxSteps` is reached. The reward
 * reported per step is r(s, a) — exact for the deterministic gridworld V3 uses.
 */
import type { MDP, Policy } from "./types";

export interface Step {
  s: number;
  a: number;
  r: number;
  sp: number;
}

/** Sample an index from a probability row using a uniform draw. */
function sample(row: number[], u: number): number {
  let acc = 0;
  for (let i = 0; i < row.length; i++) {
    acc += row[i];
    if (u <= acc) return i;
  }
  return row.length - 1; // floating-point fallback
}

export function rollout(
  mdp: MDP,
  policy: Policy,
  s0: number,
  maxSteps: number,
  rng: () => number = Math.random,
): Step[] {
  const steps: Step[] = [];
  let s = s0;
  for (let t = 0; t < maxSteps; t++) {
    if (mdp.terminals[s]) break;
    const a = sample(policy.pi[s], rng());
    const sp = sample(mdp.P[s][a], rng());
    steps.push({ s, a, r: mdp.r[s][a], sp });
    s = sp;
  }
  return steps;
}
