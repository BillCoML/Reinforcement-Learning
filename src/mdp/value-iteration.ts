/**
 * The Bellman *optimality* operator T^* and the greedy policy it induces.
 *
 * Note (curriculum boundary): this lesson does NOT implement value iteration
 * as an algorithm — that's Lesson 3. What lives here is the single-step
 * operator (used to animate the optimality mode of the Bellman Backup Lab) and
 * a thin `optimalValue` helper that iterates it purely to obtain reference
 * values (V^* targets for the convergence trace, tests, and sanity checks).
 */
import type { MDP, Policy } from "./types";

/** Q-value of (s,a) under a value function V: r(s,a) + γ Σ_s' P(s'|s,a) V(s'). */
export function qOfAction(mdp: MDP, V: number[], s: number, a: number): number {
  let q = mdp.r[s][a];
  const row = mdp.P[s][a];
  for (let sp = 0; sp < mdp.nS; sp++) q += mdp.gamma * row[sp] * V[sp];
  return q;
}

/** One application of the Bellman optimality operator T^* V = max_a [r + γP V]. */
export function bellmanOptimalityBackup(mdp: MDP, V: number[]): number[] {
  const Vnew = new Array<number>(mdp.nS).fill(0);
  for (let s = 0; s < mdp.nS; s++) {
    if (mdp.terminals[s]) {
      Vnew[s] = 0;
      continue;
    }
    let best = -Infinity;
    for (let a = 0; a < mdp.nA; a++) best = Math.max(best, qOfAction(mdp, V, s, a));
    Vnew[s] = best;
  }
  return Vnew;
}

/** Iterate T^* from V_0 = 0 to convergence — reference V^* for tests/traces. */
export function optimalValue(mdp: MDP, iters = 200): number[] {
  let V = new Array<number>(mdp.nS).fill(0);
  for (let k = 0; k < iters; k++) V = bellmanOptimalityBackup(mdp, V);
  return V;
}

/**
 * All actions tying for the argmax at state s (within `tol`). Ties are real —
 * e.g. (0,0) has two optimal actions — so we never break them silently; the
 * UI surfaces the multiplicity (V7's badge).
 */
export function greedyActions(mdp: MDP, V: number[], s: number, tol = 1e-9): number[] {
  let best = -Infinity;
  const q: number[] = [];
  for (let a = 0; a < mdp.nA; a++) {
    q[a] = qOfAction(mdp, V, s, a);
    if (q[a] > best) best = q[a];
  }
  const tied: number[] = [];
  for (let a = 0; a < mdp.nA; a++) if (q[a] >= best - tol) tied.push(a);
  return tied;
}

/**
 * Greedy policy w.r.t. V. Ties are split uniformly across the tied actions, so
 * the returned policy is the (possibly stochastic) uniform-over-argmax policy.
 * Terminal states get a harmless uniform row.
 */
export function greedyPolicy(mdp: MDP, V: number[], tol = 1e-9): Policy {
  const pi = Array.from({ length: mdp.nS }, (_, s) => {
    const row = new Array<number>(mdp.nA).fill(0);
    if (mdp.terminals[s]) {
      row.fill(1 / mdp.nA);
      return row;
    }
    const tied = greedyActions(mdp, V, s, tol);
    for (const a of tied) row[a] = 1 / tied.length;
    return row;
  });
  return { pi };
}
