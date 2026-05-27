/**
 * DP algorithms: iterative policy evaluation, policy improvement,
 * policy iteration, value iteration, modified PI, and async VI.
 * All are thin wrappers / compositions of src/mdp/ — nothing new mathematically.
 */
import type { MDP, Policy } from '../mdp/types';
import { bellmanExpectationBackup } from '../mdp/policy-evaluation';
import { bellmanOptimalityBackup, qOfAction } from '../mdp/value-iteration';
import { policyEvaluationExact } from '../mdp/policy-evaluation';
import { qFromV } from '../mdp/q-and-advantage';
import { uniformPolicy } from '../mdp/gridworld';
import { supDist, clonePolicy, policiesEqual } from './utils';

export { uniformPolicy, supDist, clonePolicy, policiesEqual };

// ── Iterative Policy Evaluation ──────────────────────────────────────────────

export interface PEResult {
  V: number[];
  iterations: number;
  trace: number[][];
}

/** Iterative PE: V_{k+1} = T^π V_k until supDist < tol. */
export function policyEvaluationIterative(
  mdp: MDP,
  policy: Policy,
  tol = 1e-9,
  maxIter = 10000,
): PEResult {
  let V = new Array<number>(mdp.nS).fill(0);
  const trace: number[][] = [V.slice()];
  for (let k = 0; k < maxIter; k++) {
    const Vn = bellmanExpectationBackup(mdp, policy, V);
    trace.push(Vn.slice());
    if (supDist(Vn, V) < tol) return { V: Vn, iterations: k + 1, trace };
    V = Vn;
  }
  return { V, iterations: maxIter, trace };
}

// ── Policy Improvement ───────────────────────────────────────────────────────

/**
 * Greedy policy improvement: π'(s) = argmax_a Q^π(s,a).
 * Tie-break: first-found (lowest action index). Terminal states get row[0]=1.
 */
export function policyImprovement(mdp: MDP, V: number[]): Policy {
  const Q = qFromV(mdp, V);
  const pi: number[][] = [];
  for (let s = 0; s < mdp.nS; s++) {
    const row = new Array<number>(mdp.nA).fill(0);
    if (mdp.terminals[s]) {
      row[0] = 1;
      pi.push(row);
      continue;
    }
    let bestA = 0;
    let bestQ = -Infinity;
    for (let a = 0; a < mdp.nA; a++) {
      if (Q[s][a] > bestQ) {
        bestQ = Q[s][a];
        bestA = a;
      }
    }
    row[bestA] = 1;
    pi.push(row);
  }
  return { pi };
}

// ── Policy Iteration ─────────────────────────────────────────────────────────

export interface PIStep {
  V: number[];
  policy: Policy;
}

export interface PIResult {
  V: number[];
  policy: Policy;
  iterations: number;
  history: PIStep[];
}

/**
 * Full policy iteration.
 * history[k] = { V^{π_k}, π_k } — one entry per outer iteration.
 * Converges in 3 outer iterations on the default gridworld.
 */
export function policyIteration(mdp: MDP, maxIter = 100): PIResult {
  let policy = uniformPolicy(mdp);
  const history: PIStep[] = [];
  for (let k = 0; k < maxIter; k++) {
    const V = policyEvaluationExact(mdp, policy);
    history.push({ V: V.slice(), policy: clonePolicy(policy) });
    const newPolicy = policyImprovement(mdp, V);
    if (policiesEqual(policy, newPolicy)) return { V, policy, iterations: k + 1, history };
    policy = newPolicy;
  }
  throw new Error('PI failed to converge');
}

// ── Value Iteration ──────────────────────────────────────────────────────────

export interface VIResult {
  V: number[];
  policy: Policy;
  iterations: number;
  trace: number[][];
}

/**
 * Synchronous value iteration: V_{k+1} = T* V_k.
 * Stopping criterion: supDist(V_{k+1}, V_k) < ε(1-γ)/γ guarantees ε-optimality.
 */
export function valueIteration(mdp: MDP, epsilon = 1e-8, maxIter = 10000): VIResult {
  let V = new Array<number>(mdp.nS).fill(0);
  const trace: number[][] = [V.slice()];
  const stopThreshold = epsilon * (1 - mdp.gamma) / mdp.gamma;
  for (let k = 0; k < maxIter; k++) {
    const Vn = bellmanOptimalityBackup(mdp, V);
    trace.push(Vn.slice());
    if (supDist(Vn, V) < stopThreshold) {
      const policy = policyImprovement(mdp, Vn);
      return { V: Vn, policy, iterations: k + 1, trace };
    }
    V = Vn;
  }
  throw new Error('VI failed to converge');
}

// ── Modified Policy Iteration ────────────────────────────────────────────────

/**
 * Modified PI: m PE steps per outer iteration.
 * m = 1 → VI-style behavior; m ≤ 0 or m = Infinity → full PI (exact PE).
 */
export function modifiedPolicyIteration(
  mdp: MDP,
  m: number,
  maxIter = 200,
): PIResult {
  let policy = uniformPolicy(mdp);
  let V = new Array<number>(mdp.nS).fill(0);
  const history: PIStep[] = [];
  const fullPE = m <= 0 || !isFinite(m);
  for (let k = 0; k < maxIter; k++) {
    if (fullPE) {
      V = policyEvaluationExact(mdp, policy);
    } else {
      for (let i = 0; i < m; i++) V = bellmanExpectationBackup(mdp, policy, V);
    }
    history.push({ V: V.slice(), policy: clonePolicy(policy) });
    const newPolicy = policyImprovement(mdp, V);
    if (policiesEqual(policy, newPolicy)) return { V, policy, iterations: k + 1, history };
    policy = newPolicy;
  }
  throw new Error('Modified PI failed to converge');
}

// ── Asynchronous (Gauss-Seidel) Value Iteration ──────────────────────────────

export interface AsyncVIResult {
  V: number[];
  policy: Policy;
  iterations: number;
}

/**
 * In-place (Gauss-Seidel) VI. Each sweep over sweepOrder is one iteration.
 * Stops when max absolute change < ε(1-γ)/γ across a sweep.
 */
export function asyncValueIteration(
  mdp: MDP,
  sweepOrder: number[],
  epsilon = 1e-8,
  maxIter = 10000,
): AsyncVIResult {
  const V = new Array<number>(mdp.nS).fill(0);
  const stopThreshold = epsilon * (1 - mdp.gamma) / mdp.gamma;
  for (let k = 0; k < maxIter; k++) {
    let maxDelta = 0;
    for (const s of sweepOrder) {
      if (mdp.terminals[s]) continue;
      const oldV = V[s];
      let best = -Infinity;
      for (let a = 0; a < mdp.nA; a++) {
        best = Math.max(best, qOfAction(mdp, V, s, a));
      }
      V[s] = best;
      maxDelta = Math.max(maxDelta, Math.abs(V[s] - oldV));
    }
    if (maxDelta < stopThreshold) {
      const policy = policyImprovement(mdp, V);
      return { V: V.slice(), policy, iterations: k + 1 };
    }
  }
  throw new Error('Async VI failed to converge');
}
