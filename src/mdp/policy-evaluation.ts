/**
 * Policy evaluation: computing V^π. Two routes, both used in the lesson —
 *   1. exact, via the linear solve (I − γP^π) V = R^π  (§6 matrix form), and
 *   2. iterative, by repeatedly applying the Bellman expectation operator T^π
 *      (used to *animate* convergence in the Bellman Backup Lab).
 * This module deliberately stops short of policy iteration (Lesson 3).
 */
import { Matrix, solve } from "ml-matrix";
import type { MDP, Policy } from "./types";

/** Policy-induced transition matrix P^π, with (P^π)_{s,s'} = Σ_a π(a|s) P(s'|s,a). */
export function pPi(mdp: MDP, policy: Policy): Matrix {
  const M = Matrix.zeros(mdp.nS, mdp.nS);
  for (let s = 0; s < mdp.nS; s++) {
    for (let a = 0; a < mdp.nA; a++) {
      const pa = policy.pi[s][a];
      if (pa === 0) continue;
      const row = mdp.P[s][a];
      for (let sp = 0; sp < mdp.nS; sp++) {
        if (row[sp] !== 0) M.set(s, sp, M.get(s, sp) + pa * row[sp]);
      }
    }
  }
  return M;
}

/** Policy-induced reward vector R^π, with (R^π)_s = Σ_a π(a|s) r(s,a). */
export function rPi(mdp: MDP, policy: Policy): number[] {
  const R = new Array<number>(mdp.nS).fill(0);
  for (let s = 0; s < mdp.nS; s++) {
    let v = 0;
    for (let a = 0; a < mdp.nA; a++) v += policy.pi[s][a] * mdp.r[s][a];
    R[s] = v;
  }
  return R;
}

/**
 * Solve V^π = (I − γP^π)^{-1} R^π via LU with partial pivoting (ml-matrix's
 * `solve`), not an explicit inverse — better-conditioned as γ → 1.
 */
export function policyEvaluationExact(mdp: MDP, policy: Policy): number[] {
  const Ppi = pPi(mdp, policy);
  const Rpi = Matrix.columnVector(rPi(mdp, policy));
  const A = Matrix.eye(mdp.nS).sub(Ppi.mul(mdp.gamma));
  const V = solve(A, Rpi).getColumn(0);
  // Terminal states are pinned to 0 by definition (guards tiny solve noise).
  for (let s = 0; s < mdp.nS; s++) if (mdp.terminals[s]) V[s] = 0;
  return V;
}

/** One application of the Bellman expectation operator T^π V = R^π + γP^π V. */
export function bellmanExpectationBackup(mdp: MDP, policy: Policy, V: number[]): number[] {
  const Vnew = new Array<number>(mdp.nS).fill(0);
  for (let s = 0; s < mdp.nS; s++) {
    if (mdp.terminals[s]) {
      Vnew[s] = 0;
      continue;
    }
    let v = 0;
    for (let a = 0; a < mdp.nA; a++) {
      let qsa = mdp.r[s][a];
      const row = mdp.P[s][a];
      for (let sp = 0; sp < mdp.nS; sp++) qsa += mdp.gamma * row[sp] * V[sp];
      v += policy.pi[s][a] * qsa;
    }
    Vnew[s] = v;
  }
  return Vnew;
}
