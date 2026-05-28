import { Matrix, solve } from "ml-matrix";
import type { MDP, Policy } from "../mdp/types";
import { uniformPolicy } from "../mdp/gridworld";
import { pPi } from "../mdp/policy-evaluation";
import { boltzmannProbs, shannonEntropy } from "./logsumexp";
import type { SoftValueIterationResult } from "./types";

export function softPolicyIteration(
  alpha: number,
  mdp: MDP,
  options?: { tol?: number; maxIter?: number },
): SoftValueIterationResult {
  const tol = options?.tol ?? 1e-9;
  const maxIter = options?.maxIter ?? 200;
  const { nS, nA } = mdp;

  let policy: Policy = uniformPolicy(mdp);

  for (let iter = 0; iter < maxIter; iter++) {
    // --- Soft policy evaluation ---
    // R_soft^pi(s) = sum_a pi(a|s) r(s,a) + alpha * H(pi(.|s))
    const Rsoft = new Array<number>(nS).fill(0);
    for (let s = 0; s < nS; s++) {
      if (mdp.terminals[s]) continue;
      let r = 0;
      const piRow = new Float64Array(nA);
      for (let a = 0; a < nA; a++) {
        piRow[a] = policy.pi[s][a];
        r += policy.pi[s][a] * mdp.r[s][a];
      }
      Rsoft[s] = r + alpha * shannonEntropy(piRow);
    }
    // Solve (I - gamma * P^pi) V = Rsoft
    const Ppi = pPi(mdp, policy);
    const A = Matrix.eye(nS).sub(Ppi.mul(mdp.gamma));
    const Rmat = Matrix.columnVector(Rsoft);
    const Varr = solve(A, Rmat).getColumn(0);
    for (let s = 0; s < nS; s++) if (mdp.terminals[s]) Varr[s] = 0;

    // --- Compute Q^soft-pi ---
    const Q = new Float64Array(nS * nA);
    for (let s = 0; s < nS; s++) {
      if (mdp.terminals[s]) continue;
      for (let a = 0; a < nA; a++) {
        let q = mdp.r[s][a];
        const row = mdp.P[s][a];
        for (let sp = 0; sp < nS; sp++) q += mdp.gamma * row[sp] * Varr[sp];
        Q[s * nA + a] = q;
      }
    }

    // --- Soft policy improvement: pi_new = Boltzmann(Q/alpha) ---
    const piNew = new Float64Array(nS * nA);
    const policyNew: number[][] = [];
    for (let s = 0; s < nS; s++) {
      if (mdp.terminals[s]) {
        for (let a = 0; a < nA; a++) piNew[s * nA + a] = 1 / nA;
        policyNew.push(Array.from({ length: nA }, () => 1 / nA));
        continue;
      }
      const probs = boltzmannProbs(Q.subarray(s * nA, (s + 1) * nA), alpha);
      for (let a = 0; a < nA; a++) piNew[s * nA + a] = probs[a];
      policyNew.push(Array.from(probs));
    }

    // Convergence check on policy (sup norm of pi change)
    let maxDelta = 0;
    for (let i = 0; i < piNew.length; i++) {
      const oldVal = policy.pi[Math.floor(i / nA)][i % nA];
      maxDelta = Math.max(maxDelta, Math.abs(piNew[i] - oldVal));
    }

    policy = { pi: policyNew };
    const V = new Float64Array(Varr);

    if (maxDelta < tol) {
      return { Q, V, pi: piNew, iterations: iter + 1, converged: true };
    }
  }

  // Final values
  const piFlat = new Float64Array(nS * nA);
  for (let s = 0; s < nS; s++)
    for (let a = 0; a < nA; a++)
      piFlat[s * nA + a] = policy.pi[s][a];

  const Rsoft = new Array<number>(nS).fill(0);
  for (let s = 0; s < nS; s++) {
    if (mdp.terminals[s]) continue;
    const piRow = new Float64Array(nA);
    for (let a = 0; a < nA; a++) {
      piRow[a] = policy.pi[s][a];
      Rsoft[s] += policy.pi[s][a] * mdp.r[s][a];
    }
    Rsoft[s] += alpha * shannonEntropy(piRow);
  }
  const Ppi = pPi(mdp, policy);
  const A = Matrix.eye(nS).sub(Ppi.mul(mdp.gamma));
  const Varr = solve(A, Matrix.columnVector(Rsoft)).getColumn(0);
  for (let s = 0; s < nS; s++) if (mdp.terminals[s]) Varr[s] = 0;

  const Q = new Float64Array(nS * nA);
  for (let s = 0; s < nS; s++) {
    if (mdp.terminals[s]) continue;
    for (let a = 0; a < nA; a++) {
      let q = mdp.r[s][a];
      const row = mdp.P[s][a];
      for (let sp = 0; sp < nS; sp++) q += mdp.gamma * row[sp] * Varr[sp];
      Q[s * nA + a] = q;
    }
  }

  return { Q, V: new Float64Array(Varr), pi: piFlat, iterations: maxIter, converged: false };
}
