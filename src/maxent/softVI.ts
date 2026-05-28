import type { MDP } from "../mdp/types";
import { logSumExp, boltzmannProbs } from "./logsumexp";
import type { SoftValueIterationResult } from "./types";

export function softValueIteration(
  alpha: number,
  mdp: MDP,
  options?: { tol?: number; maxIter?: number },
): SoftValueIterationResult {
  const tol = options?.tol ?? 1e-9;
  const maxIter = options?.maxIter ?? 10000;
  const { nS, nA } = mdp;

  let V = new Float64Array(nS);           // soft value function
  const Q = new Float64Array(nS * nA);   // soft Q-function

  for (let iter = 0; iter < maxIter; iter++) {
    const Vnew = new Float64Array(nS);    // terminals stay 0
    let maxDelta = 0;

    for (let s = 0; s < nS; s++) {
      if (mdp.terminals[s]) continue;     // V_soft(terminal) = 0 always

      // Q(s,a) = r(s,a) + gamma * sum_sp P(sp|s,a) * V(sp)
      for (let a = 0; a < nA; a++) {
        let q = mdp.r[s][a];
        const row = mdp.P[s][a];
        for (let sp = 0; sp < nS; sp++) q += mdp.gamma * row[sp] * V[sp];
        Q[s * nA + a] = q;
      }

      // V_soft(s) = alpha * log( sum_a exp(Q(s,a)/alpha) )  [shifted]
      Vnew[s] = logSumExp(Q.subarray(s * nA, (s + 1) * nA), alpha);
      maxDelta = Math.max(maxDelta, Math.abs(Vnew[s] - V[s]));
    }

    V = Vnew;

    if (maxDelta < tol) {
      // Recompute Q from converged V for clean output
      _recomputeQ(Q, V, mdp);
      const pi = _boltzmannPolicy(Q, mdp, alpha, nS, nA);
      return { Q: Q.slice(), V: V.slice(), pi, iterations: iter + 1, converged: true };
    }
  }

  _recomputeQ(Q, V, mdp);
  const pi = _boltzmannPolicy(Q, mdp, alpha, nS, nA);
  return { Q: Q.slice(), V: V.slice(), pi, iterations: maxIter, converged: false };
}

function _recomputeQ(Q: Float64Array, V: Float64Array, mdp: MDP): void {
  const { nS, nA } = mdp;
  for (let s = 0; s < nS; s++) {
    if (mdp.terminals[s]) {
      for (let a = 0; a < nA; a++) Q[s * nA + a] = 0;
      continue;
    }
    for (let a = 0; a < nA; a++) {
      let q = mdp.r[s][a];
      const row = mdp.P[s][a];
      for (let sp = 0; sp < nS; sp++) q += mdp.gamma * row[sp] * V[sp];
      Q[s * nA + a] = q;
    }
  }
}

function _boltzmannPolicy(
  Q: Float64Array,
  mdp: MDP,
  alpha: number,
  nS: number,
  nA: number,
): Float64Array {
  const pi = new Float64Array(nS * nA);
  for (let s = 0; s < nS; s++) {
    if (mdp.terminals[s]) {
      // Uniform over terminals (convention; they never matter).
      for (let a = 0; a < nA; a++) pi[s * nA + a] = 1 / nA;
      continue;
    }
    const probs = boltzmannProbs(Q.subarray(s * nA, (s + 1) * nA), alpha);
    for (let a = 0; a < nA; a++) pi[s * nA + a] = probs[a];
  }
  return pi;
}
