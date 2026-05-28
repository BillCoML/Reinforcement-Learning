import { Matrix, solve } from "ml-matrix";
import type { MDP, Policy } from "../mdp/types";
import { pPi, policyEvaluationExact } from "../mdp/policy-evaluation";
import { shannonEntropy } from "./logsumexp";
import type { SoftEvaluationResult } from "./types";

/**
 * Two-track evaluation of a policy under the entropy-regularized objective.
 *
 * V_soft[s] = J_alpha(pi) from state s — the expected sum of (reward + alpha*entropy),
 *             computed via the soft Bellman evaluation (linear solve with augmented reward).
 *
 * V_pi[s]   = E^pi[ sum gamma^t r_t ] from state s — the standard discounted return,
 *             no entropy term. Computed via policyEvaluationExact (L5 linear solver).
 *
 * These are different quantities. At alpha=0.02, V_soft[0] ≈ 0.7392 while V_pi[0] ≈ 0.7217.
 */
export function softEvaluate(
  pi: Float64Array,   // flat nS*nA Boltzmann policy
  alpha: number,
  mdp: MDP,
): SoftEvaluationResult {
  const { nS, nA } = mdp;

  // Convert flat Float64Array to Policy object for existing utilities.
  const piMatrix: number[][] = Array.from({ length: nS }, (_, s) =>
    Array.from({ length: nA }, (__, a) => pi[s * nA + a]),
  );
  const policy: Policy = { pi: piMatrix };

  // --- V_soft: solve (I - gamma*P^pi) V = R_soft^pi ---
  const Rsoft = new Array<number>(nS).fill(0);
  for (let s = 0; s < nS; s++) {
    if (mdp.terminals[s]) continue;
    let r = 0;
    const piRow = new Float64Array(nA);
    for (let a = 0; a < nA; a++) {
      piRow[a] = pi[s * nA + a];
      r += piRow[a] * mdp.r[s][a];
    }
    Rsoft[s] = r + alpha * shannonEntropy(piRow);
  }
  const Ppi = pPi(mdp, policy);
  const A = Matrix.eye(nS).sub(Ppi.mul(mdp.gamma));
  const VsoftArr = solve(A, Matrix.columnVector(Rsoft)).getColumn(0);
  for (let s = 0; s < nS; s++) if (mdp.terminals[s]) VsoftArr[s] = 0;

  // --- V_pi: standard discounted return, no entropy ---
  const VpiArr = policyEvaluationExact(mdp, policy);

  return {
    V_soft: new Float64Array(VsoftArr),
    V_pi: new Float64Array(VpiArr),
  };
}
