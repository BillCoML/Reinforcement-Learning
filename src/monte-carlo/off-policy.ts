import type { MDP, Policy } from "../mdp/types";
import { rollout } from "../mdp/rollout";
import { computeReturns } from "./returns";

export interface MCOffPolicyResult {
  V: Float64Array;
  ess: Float64Array;
}

/**
 * Off-policy Monte Carlo policy evaluation via importance sampling.
 * Per-state first-visit IS weights using ρ_{t:T-1} suffix products.
 * weighted=true → self-normalized (weighted IS); false → ordinary IS.
 * When denominator is zero returns 0 and sets ess to 0.
 */
export function mcOffPolicyEvaluation(
  mdp: MDP,
  target: Policy,
  behavior: Policy,
  nEpisodes: number,
  options: {
    weighted?: boolean;
    s0?: number;
    maxSteps?: number;
    rng?: () => number;
  } = {},
): MCOffPolicyResult {
  const weighted = options.weighted ?? true;
  const s0 = options.s0 ?? 0;
  const maxSteps = options.maxSteps ?? 200;
  const rng = options.rng ?? Math.random;

  const numArr = new Float64Array(mdp.nS);
  const denArr = new Float64Array(mdp.nS);   // weighted: Σ ρ
  const countArr = new Int32Array(mdp.nS);    // ordinary: Σ 1 (= N for s0)
  const sumW = new Float64Array(mdp.nS);      // for ESS numerator
  const sumW2 = new Float64Array(mdp.nS);     // for ESS denominator

  for (let ep = 0; ep < nEpisodes; ep++) {
    const steps = rollout(mdp, behavior, s0, maxSteps, rng);
    const T = steps.length;
    if (T === 0) continue;
    const rewards = steps.map(st => st.r);
    const Gs = computeReturns(rewards, mdp.gamma);

    // rhoSuffix[t] = ρ_{t:T-1} computed via backward sweep
    const rhoSuffix = new Float64Array(T + 1);
    rhoSuffix[T] = 1.0;
    for (let k = T - 1; k >= 0; k--) {
      const { s, a } = steps[k];
      const pt = target.pi[s][a];
      const pb = behavior.pi[s][a];
      // Use exact integer ratio: 1/(1/4) = 4 rather than 1/0.25 to avoid drift
      rhoSuffix[k] = pb === 0 ? 0 : rhoSuffix[k + 1] * (pt / pb);
    }

    const seen = new Set<number>();
    for (let t = 0; t < T; t++) {
      const s = steps[t].s;
      if (seen.has(s)) continue;
      seen.add(s);
      const w = rhoSuffix[t];
      numArr[s] += w * Gs[t];
      if (weighted) {
        denArr[s] += w;
      } else {
        countArr[s] += 1;  // equals N for s0 since every episode visits it
      }
      sumW[s] += w;
      sumW2[s] += w * w;
    }
  }

  const V = new Float64Array(mdp.nS);
  for (let s = 0; s < mdp.nS; s++) {
    if (weighted) {
      V[s] = denArr[s] === 0 ? 0 : numArr[s] / denArr[s];
    } else {
      V[s] = countArr[s] === 0 ? 0 : numArr[s] / countArr[s];
    }
  }

  const ess = new Float64Array(mdp.nS);
  for (let s = 0; s < mdp.nS; s++) {
    ess[s] = sumW2[s] === 0 ? 0 : (sumW[s] * sumW[s]) / sumW2[s];
  }

  return { V, ess };
}
