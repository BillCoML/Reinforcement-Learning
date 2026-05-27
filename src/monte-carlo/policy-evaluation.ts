import type { MDP, Policy } from "../mdp/types";
import { rollout } from "../mdp/rollout";
import { computeReturns } from "./returns";

export interface MCEvalResult {
  V: Float64Array;
  visits: Int32Array;
}

/**
 * First-visit or every-visit Monte Carlo policy evaluation.
 * Starts every episode from s0 (default 0 = cell (0,0)).
 */
export function mcPolicyEvaluation(
  mdp: MDP,
  policy: Policy,
  nEpisodes: number,
  options: {
    firstVisit?: boolean;
    s0?: number;
    maxSteps?: number;
    rng?: () => number;
  } = {},
): MCEvalResult {
  const firstVisit = options.firstVisit ?? true;
  const s0 = options.s0 ?? 0;
  const maxSteps = options.maxSteps ?? 200;
  const rng = options.rng ?? Math.random;

  const V = new Float64Array(mdp.nS);
  const visits = new Int32Array(mdp.nS);

  for (let ep = 0; ep < nEpisodes; ep++) {
    const steps = rollout(mdp, policy, s0, maxSteps, rng);
    if (steps.length === 0) continue;
    const rewards = steps.map(st => st.r);
    const Gs = computeReturns(rewards, mdp.gamma);
    const seen = new Set<number>();
    for (let t = 0; t < steps.length; t++) {
      const s = steps[t].s;
      if (firstVisit && seen.has(s)) continue;
      seen.add(s);
      visits[s] += 1;
      V[s] += (Gs[t] - V[s]) / visits[s];
    }
  }
  return { V, visits };
}
