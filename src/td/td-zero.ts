/**
 * TD(0) prediction: one-step bootstrap estimate of V^π.
 * Updates V(s) ← V(s) + α[r + γV(s') − V(s)] after each transition.
 */
import type { MDP, Policy } from "../mdp/types";
import { samplePolicyAction, sampleStep, START_STATE, type RNG } from "./helpers";

export interface TDZeroResult {
  V: Float64Array;
  /** V(start_state) after each episode. */
  history: Float64Array;
}

/**
 * TD(0) prediction under a fixed policy.
 *
 * @param alpha - Constant step size, or a function (episode, state) → α
 *   for Robbins-Monro schedules (e.g. 1/visit_count).
 */
export function tdZero(
  mdp: MDP,
  policy: Policy,
  nEpisodes: number,
  alpha: number | ((episode: number, state: number) => number),
  options: { rng?: RNG; maxSteps?: number } = {},
): TDZeroResult {
  const V = new Float64Array(mdp.nS);
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 500;

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = START_STATE;
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const a = samplePolicyAction(policy, s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);
      const aEff = typeof alpha === "number" ? alpha : alpha(ep, s);
      const target = done ? r : r + mdp.gamma * V[sp];
      V[s] += aEff * (target - V[s]);
      if (done) break;
      s = sp;
    }
    history[ep] = V[START_STATE];
  }

  return { V, history };
}
