/**
 * TD(λ) prediction with backward-view eligibility traces.
 *
 * Trace e(s) decays by γλ per step and bumps by 1 at the current state.
 * Every state's V is updated by α * δ_t * e_t(s) after each transition.
 *
 * λ=0 reduces exactly to TD(0): e is 1 at s_t and 0 elsewhere.
 * λ=1 distributes updates backward through the episode like MC.
 * Traces reset to zero between episodes.
 */
import type { MDP, Policy } from "../mdp/types";
import { samplePolicyAction, sampleStep, START_STATE, type RNG } from "./helpers";

export interface TDLambdaResult {
  V: Float64Array;
  /** V(start_state) after each episode. */
  history: Float64Array;
  /** Eligibility trace array at end of final episode (for visualization). */
  finalTraces: Float64Array;
}

export function tdLambda(
  mdp: MDP,
  policy: Policy,
  lambda: number,
  nEpisodes: number,
  alpha: number,
  options: { rng?: RNG; maxSteps?: number } = {},
): TDLambdaResult {
  const nS = mdp.nS;
  const V = new Float64Array(nS);
  const e = new Float64Array(nS); // eligibility traces
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 500;
  const gammaLambda = mdp.gamma * lambda;

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = START_STATE;
    e.fill(0); // traces must reset between episodes

    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const a = samplePolicyAction(policy, s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);

      const delta = r + (done ? 0 : mdp.gamma * V[sp]) - V[s];

      // Decay all traces, then bump the current state.
      for (let st = 0; st < nS; st++) e[st] *= gammaLambda;
      e[s] += 1;

      // Update every state proportionally to its trace.
      const alphaD = alpha * delta;
      for (let st = 0; st < nS; st++) V[st] += alphaD * e[st];

      if (done) break;
      s = sp;
    }

    history[ep] = V[START_STATE];
  }

  return { V, history, finalTraces: e.slice() };
}
