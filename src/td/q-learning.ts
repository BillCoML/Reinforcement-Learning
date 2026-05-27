/**
 * Q-learning: off-policy TD control.
 * Target: Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]
 * The max sidesteps importance sampling entirely. Converges to Q* regardless
 * of the behavior policy (ε-greedy here), as long as all (s,a) are visited.
 * History tracks max_a Q(s0,a) — converges to V*(s0) ≈ 0.7290.
 */
import type { MDP } from "../mdp/types";
import {
  sampleStep,
  argmaxQ,
  maxQ,
  epsGreedyAction,
  START_STATE,
  type RNG,
} from "./helpers";

export interface QLearningResult {
  Q: Float64Array;
  greedyPolicy: Int32Array;
  /** max_a Q(start_state, a) after each episode — converges to V*(s0) ≈ 0.7290. */
  history: Float64Array;
}

export function qLearning(
  mdp: MDP,
  nEpisodes: number,
  epsilon: number | ((episode: number) => number),
  alpha: number,
  options: { rng?: RNG; maxSteps?: number } = {},
): QLearningResult {
  const nA = mdp.nA;
  const Q = new Float64Array(mdp.nS * nA);
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 500;
  const epsAt = typeof epsilon === "number" ? () => epsilon : epsilon;

  for (let ep = 0; ep < nEpisodes; ep++) {
    const eps = epsAt(ep);
    let s = START_STATE;

    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const a = epsGreedyAction(Q, s, nA, eps, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);
      // Off-policy target: uses max over next actions, not the sampled next action.
      const target = done ? r : r + mdp.gamma * maxQ(Q, sp, nA);
      Q[s * nA + a] += alpha * (target - Q[s * nA + a]);
      if (done) break;
      s = sp;
    }

    history[ep] = maxQ(Q, START_STATE, nA);
  }

  const greedyPolicy = new Int32Array(mdp.nS);
  for (let s = 0; s < mdp.nS; s++) greedyPolicy[s] = argmaxQ(Q, s, nA);
  return { Q, greedyPolicy, history };
}
