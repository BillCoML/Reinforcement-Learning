/**
 * SARSA: on-policy TD control.
 * Target: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]
 * where a' is sampled from the ε-greedy policy at s' — the action the
 * policy will actually take. This makes SARSA converge to Q^π_{ε-soft},
 * not Q*. History tracks V^{π_ε-soft}(start) = E_{ε-greedy}[Q(s0,·)].
 */
import type { MDP } from "../mdp/types";
import {
  sampleStep,
  argmaxQ,
  epsGreedyAction,
  epsSoftValue,
  START_STATE,
  type RNG,
} from "./helpers";

export interface SARSAResult {
  Q: Float64Array;
  greedyPolicy: Int32Array;
  /** V^{π_ε-soft}(start_state) after each episode — converges to ~0.6274. */
  history: Float64Array;
}

export function sarsa(
  mdp: MDP,
  nEpisodes: number,
  epsilon: number | ((episode: number) => number),
  alpha: number,
  options: { rng?: RNG; maxSteps?: number } = {},
): SARSAResult {
  const nA = mdp.nA;
  const Q = new Float64Array(mdp.nS * nA);
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 500;
  const epsAt = typeof epsilon === "number" ? () => epsilon : epsilon;

  for (let ep = 0; ep < nEpisodes; ep++) {
    const eps = epsAt(ep);
    let s = START_STATE;
    // Sample the first action before the loop so s,a are available together
    let a = epsGreedyAction(Q, s, nA, eps, rng);

    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const { sp, r, done } = sampleStep(mdp, s, a, rng);

      // Sample a' exactly once — used both for the target and as the next action.
      // Using a' for the target is what makes SARSA on-policy (not Q-learning).
      const aPrime = done ? 0 : epsGreedyAction(Q, sp, nA, eps, rng);
      const target = done ? r : r + mdp.gamma * Q[sp * nA + aPrime];
      Q[s * nA + a] += alpha * (target - Q[s * nA + a]);

      if (done) break;
      s = sp;
      a = aPrime; // carry a' forward — same sample as above
    }

    history[ep] = epsSoftValue(Q, START_STATE, nA, eps);
  }

  const greedyPolicy = new Int32Array(mdp.nS);
  for (let s = 0; s < mdp.nS; s++) greedyPolicy[s] = argmaxQ(Q, s, nA);
  return { Q, greedyPolicy, history };
}
