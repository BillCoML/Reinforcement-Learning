import type { MDP } from "../mdp/types";
import { sampleStep } from "../td/helpers";
import { SoftmaxPolicy } from "./softmax-policy";

/**
 * n-step advantage estimator: Â^π(s_t, a_t) = (Σ_{k=0}^{n-1} γ^k r_{t+k}) + γ^n V_φ(s_{t+n}) - V_φ(s_t).
 *
 * n=1: TD error (low bias, low variance when V_φ ≈ V^π).
 * n=∞: MC return minus baseline (zero bias, high variance).
 */
export function nStepAdvantage(
  rewards: number[],
  states: number[],
  V: Float64Array,
  t: number,
  n: number,
  gamma: number,
  terminals: boolean[],
): number {
  const T = rewards.length;
  let G = 0;
  for (let k = 0; k < n; k++) {
    if (t + k >= T) break;
    G += Math.pow(gamma, k) * rewards[t + k];
  }
  const bootstrapIdx = t + n;
  if (bootstrapIdx < T && !terminals[states[bootstrapIdx]]) {
    G += Math.pow(gamma, n) * V[states[bootstrapIdx]];
  }
  return G - V[states[t]];
}

export interface NStepRMSEResult {
  n: number;
  rmse: number;
  biasSq: number;
  variance: number;
}

/**
 * Compute RMSE of n-step advantage estimator for (s=0, a=RIGHT=1)
 * against the true advantage A^π(0, 1), evaluated under a near-optimal policy.
 *
 * Used by V7 (BiasVarianceAdvantage) and the offline trace script.
 */
export function nStepAdvantageRMSE(
  mdp: MDP,
  policy: SoftmaxPolicy,
  trueAdvantage: number,
  nValues: number[],
  nEpisodes: number,
  rng: () => number,
): NStepRMSEResult[] {
  // Build critic via many TD(0) rollouts
  const V = buildCritic(mdp, policy, 5000, 0.1, rng);

  const results: NStepRMSEResult[] = [];
  for (const n of nValues) {
    const estimates: number[] = [];

    for (let ep = 0; ep < nEpisodes; ep++) {
      // Roll out from s=0, look for action RIGHT at first step
      let s = 0;
      if (mdp.terminals[s]) continue;

      const a = 1; // RIGHT — fixed action for evaluation
      const { sp, r, done } = sampleStep(mdp, s, a, rng);

      // Collect subsequent rewards
      const rewards = [r];
      const stateSeq = [s, sp];
      let cur = sp;
      for (let k = 1; k < n && !done && !mdp.terminals[cur]; k++) {
        const nextA = policy.sample(cur, rng);
        const { sp: nextSp, r: nextR, done: nextDone } = sampleStep(
          mdp,
          cur,
          nextA,
          rng,
        );
        rewards.push(nextR);
        stateSeq.push(nextSp);
        cur = nextSp;
        if (nextDone) break;
      }

      const adv = nStepAdvantage(rewards, stateSeq, V, 0, n, mdp.gamma, mdp.terminals);
      estimates.push(adv);
    }

    if (estimates.length === 0) {
      results.push({ n, rmse: 0, biasSq: 0, variance: 0 });
      continue;
    }

    const mean = estimates.reduce((a, b) => a + b, 0) / estimates.length;
    const bias = mean - trueAdvantage;
    const variance =
      estimates.reduce((a, b) => a + (b - mean) ** 2, 0) / estimates.length;
    const rmse = Math.sqrt(bias * bias + variance);
    results.push({ n, rmse, biasSq: bias * bias, variance });
  }

  return results;
}

/** Build a tabular critic V_φ for the given policy via TD(0). */
function buildCritic(
  mdp: MDP,
  policy: SoftmaxPolicy,
  nEpisodes: number,
  alphaCritic: number,
  rng: () => number,
): Float64Array {
  const V = new Float64Array(mdp.nS);
  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = 0;
    let steps = 0;
    while (!mdp.terminals[s] && steps < 500) {
      const a = policy.sample(s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);
      const vNext = done ? 0 : V[sp];
      V[s] += alphaCritic * (r + mdp.gamma * vNext - V[s]);
      s = sp;
      steps++;
      if (done) break;
    }
  }
  return V;
}
