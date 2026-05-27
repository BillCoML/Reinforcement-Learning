/**
 * n-step TD prediction: look ahead n steps before bootstrapping.
 * Interpolates between TD(0) (n=1) and MC (n=∞).
 *
 * Implementation uses a buffer: defer the update for state s_τ until
 * n steps have accumulated past it. At episode end, flush remaining
 * states with partial MC returns (no bootstrap, since the future is
 * fully observed).
 *
 * n=1 must produce identical output to tdZero given the same seeded RNG,
 * because both sample actions and transitions in the same order.
 */
import type { MDP, Policy } from "../mdp/types";
import { samplePolicyAction, sampleStep, START_STATE, type RNG } from "./helpers";

export interface NStepTDResult {
  V: Float64Array;
  /** V(start_state) after each episode. */
  history: Float64Array;
}

export function nStepTD(
  mdp: MDP,
  policy: Policy,
  n: number,
  nEpisodes: number,
  alpha: number,
  options: { rng?: RNG; maxSteps?: number } = {},
): NStepTDResult {
  const V = new Float64Array(mdp.nS);
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 1000;

  // Precompute γ^k for k = 0..n to avoid repeated Math.pow calls.
  const gammaPow = new Float64Array(n + 1);
  gammaPow[0] = 1;
  for (let k = 1; k <= n; k++) gammaPow[k] = gammaPow[k - 1] * mdp.gamma;

  for (let ep = 0; ep < nEpisodes; ep++) {
    // Trajectory buffers for the current episode.
    const states: number[] = [START_STATE];
    const rewards: number[] = [];
    let T = Infinity; // episode length (unknown until terminal)
    let t = 0;

    while (true) {
      // Advance the trajectory by one step if the episode is still running.
      if (t < T) {
        const s = states[t];
        if (mdp.terminals[s]) {
          T = t;
        } else {
          const a = samplePolicyAction(policy, s, rng);
          const { sp, r, done } = sampleStep(mdp, s, a, rng);
          rewards.push(r);
          states.push(sp);
          if (done) T = t + 1;
        }
      }

      // Compute and apply the n-step return for state τ = t − n + 1.
      const tau = t - n + 1;
      if (tau >= 0 && tau < T) {
        let G = 0;
        const limit = Math.min(tau + n, T);
        for (let k = tau + 1; k <= limit; k++) {
          G += gammaPow[k - tau - 1] * rewards[k - 1];
        }
        // Bootstrap only if the episode extends beyond τ + n.
        if (tau + n < T) {
          G += gammaPow[n] * V[states[tau + n]];
        }
        V[states[tau]] += alpha * (G - V[states[tau]]);
      }

      if (tau === T - 1) break;
      t++;
      if (t > maxSteps + n) break; // safety against pathological MDPs
    }

    history[ep] = V[START_STATE];
  }

  return { V, history };
}
