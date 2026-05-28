import { mulberry32 } from "../importance-sampling/gaussian";
import type { MDP } from "../mdp/types";
import type { MonteCarloDiagnostic } from "./types";

const HIST_BINS = 50;

/**
 * Monte Carlo diagnostic: roll out the policy from state 0 and count outcomes.
 * Uses a seeded PRNG for reproducibility — all randomness (action draws AND
 * transition draws for stochastic MDPs) flows through the same RNG.
 */
export function monteCarloDiagnostic(
  pi: Float64Array,   // flat nS*nA Boltzmann policy
  mdp: MDP,
  options: { nRollouts: number; maxSteps: number; seed: number },
): MonteCarloDiagnostic {
  const { nRollouts, maxSteps, seed } = options;
  const { nS, nA } = mdp;
  const rng = mulberry32(seed);

  let goalCount = 0, pitCount = 0, timeoutCount = 0;
  let terminatedSteps = 0, terminatedCount = 0;
  const histBuckets = new Float64Array(HIST_BINS);
  const binWidth = maxSteps / HIST_BINS;

  // Identify goal and pit from terminal rewards.
  // Goal: terminal state where entering gives +1 reward from some action.
  // Pit: terminal state where entering gives -1 reward.
  // We infer by looking at reward of transitions into each terminal.
  const goalState = findTerminalByReward(mdp, +1);
  const pitState = findTerminalByReward(mdp, -1);

  for (let roll = 0; roll < nRollouts; roll++) {
    let s = 0;  // start at (0,0)
    let steps = 0;
    let terminated = false;

    while (steps < maxSteps) {
      if (mdp.terminals[s]) {
        terminated = true;
        break;
      }

      // Sample action from pi[s]
      const base = s * nA;
      let u = rng();
      let acc = 0;
      let a = nA - 1;
      for (let ai = 0; ai < nA; ai++) {
        acc += pi[base + ai];
        if (u <= acc) { a = ai; break; }
      }

      // Sample next state from P[s][a]
      u = rng();
      acc = 0;
      let sp = nS - 1;
      const row = mdp.P[s][a];
      for (let si = 0; si < nS; si++) {
        acc += row[si];
        if (u <= acc) { sp = si; break; }
      }

      s = sp;
      steps++;
    }

    if (terminated) {
      if (goalState >= 0 && s === goalState) goalCount++;
      else if (pitState >= 0 && s === pitState) pitCount++;
      else timeoutCount++;
      terminatedSteps += steps;
      terminatedCount++;
    } else {
      timeoutCount++;
    }

    const bin = Math.min(Math.floor(steps / binWidth), HIST_BINS - 1);
    histBuckets[bin]++;
  }

  // Normalize histogram to probabilities
  for (let i = 0; i < HIST_BINS; i++) histBuckets[i] /= nRollouts;

  // meanStepsToTerminal: mean over rollouts that DID reach a terminal (goal or pit).
  // Excludes timeouts so the metric reflects actual traversal time, not the cap.
  const meanStepsToTerminal = terminatedCount > 0
    ? terminatedSteps / terminatedCount
    : maxSteps;

  return {
    goalReachProb: goalCount / nRollouts,
    pitReachProb: pitCount / nRollouts,
    timeoutProb: timeoutCount / nRollouts,
    meanStepsToTerminal,
    lengthHistogram: histBuckets,
  };
}

function findTerminalByReward(mdp: MDP, targetReward: number): number {
  for (let s = 0; s < mdp.nS; s++) {
    if (!mdp.terminals[s]) continue;
    // Check if any non-terminal state transitions into s with the target reward.
    for (let src = 0; src < mdp.nS; src++) {
      if (mdp.terminals[src]) continue;
      for (let a = 0; a < mdp.nA; a++) {
        if (mdp.P[src][a][s] > 0 && Math.abs(mdp.r[src][a] - targetReward) < 0.01) return s;
      }
    }
  }
  return -1;
}
