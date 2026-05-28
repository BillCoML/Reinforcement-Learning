import type { MDP } from "../mdp/types";
import { SoftmaxPolicy } from "../pg/softmax-policy";
import { sampleStep, mulberry32 } from "../td/helpers";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { gaeAdvantages } from "./gae";
import type { PPOConfig, PPOState, PPOIterationLog, Trajectory } from "./types";

/** Numerically stable softmax over a logit slice of length nActions. */
export function softmax(logits: Float64Array, nActions: number, offset = 0): Float64Array {
  const p = new Float64Array(nActions);
  let maxVal = -Infinity;
  for (let a = 0; a < nActions; a++) {
    if (logits[offset + a] > maxVal) maxVal = logits[offset + a];
  }
  let sum = 0;
  for (let a = 0; a < nActions; a++) {
    p[a] = Math.exp(logits[offset + a] - maxVal);
    sum += p[a];
  }
  for (let a = 0; a < nActions; a++) p[a] /= sum;
  return p;
}

/** Sample action from softmax distribution using uniform draw u in [0,1). */
function sampleAction(probs: Float64Array, u: number): number {
  let acc = 0;
  for (let a = 0; a < probs.length; a++) {
    acc += probs[a];
    if (u < acc) return a;
  }
  return probs.length - 1;
}

/** Roll out one episode under current policy. Max 500 steps. */
function rollout(
  theta: Float64Array,
  mdp: MDP,
  rng: () => number,
): Trajectory {
  const states: number[] = [];
  const actions: number[] = [];
  const rewards: number[] = [];
  let s = 0;
  let steps = 0;

  while (!mdp.terminals[s] && steps < 500) {
    const base = s * mdp.nA;
    const probs = softmax(theta, mdp.nA, base);
    const a = sampleAction(probs, rng());
    const { sp, r } = sampleStep(mdp, s, a, rng);
    states.push(s);
    actions.push(a);
    rewards.push(r);
    s = sp;
    steps++;
    if (mdp.terminals[sp]) {
      states.push(sp);
      break;
    }
  }

  return { states, actions, rewards };
}

/**
 * KL divergence KL(π_old ‖ π_new) averaged over visited states.
 * visitedStates: states seen in the batch (not including terminal absorbing states).
 */
export function policyKL(
  thetaOld: Float64Array,
  thetaNew: Float64Array,
  visitedStates: ReadonlySet<number>,
  nActions: number,
): number {
  let totalKL = 0;
  for (const s of visitedStates) {
    const base = s * nActions;
    const pOld = softmax(thetaOld, nActions, base);
    const pNew = softmax(thetaNew, nActions, base);
    let kl = 0;
    for (let a = 0; a < nActions; a++) {
      if (pOld[a] > 1e-12) kl += pOld[a] * Math.log(pOld[a] / (pNew[a] + 1e-12));
    }
    totalKL += kl;
  }
  return visitedStates.size > 0 ? totalKL / visitedStates.size : 0;
}

/** Entropy H[π(·|s)] for a single state. */
function stateEntropy(theta: Float64Array, s: number, nActions: number): number {
  const base = s * nActions;
  const p = softmax(theta, nActions, base);
  let h = 0;
  for (let a = 0; a < nActions; a++) {
    if (p[a] > 1e-12) h -= p[a] * Math.log(p[a]);
  }
  return h;
}

/**
 * One PPO iteration:
 *  1. Collect batchEpisodes trajectories.
 *  2. Compute GAE advantages.
 *  3. Normalize advantages (per-batch).
 *  4. Run `epochs` gradient steps using clipped surrogate loss.
 *  5. Update critic via MSE on returns.
 */
export function ppoUpdate(
  state: PPOState,
  batch: readonly Trajectory[],
  config: PPOConfig,
  mdp: MDP,
): { state: PPOState; log: PPOIterationLog; ratios: Float64Array } {
  const nS = mdp.nS;
  const nA = mdp.nA;

  const thetaOld = state.theta.slice();
  const newTheta = state.theta.slice();
  const newV = state.V.slice();

  // Collect all transitions from the batch.
  const allStates: number[] = [];
  const allActions: number[] = [];
  const allAdvantages: number[] = [];
  const allReturns: number[] = [];
  const visitedStates = new Set<number>();

  for (const traj of batch) {
    const adv = gaeAdvantages(traj, state.V, mdp.gamma, config.gaeLambda, mdp);
    const T = traj.rewards.length;
    // Discounted return for critic target
    let G = 0;
    const returns: number[] = new Array(T);
    for (let t = T - 1; t >= 0; t--) {
      G = traj.rewards[t] + mdp.gamma * G;
      returns[t] = G;
    }
    for (let t = 0; t < T; t++) {
      allStates.push(traj.states[t]);
      allActions.push(traj.actions[t]);
      allAdvantages.push(adv[t]);
      allReturns.push(returns[t]);
      visitedStates.add(traj.states[t]);
    }
  }

  const N = allStates.length;

  // Normalize advantages per-batch.
  if (config.normalizeAdvantages && N > 1) {
    let mean = 0;
    for (let i = 0; i < N; i++) mean += allAdvantages[i];
    mean /= N;
    let variance = 0;
    for (let i = 0; i < N; i++) variance += (allAdvantages[i] - mean) ** 2;
    variance /= N;
    const std = Math.sqrt(variance) + 1e-8;
    for (let i = 0; i < N; i++) allAdvantages[i] = (allAdvantages[i] - mean) / std;
  }

  // Convert advantages to typed array for consistency.
  const advantages = new Float64Array(allAdvantages);
  const returns_ = new Float64Array(allReturns);

  // Policy gradient epochs with clipped surrogate.
  let lastEpochClipCount = 0;
  let lastEpochTotal = 0;
  let lastEpochSurrogate = 0;
  let lastEpochRatios: Float64Array = new Float64Array(N).fill(1);

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    let clipCount = 0;
    let surrogateSum = 0;
    const epochRatios = new Float64Array(N);

    // Gradient accumulator for actor.
    const thetaGrad = new Float64Array(nS * nA);

    for (let i = 0; i < N; i++) {
      const s = allStates[i];
      const a = allActions[i];
      const Ahat = advantages[i];
      const base = s * nA;

      // Current and old policy probabilities.
      const pNew = softmax(newTheta, nA, base);
      const pOld = softmax(thetaOld, nA, base);

      const ratio = pNew[a] / (pOld[a] + 1e-12);
      epochRatios[i] = ratio;

      const rClipped = Math.max(1 - config.clipEps, Math.min(1 + config.clipEps, ratio));
      const obj1 = ratio * Ahat;
      const obj2 = rClipped * Ahat;
      const minObj = Math.min(obj1, obj2);
      surrogateSum += minObj;

      // Determine if clipped (gradient blocked).
      const isClipped =
        (Ahat > 0 && ratio > 1 + config.clipEps) ||
        (Ahat < 0 && ratio < 1 - config.clipEps);

      if (isClipped) {
        clipCount++;
        continue; // gradient = 0 for clipped samples
      }

      // Score function: ∂/∂θ_{s,a'} log π(a|s) = 1[a=a'] - π(a'|s)
      for (let ap = 0; ap < nA; ap++) {
        thetaGrad[base + ap] += Ahat * ((ap === a ? 1 : 0) - pNew[ap]);
      }
    }

    // Apply actor gradient (ascent).
    for (let idx = 0; idx < nS * nA; idx++) {
      newTheta[idx] += config.lrPolicy * thetaGrad[idx] / N;
    }

    lastEpochClipCount = clipCount;
    lastEpochTotal = N;
    lastEpochSurrogate = surrogateSum / N;
    lastEpochRatios = epochRatios;
  }

  // Critic update: MSE loss, one pass after all policy epochs.
  for (let i = 0; i < N; i++) {
    const s = allStates[i];
    const td = returns_[i] - newV[s];
    newV[s] += config.lrValue * td;
  }
  // Zero out terminal states.
  for (let s = 0; s < nS; s++) if (mdp.terminals[s]) newV[s] = 0;

  const finalTheta = new Float64Array(newTheta);
  const finalV = new Float64Array(newV);

  // Compute post-update metrics.
  const meanKL = policyKL(thetaOld, finalTheta, visitedStates, nA);
  const clipFraction = lastEpochTotal > 0 ? lastEpochClipCount / lastEpochTotal : 0;

  let meanRatio = 0, maxRatio = -Infinity, minRatio = Infinity;
  for (let i = 0; i < lastEpochRatios.length; i++) {
    meanRatio += lastEpochRatios[i];
    if (lastEpochRatios[i] > maxRatio) maxRatio = lastEpochRatios[i];
    if (lastEpochRatios[i] < minRatio) minRatio = lastEpochRatios[i];
  }
  meanRatio /= lastEpochRatios.length || 1;

  let entropySum = 0;
  for (const s of visitedStates) entropySum += stateEntropy(finalTheta, s, nA);
  const entropyMean = visitedStates.size > 0 ? entropySum / visitedStates.size : 0;

  // Exact V(start) via policy evaluation.
  const pol = new SoftmaxPolicy(nS, nA);
  finalTheta.forEach((v, i) => { pol.theta[i] = v; });
  const Vexact = policyEvaluationExact(mdp, pol.toPolicy());
  const vStart = Vexact[0];

  const log: PPOIterationLog = {
    iter: 0, // caller fills this in
    vStart,
    meanKL,
    clipFraction,
    meanRatio,
    maxRatio: isFinite(maxRatio) ? maxRatio : 1,
    minRatio: isFinite(minRatio) ? minRatio : 1,
    surrogateValue: lastEpochSurrogate,
    entropyMean,
    batchSize: N,
  };

  return {
    state: { theta: finalTheta, V: finalV },
    log,
    ratios: lastEpochRatios,
  };
}

/** Run PPO for nIters iterations. Returns full log history. */
export function runPPO(
  config: PPOConfig,
  nIters: number,
  seed: number,
  mdp: MDP,
): { finalState: PPOState; logs: readonly PPOIterationLog[] } {
  const rng = mulberry32(seed);

  let state: PPOState = {
    theta: new Float64Array(mdp.nS * mdp.nA),
    V: new Float64Array(mdp.nS),
  };

  const logs: PPOIterationLog[] = [];

  for (let iter = 0; iter < nIters; iter++) {
    const batch: Trajectory[] = [];
    for (let ep = 0; ep < config.batchEpisodes; ep++) {
      batch.push(rollout(state.theta, mdp, rng));
    }
    const { state: newState, log } = ppoUpdate(state, batch, config, mdp);
    logs.push({ ...log, iter });
    state = newState;
  }

  return { finalState: state, logs };
}

/** Collect a batch of trajectories for use with ppoUpdate externally (e.g., PPOLab). */
export function collectBatch(
  theta: Float64Array,
  mdp: MDP,
  batchEpisodes: number,
  rng: () => number,
): Trajectory[] {
  const batch: Trajectory[] = [];
  for (let ep = 0; ep < batchEpisodes; ep++) {
    batch.push(rollout(theta, mdp, rng));
  }
  return batch;
}
