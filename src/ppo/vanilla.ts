import type { MDP } from "../mdp/types";
import { SoftmaxPolicy } from "../pg/softmax-policy";
import { sampleStep, mulberry32 } from "../td/helpers";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { gaeAdvantages } from "./gae";
import type { Trajectory, VanillaPGLog } from "./types";

interface VanillaConfig {
  lrPolicy: number;
  lrValue: number;
  gaeLambda: number;
  normalizeAdvantages: boolean;
  batchEpisodes: number;
}

/** Roll out one episode under a flat theta array. */
function rollout(theta: Float64Array, mdp: MDP, rng: () => number): Trajectory {
  const states: number[] = [];
  const actions: number[] = [];
  const rewards: number[] = [];
  let s = 0;
  let steps = 0;

  while (!mdp.terminals[s] && steps < 500) {
    const base = s * mdp.nA;
    let maxVal = -Infinity;
    for (let a = 0; a < mdp.nA; a++) {
      if (theta[base + a] > maxVal) maxVal = theta[base + a];
    }
    let sum = 0;
    const p = new Float64Array(mdp.nA);
    for (let a = 0; a < mdp.nA; a++) {
      p[a] = Math.exp(theta[base + a] - maxVal);
      sum += p[a];
    }
    for (let a = 0; a < mdp.nA; a++) p[a] /= sum;

    const u = rng();
    let acc = 0;
    let action = mdp.nA - 1;
    for (let a = 0; a < mdp.nA; a++) {
      acc += p[a];
      if (u < acc) { action = a; break; }
    }

    const { sp, r } = sampleStep(mdp, s, action, rng);
    states.push(s);
    actions.push(action);
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
 * Vanilla PG update: same GAE advantages as PPO, but no clipping.
 * This isolates the effect of clipping by holding everything else equal.
 */
export function vanillaPGUpdate(
  state: { theta: Float64Array; V: Float64Array },
  batch: readonly Trajectory[],
  config: VanillaConfig,
  mdp: MDP,
): { state: { theta: Float64Array; V: Float64Array }; log: VanillaPGLog } {
  const nS = mdp.nS;
  const nA = mdp.nA;
  const newTheta = state.theta.slice();
  const newV = state.V.slice();

  const allStates: number[] = [];
  const allActions: number[] = [];
  const allAdvantages: number[] = [];
  const allReturns: number[] = [];

  for (const traj of batch) {
    const adv = gaeAdvantages(traj, state.V, mdp.gamma, config.gaeLambda, mdp);
    const T = traj.rewards.length;
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
    }
  }

  const N = allStates.length;
  const advantages = new Float64Array(allAdvantages);
  const returns_ = new Float64Array(allReturns);

  if (config.normalizeAdvantages && N > 1) {
    let mean = 0;
    for (let i = 0; i < N; i++) mean += advantages[i];
    mean /= N;
    let variance = 0;
    for (let i = 0; i < N; i++) variance += (advantages[i] - mean) ** 2;
    variance /= N;
    const std = Math.sqrt(variance) + 1e-8;
    for (let i = 0; i < N; i++) advantages[i] = (advantages[i] - mean) / std;
  }

  // Single gradient pass (no epochs, no clipping).
  const thetaGrad = new Float64Array(nS * nA);
  for (let i = 0; i < N; i++) {
    const s = allStates[i];
    const a = allActions[i];
    const Ahat = advantages[i];
    const base = s * nA;
    let maxVal = -Infinity;
    for (let ap = 0; ap < nA; ap++) {
      if (newTheta[base + ap] > maxVal) maxVal = newTheta[base + ap];
    }
    let sum = 0;
    const p = new Float64Array(nA);
    for (let ap = 0; ap < nA; ap++) {
      p[ap] = Math.exp(newTheta[base + ap] - maxVal);
      sum += p[ap];
    }
    for (let ap = 0; ap < nA; ap++) p[ap] /= sum;
    for (let ap = 0; ap < nA; ap++) {
      thetaGrad[base + ap] += Ahat * ((ap === a ? 1 : 0) - p[ap]);
    }
  }
  for (let idx = 0; idx < nS * nA; idx++) {
    newTheta[idx] += config.lrPolicy * thetaGrad[idx] / N;
  }

  // Critic update.
  for (let i = 0; i < N; i++) {
    const s = allStates[i];
    newV[s] += config.lrValue * (returns_[i] - newV[s]);
  }
  for (let s = 0; s < nS; s++) if (mdp.terminals[s]) newV[s] = 0;

  const finalTheta = new Float64Array(newTheta);
  const finalV = new Float64Array(newV);

  const pol = new SoftmaxPolicy(nS, nA);
  finalTheta.forEach((v, i) => { pol.theta[i] = v; });
  const Vexact = policyEvaluationExact(mdp, pol.toPolicy());
  const vStart = Vexact[0];

  return {
    state: { theta: finalTheta, V: finalV },
    log: { iter: 0, vStart, batchSize: N },
  };
}

/** Run vanilla PG for nIters iterations. */
export function runVanilla(
  config: VanillaConfig,
  nIters: number,
  seed: number,
  mdp: MDP,
): { finalState: { theta: Float64Array; V: Float64Array }; logs: readonly VanillaPGLog[] } {
  const rng = mulberry32(seed);
  let state = {
    theta: new Float64Array(mdp.nS * mdp.nA),
    V: new Float64Array(mdp.nS),
  };

  const logs: VanillaPGLog[] = [];

  for (let iter = 0; iter < nIters; iter++) {
    const batch: Trajectory[] = [];
    for (let ep = 0; ep < config.batchEpisodes; ep++) {
      batch.push(rollout(state.theta, mdp, rng));
    }
    const { state: newState, log } = vanillaPGUpdate(state, batch, config, mdp);
    logs.push({ ...log, iter });
    state = newState as typeof state;
  }

  return { finalState: state, logs };
}
