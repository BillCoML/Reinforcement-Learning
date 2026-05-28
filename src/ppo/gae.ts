import type { MDP } from "../mdp/types";
import type { Trajectory } from "./types";

/**
 * Generalized Advantage Estimation (GAE-λ).
 *
 * Â_t^GAE(λ) = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
 * where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
 *
 * At λ=0: Â_t = δ_t (one-step TD residual).
 * At λ=1: Â_t = G_t - V(s_t) (MC advantage, telescoping sum).
 *
 * Computed backwards from the end of the trajectory.
 */
export function gaeAdvantages(
  traj: Trajectory,
  V: Float64Array,
  gamma: number,
  lambda: number,
  mdp: MDP,
): Float64Array {
  const T = traj.rewards.length;
  const adv = new Float64Array(T);
  let lastGae = 0;

  for (let t = T - 1; t >= 0; t--) {
    const s = traj.states[t];
    const sNext = t + 1 < traj.states.length ? traj.states[t + 1] : -1;
    const vNext = sNext >= 0 && !mdp.terminals[sNext] ? V[sNext] : 0;
    const delta = traj.rewards[t] + gamma * vNext - V[s];
    lastGae = delta + gamma * lambda * lastGae;
    adv[t] = lastGae;
  }

  return adv;
}

/** Discounted returns G_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k} */
export function discountedReturns(rewards: number[], gamma: number): Float64Array {
  const T = rewards.length;
  const G = new Float64Array(T);
  let running = 0;
  for (let t = T - 1; t >= 0; t--) {
    running = rewards[t] + gamma * running;
    G[t] = running;
  }
  return G;
}
