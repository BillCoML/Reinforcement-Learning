import type { MDP } from "../mdp/types";
import { sampleStep, mulberry32 } from "../td/helpers";
import { computeReturns } from "../monte-carlo/returns";
import { SoftmaxPolicy } from "./softmax-policy";

export interface ReinforceResult {
  policy: SoftmaxPolicy;
  /** Estimated V(s0) via MC rollouts recorded after each episode. */
  history: number[];
  /** Per-episode gradient norm ‖∇J‖. */
  gradNorms: number[];
  /** Per-episode total discounted return. */
  episodeReturns: number[];
}

/**
 * REINFORCE (Williams 1992): Monte Carlo policy gradient.
 *
 * Update: θ += α · γ^t · (G_t − b(s_t)) · ∇_θ log π_θ(a_t | s_t)
 *
 * Baseline options (set useBaseline=true):
 *   alphaCritic > 0  → TD(0) critic updated per step, used as b(s_t) = V_φ(s_t).
 *                      This is the per-state state-value baseline from §5 of the README
 *                      and gives ~5–7× variance reduction on the 3×3 gridworld.
 *   alphaCritic = 0  → Running-mean of G_0 (global). Simple but weaker reduction.
 *
 * NOTE: with alphaCritic > 0, the critic is updated DURING the episode (per-step TD),
 * but the POLICY is updated only at the EPISODE END using G_t (full MC return). This
 * keeps the policy update unbiased (MC return) while the critic provides a good
 * baseline. This is different from actor-critic, where both actor and critic are
 * updated per step using the TD error.
 */
export function reinforce(
  mdp: MDP,
  nEpisodes: number,
  alpha: number,
  options: {
    rng?: () => number;
    useBaseline?: boolean;
    alphaCritic?: number;
    maxSteps?: number;
  } = {},
): ReinforceResult {
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 500;
  const useBaseline = options.useBaseline ?? false;
  const alphaCritic = options.alphaCritic ?? 0.1;

  const policy = new SoftmaxPolicy(mdp.nS, mdp.nA);
  const history: number[] = [];
  const gradNorms: number[] = [];
  const episodeReturns: number[] = [];

  // TD critic for the baseline (only used when useBaseline=true and alphaCritic>0).
  const V = new Float64Array(mdp.nS);
  // Global running-mean fallback (when alphaCritic=0).
  let runningMean = 0;
  let seenCount = 0;

  for (let ep = 0; ep < nEpisodes; ep++) {
    // ── Roll out one episode, updating TD critic per-step ────────────────
    const states: number[] = [];
    const actions: number[] = [];
    const rewards: number[] = [];

    let s = 0; // start state (0,0)
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const a = policy.sample(s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);
      states.push(s);
      actions.push(a);
      rewards.push(r);

      // TD critic update (per step, if baseline requested)
      if (useBaseline && alphaCritic > 0) {
        const vNext = done ? 0 : V[sp];
        V[s] += alphaCritic * (r + mdp.gamma * vNext - V[s]);
      }

      s = sp;
      if (done) break;
    }

    const T = states.length;
    if (T === 0) {
      history.push(estimateV(mdp, policy, ep));
      gradNorms.push(0);
      episodeReturns.push(0);
      continue;
    }

    // ── Compute reward-to-go G_t ──────────────────────────────────────────
    const Gs = computeReturns(rewards, mdp.gamma);
    episodeReturns.push(Gs[0]);

    // ── Update global running-mean fallback ───────────────────────────────
    if (useBaseline && alphaCritic === 0) {
      seenCount++;
      runningMean += (Gs[0] - runningMean) / seenCount;
    }

    // ── Policy gradient update ────────────────────────────────────────────
    let gradNormSq = 0;

    for (let t = 0; t < T; t++) {
      const st = states[t];
      const at = actions[t];

      let b = 0;
      if (useBaseline) {
        b = alphaCritic > 0 ? V[st] : runningMean;
      }
      const advantage = Gs[t] - b;
      const gammaT = Math.pow(mdp.gamma, t);
      const score = policy.scoreFunction(st, at);
      const base = st * mdp.nA;
      for (let ap = 0; ap < mdp.nA; ap++) {
        const delta = alpha * gammaT * advantage * score[ap];
        policy.theta[base + ap] += delta;
        gradNormSq += delta * delta;
      }
    }

    gradNorms.push(Math.sqrt(gradNormSq));
    history.push(estimateV(mdp, policy, ep));
  }

  return { policy, history, gradNorms, episodeReturns };
}

/**
 * Estimate V(s0=0) via MC rollouts using an eval-only RNG (independent of
 * the training stream).
 */
export function estimateV(
  mdp: MDP,
  policy: SoftmaxPolicy,
  episodeIndex: number,
): number {
  const evalRng = mulberry32(0xDEAD0000 + (episodeIndex & 0xFFFF));
  const nRollouts = 20;
  let total = 0;
  for (let i = 0; i < nRollouts; i++) {
    let s = 0;
    let G = 0;
    let t = 0;
    while (!mdp.terminals[s] && t < 200) {
      const a = policy.sample(s, evalRng);
      const { sp, r } = sampleStep(mdp, s, a, evalRng);
      G += Math.pow(mdp.gamma, t) * r;
      s = sp;
      t++;
    }
    total += G;
  }
  return total / nRollouts;
}

/**
 * Compute gradient norms at a fixed policy (θ=0, uniform) over nEpisodes.
 * Used by V5 variance-reduction histogram.
 * The policy is NOT updated — this isolates estimator variance from policy variance.
 */
export function gradientVarianceAtUniform(
  mdp: MDP,
  nEpisodes: number,
  rng: () => number,
  maxSteps = 200,
): { vanilla: number[]; baseline: number[] } {
  const policy = new SoftmaxPolicy(mdp.nS, mdp.nA); // theta stays 0

  // Build a TD critic for the baseline by running TD(0) separately.
  const V = new Float64Array(mdp.nS);
  const alphaCritic = 0.1;

  // Collect all trajectories + update critic
  const trajectories: { states: number[]; actions: number[]; Gs: number[] }[] = [];
  const rngCopy = mulberry32(0x5EED); // separate stream for critic pre-training

  for (let ep = 0; ep < 2000; ep++) {
    let s = 0;
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const a = policy.sample(s, rngCopy);
      const { sp, r, done } = sampleStep(mdp, s, a, rngCopy);
      const vNext = done ? 0 : V[sp];
      V[s] += alphaCritic * (r + 0.9 * vNext - V[s]);
      s = sp;
      if (done) break;
    }
  }

  // Now collect main evaluation trajectories under the uniform policy
  for (let ep = 0; ep < nEpisodes; ep++) {
    const states: number[] = [];
    const actions: number[] = [];
    const rewards: number[] = [];
    let s = 0;
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.terminals[s]) break;
      const a = policy.sample(s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);
      states.push(s);
      actions.push(a);
      rewards.push(r);
      s = sp;
      if (done) break;
    }
    const Gs = states.length > 0 ? computeReturns(rewards, mdp.gamma) : [0];
    trajectories.push({ states, actions, Gs });
  }

  const vanilla: number[] = [];
  const baselineNorms: number[] = [];

  for (const { states, actions, Gs } of trajectories) {
    const T = states.length;
    let normVanilla = 0;
    let normBaseline = 0;
    for (let t = 0; t < T; t++) {
      const score = policy.scoreFunction(states[t], actions[t]);
      const gammaT = Math.pow(mdp.gamma, t);
      const bst = V[states[t]];
      for (let ap = 0; ap < mdp.nA; ap++) {
        const gv = gammaT * Gs[t] * score[ap];
        const gb = gammaT * (Gs[t] - bst) * score[ap];
        normVanilla += gv * gv;
        normBaseline += gb * gb;
      }
    }
    vanilla.push(Math.sqrt(normVanilla));
    baselineNorms.push(Math.sqrt(normBaseline));
  }

  return { vanilla, baseline: baselineNorms };
}
