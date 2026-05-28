import type { MDP } from "../mdp/types";
import { sampleStep, mulberry32 } from "../td/helpers";
import { SoftmaxPolicy } from "./softmax-policy";

export interface ActorCriticResult {
  policy: SoftmaxPolicy;
  /** Tabular critic: V_φ[s] for each state. */
  critic: Float64Array;
  /** Estimated V(s0) after each episode. */
  history: number[];
  /** Per-episode gradient norm ||∇J||. */
  gradNorms: number[];
  /** Per-episode sum of TD errors (proxy for actor update magnitude). */
  episodeReturns: number[];
  /** Total update count across all episodes. */
  updateCount: number;
}

/**
 * One-step actor-critic with tabular critic V_φ (Float64Array).
 *
 * Per step:
 *   δ = r + γ V_φ(s') - V_φ(s)                (TD error = advantage estimate)
 *   V_φ(s) += α_c · δ                           (critic update = TD(0))
 *   θ_{s,:} += α_a · δ · score(s, a)            (actor update)
 *
 * No γ^t scaling on the actor update (standard one-step AC formulation).
 * The critic already absorbs the discount through bootstrapping.
 */
export function actorCritic(
  mdp: MDP,
  nEpisodes: number,
  alphaActor: number,
  alphaCritic: number,
  options: {
    rng?: () => number;
    maxSteps?: number;
  } = {},
): ActorCriticResult {
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 500;

  const policy = new SoftmaxPolicy(mdp.nS, mdp.nA);
  const V = new Float64Array(mdp.nS);
  const history: number[] = [];
  const gradNorms: number[] = [];
  const episodeReturns: number[] = [];
  let updateCount = 0;

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = 0; // start state
    let episodeReturn = 0;
    let gradNormSq = 0;
    let t = 0;

    while (!mdp.terminals[s] && t < maxSteps) {
      const a = policy.sample(s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);

      const vNext = done ? 0 : V[sp];
      const delta = r + mdp.gamma * vNext - V[s];

      // Critic update: TD(0)
      V[s] += alphaCritic * delta;

      // Actor update
      const score = policy.scoreFunction(s, a);
      const base = s * mdp.nA;
      for (let ap = 0; ap < mdp.nA; ap++) {
        const g = alphaActor * delta * score[ap];
        policy.theta[base + ap] += g;
        gradNormSq += g * g;
      }

      episodeReturn += Math.pow(mdp.gamma, t) * r;
      s = sp;
      t++;
      updateCount++;
      if (done) break;
    }

    gradNorms.push(Math.sqrt(gradNormSq));
    episodeReturns.push(episodeReturn);
    history.push(estimateV(mdp, policy, ep));
  }

  return { policy, critic: V, history, gradNorms, episodeReturns, updateCount };
}

/** Estimate V(s0=0) via 20 MC rollouts using an eval-only RNG. */
function estimateV(
  mdp: MDP,
  policy: SoftmaxPolicy,
  episodeIndex: number,
): number {
  const evalRng = mulberry32(0xCAFE0000 + (episodeIndex & 0xFFFF));
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
