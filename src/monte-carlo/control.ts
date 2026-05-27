import type { MDP, Policy } from "../mdp/types";
import { rollout } from "../mdp/rollout";
import { computeReturns } from "./returns";

export interface MCControlResult {
  Q: Float64Array;
  policy: Int32Array;
}

function sampleFrom(probs: number[], u: number): number {
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return probs.length - 1;
}

function argmaxQ(Q: Float64Array, s: number, nA: number): number {
  let bestA = 0;
  let bestQ = Q[s * nA];
  for (let a = 1; a < nA; a++) {
    if (Q[s * nA + a] > bestQ) {
      bestQ = Q[s * nA + a];
      bestA = a;
    }
  }
  return bestA;
}

function greedyPolicy(mdp: MDP, greedyActions: Int32Array): Policy {
  const pi = Array.from({ length: mdp.nS }, (_, s) => {
    const row = new Array<number>(mdp.nA).fill(0);
    if (mdp.terminals[s]) {
      row.fill(1 / mdp.nA);
    } else {
      row[greedyActions[s]] = 1;
    }
    return row;
  });
  return { pi };
}

function epsGreedyPolicy(mdp: MDP, greedyActions: Int32Array, epsilon: number): Policy {
  const pi = Array.from({ length: mdp.nS }, (_, s) => {
    const row = new Array<number>(mdp.nA).fill(epsilon / mdp.nA);
    if (!mdp.terminals[s]) {
      row[greedyActions[s]] += 1 - epsilon;
    }
    return row;
  });
  return { pi };
}

/**
 * MC Control with Exploring Starts.
 * Converges to Q* — requires that every (s, a) is the starting pair in some episode.
 */
export function mcControlExploringStarts(
  mdp: MDP,
  nEpisodes: number,
  options: { rng?: () => number } = {},
): MCControlResult {
  const rng = options.rng ?? Math.random;
  const { nS, nA } = mdp;

  const Q = new Float64Array(nS * nA).fill(0);
  const visitCounts = new Int32Array(nS * nA).fill(0);
  const greedyActions = new Int32Array(nS).fill(0);

  const nonTermStates: number[] = [];
  for (let s = 0; s < nS; s++) {
    if (!mdp.terminals[s]) nonTermStates.push(s);
  }

  for (let ep = 0; ep < nEpisodes; ep++) {
    // Exploring starts: pick random non-terminal (s0, a0)
    const s0 = nonTermStates[Math.floor(rng() * nonTermStates.length)];
    const a0 = Math.floor(rng() * nA);

    // Forced first step
    const sp0 = sampleFrom(mdp.P[s0][a0], rng());
    const r0 = mdp.r[s0][a0];

    // Remainder follows greedy policy
    const gPolicy = greedyPolicy(mdp, greedyActions);
    const restSteps = rollout(mdp, gPolicy, sp0, 199, rng);

    // Full episode
    const fullSteps = [{ s: s0, a: a0, r: r0, sp: sp0 }, ...restSteps];
    const rewards = fullSteps.map(st => st.r);
    const Gs = computeReturns(rewards, mdp.gamma);

    // First-visit Q updates
    const seen = new Set<number>();
    for (let t = 0; t < fullSteps.length; t++) {
      const { s, a } = fullSteps[t];
      if (mdp.terminals[s]) continue;
      const key = s * nA + a;
      if (seen.has(key)) continue;
      seen.add(key);
      visitCounts[key] += 1;
      Q[key] += (Gs[t] - Q[key]) / visitCounts[key];
      greedyActions[s] = argmaxQ(Q, s, nA);
    }
  }

  return { Q, policy: greedyActions };
}

/**
 * MC Control with ε-greedy.
 * Converges to Q^π for the ε-soft optimal policy, NOT Q*.
 * Pass a function ε(episode) for a GLIE schedule to recover Q*.
 */
export function mcControlEpsGreedy(
  mdp: MDP,
  nEpisodes: number,
  epsilon: number | ((episode: number) => number),
  options: { rng?: () => number } = {},
): MCControlResult {
  const rng = options.rng ?? Math.random;
  const { nS, nA } = mdp;
  const s0 = 0;

  const Q = new Float64Array(nS * nA).fill(0);
  const visitCounts = new Int32Array(nS * nA).fill(0);
  const greedyActions = new Int32Array(nS).fill(0);

  for (let ep = 0; ep < nEpisodes; ep++) {
    const eps = typeof epsilon === "function" ? epsilon(ep) : epsilon;

    // Full episode follows ε-greedy policy (never pure greedy)
    const epPolicy = epsGreedyPolicy(mdp, greedyActions, eps);
    const steps = rollout(mdp, epPolicy, s0, 200, rng);
    if (steps.length === 0) continue;

    const rewards = steps.map(st => st.r);
    const Gs = computeReturns(rewards, mdp.gamma);

    const seen = new Set<number>();
    for (let t = 0; t < steps.length; t++) {
      const { s, a } = steps[t];
      if (mdp.terminals[s]) continue;
      const key = s * nA + a;
      if (seen.has(key)) continue;
      seen.add(key);
      visitCounts[key] += 1;
      Q[key] += (Gs[t] - Q[key]) / visitCounts[key];
      greedyActions[s] = argmaxQ(Q, s, nA);
    }
  }

  return { Q, policy: greedyActions };
}
