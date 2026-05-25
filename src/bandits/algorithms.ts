/**
 * Bandit algorithms. Each is a class implementing a shared interface so the
 * Battle Arena can run them through an identical simulation loop.
 *
 * Convention: `selectArm(t)` receives the 0-indexed number of pulls made so
 * far (so the very first call is t=0). During initialization (t < K) every
 * arm is pulled once in order. This matches the §5 numerical trace, where the
 * UCB1 bonus at the 4th pull (after 3 init pulls, t=3) uses ln(t+1) = ln 4.
 */

import { argmax, sampleBeta, type RNG } from "./stats";

export interface AlgorithmState {
  N: number[]; // pull counts
  S: number[]; // sums of rewards
  muhat: number[]; // empirical means
  extra?: Record<string, unknown>; // algo-specific (e.g. Beta posteriors, UCB bonuses)
}

export interface BanditAlgorithm {
  name: string;
  reset(K: number): void;
  selectArm(t: number): number; // returns 0-indexed arm
  update(arm: number, reward: number): void;
  state(): AlgorithmState;
}

/** Shared bookkeeping for sample-mean-based algorithms. */
abstract class BaseAlgorithm implements BanditAlgorithm {
  abstract name: string;
  protected K = 0;
  protected N: number[] = [];
  protected S: number[] = [];
  protected muhat: number[] = [];

  constructor(public rng: RNG = Math.random) {}

  reset(K: number): void {
    this.K = K;
    this.N = new Array(K).fill(0);
    this.S = new Array(K).fill(0);
    this.muhat = new Array(K).fill(0);
  }

  update(arm: number, reward: number): void {
    this.N[arm] += 1;
    this.S[arm] += reward;
    this.muhat[arm] = this.S[arm] / this.N[arm];
  }

  abstract selectArm(t: number): number;

  state(): AlgorithmState {
    return { N: [...this.N], S: [...this.S], muhat: [...this.muhat] };
  }
}

/** Uniform random baseline — the "pure exploration" failure mode. */
export class RandomPolicy extends BaseAlgorithm {
  name = "Random";
  selectArm(_t: number): number {
    return Math.floor(this.rng() * this.K);
  }
}

/**
 * ε-greedy. With probability ε pull a uniformly random arm, else the
 * empirically-best arm. Supports an optional decay schedule
 * ε_t = min(1, c·K / (d²·t)) (Auer–Cesa-Bianchi–Fischer 2002).
 */
export class EpsilonGreedy extends BaseAlgorithm {
  name: string;
  constructor(
    public epsilon: number,
    rng: RNG = Math.random,
    public decay: { c: number; d: number } | null = null,
  ) {
    super(rng);
    this.name = `ε-greedy(${epsilon})`;
  }

  /** Effective ε at step t (1-indexed round t+1). */
  epsilonAt(t: number): number {
    if (!this.decay) return this.epsilon;
    const round = t + 1;
    return Math.min(1, (this.decay.c * this.K) / (this.decay.d * this.decay.d * round));
  }

  selectArm(t: number): number {
    if (t < this.K) return t; // initialize: pull each arm once
    if (this.rng() < this.epsilonAt(t)) {
      return Math.floor(this.rng() * this.K);
    }
    return argmax(this.muhat);
  }
}

/** UCB1 (Auer et al. 2002): pull argmax μ̂_i + √(2 ln t / N_i). */
export class UCB1 extends BaseAlgorithm {
  name = "UCB1";

  /** Per-arm bonus √(2 ln(t+1) / N_i). */
  bonuses(t: number): number[] {
    const round = t + 1;
    const logT = Math.log(round);
    return this.N.map((n) => (n > 0 ? Math.sqrt((2 * logT) / n) : Infinity));
  }

  /** Upper confidence bounds μ̂_i + bonus_i. */
  ucb(t: number): number[] {
    const b = this.bonuses(t);
    return this.muhat.map((m, i) => m + b[i]);
  }

  selectArm(t: number): number {
    if (t < this.K) return t; // initialize
    return argmax(this.ucb(t));
  }

  override state(): AlgorithmState {
    const t = this.N.reduce((a, b) => a + b, 0);
    return {
      N: [...this.N],
      S: [...this.S],
      muhat: [...this.muhat],
      extra: { bonus: this.bonuses(t), ucb: this.ucb(t) },
    };
  }
}

/**
 * Thompson sampling with Beta-Bernoulli conjugacy.
 * α_i = 1 + S_i, β_i = 1 + N_i − S_i, starting from the uniform prior Beta(1,1).
 * Each step samples θ_i ~ Beta(α_i, β_i) and pulls argmax θ_i.
 */
export class ThompsonBetaBernoulli extends BaseAlgorithm {
  name = "Thompson";
  alpha: number[] = [];
  beta: number[] = [];

  override reset(K: number): void {
    super.reset(K);
    this.alpha = new Array(K).fill(1);
    this.beta = new Array(K).fill(1);
  }

  override update(arm: number, reward: number): void {
    super.update(arm, reward);
    if (reward >= 0.5) this.alpha[arm] += 1;
    else this.beta[arm] += 1;
  }

  selectArm(_t: number): number {
    const samples = this.alpha.map((a, i) => sampleBeta(a, this.beta[i], this.rng));
    return argmax(samples);
  }

  override state(): AlgorithmState {
    return {
      N: [...this.N],
      S: [...this.S],
      muhat: [...this.muhat],
      extra: { alpha: [...this.alpha], beta: [...this.beta] },
    };
  }
}

/** Convenience: the five algorithms used in §7's comparison, by key. */
export function makeAlgorithm(key: string, rng: RNG = Math.random): BanditAlgorithm {
  switch (key) {
    case "random":
      return new RandomPolicy(rng);
    case "eps001":
      return new EpsilonGreedy(0.01, rng);
    case "eps01":
      return new EpsilonGreedy(0.1, rng);
    case "ucb1":
      return new UCB1(rng);
    case "thompson":
      return new ThompsonBetaBernoulli(rng);
    default:
      throw new Error(`unknown algorithm key: ${key}`);
  }
}
