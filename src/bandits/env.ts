/**
 * Bandit environments. The environment owns the *truth* (the means); the
 * algorithm never sees it. `pull` returns a realized reward; the helpers
 * expose oracle quantities used only for regret accounting / display.
 */

import { argmax, sampleNormal, type RNG } from "./stats";

export interface BanditEnvironment {
  K: number;
  means: number[];
  pull(arm: number): number;
  optimalArm(): number;
  optimalMean(): number;
  gaps(): number[]; // Δ_i = μ* − μ_i for each arm
}

/** Bernoulli arms: reward ∈ {0, 1} with P(1) = μ_i. */
export class BernoulliBandit implements BanditEnvironment {
  K: number;
  private _gaps: number[];
  private _optArm: number;
  private _optMean: number;

  constructor(
    public means: number[],
    public rng: RNG = Math.random,
  ) {
    this.K = means.length;
    this._optArm = argmax(means);
    this._optMean = means[this._optArm];
    this._gaps = means.map((m) => this._optMean - m);
  }

  pull(arm: number): number {
    return this.rng() < this.means[arm] ? 1 : 0;
  }

  optimalArm(): number {
    return this._optArm;
  }
  optimalMean(): number {
    return this._optMean;
  }
  gaps(): number[] {
    return this._gaps;
  }
}

/** Gaussian arms: reward ~ N(μ_i, σ_i²). Rewards are unbounded. */
export class GaussianBandit implements BanditEnvironment {
  K: number;
  private _gaps: number[];
  private _optArm: number;
  private _optMean: number;

  constructor(
    public means: number[],
    public stds: number[],
    public rng: RNG = Math.random,
  ) {
    this.K = means.length;
    this._optArm = argmax(means);
    this._optMean = means[this._optArm];
    this._gaps = means.map((m) => this._optMean - m);
  }

  pull(arm: number): number {
    return this.means[arm] + this.stds[arm] * sampleNormal(this.rng);
  }

  optimalArm(): number {
    return this._optArm;
  }
  optimalMean(): number {
    return this._optMean;
  }
  gaps(): number[] {
    return this._gaps;
  }
}
