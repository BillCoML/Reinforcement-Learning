/** Core types for the PPO module (Lesson 11). */

export interface Trajectory {
  readonly states: number[];
  readonly actions: number[];
  readonly rewards: number[];
}

export interface PPOConfig {
  readonly lrPolicy: number;
  readonly lrValue: number;
  readonly clipEps: number;
  readonly gaeLambda: number;
  readonly epochs: number;
  readonly batchEpisodes: number;
  readonly entropyCoef: number;
  readonly valueCoef: number;
  readonly normalizeAdvantages: boolean;
}

export interface PPOState {
  readonly theta: Float64Array;  // 9*4 = 36 logits, row-major
  readonly V: Float64Array;      // 9 state values
}

export interface PPOIterationLog {
  readonly iter: number;
  readonly vStart: number;
  readonly meanKL: number;
  readonly clipFraction: number;
  readonly meanRatio: number;
  readonly maxRatio: number;
  readonly minRatio: number;
  readonly surrogateValue: number;
  readonly entropyMean: number;
  readonly batchSize: number;
}

export interface VanillaPGLog {
  readonly iter: number;
  readonly vStart: number;
  readonly batchSize: number;
}

export const DEFAULT_PPO_CONFIG: PPOConfig = {
  lrPolicy: 0.5,
  lrValue: 0.5,
  clipEps: 0.2,
  gaeLambda: 0.95,
  epochs: 4,
  batchEpisodes: 20,
  entropyCoef: 0.01,
  valueCoef: 0.5,
  normalizeAdvantages: true,
};

export const AGGRESSIVE_PPO_CONFIG: PPOConfig = {
  lrPolicy: 2.0,
  lrValue: 1.0,
  clipEps: 0.2,
  gaeLambda: 0.95,
  epochs: 10,
  batchEpisodes: 5,
  entropyCoef: 0.01,
  valueCoef: 0.5,
  normalizeAdvantages: true,
};
