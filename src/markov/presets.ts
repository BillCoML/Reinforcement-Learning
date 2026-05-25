/**
 * Named example chains used across the lesson's visualizations. The numeric
 * values are locked by the spec — prose, tests, and the offline JSON all refer
 * back to these matrices.
 */

export interface ChainPreset {
  key: string;
  /** Human label shown in dropdowns. */
  label: string;
  /** Short structural tag for the convergence lab. */
  tag: string;
  /** Row-stochastic transition matrix. */
  P: number[][];
  /** Optional per-state display names (e.g. weather: sunny/cloudy/rainy). */
  stateNames?: string[];
}

/** The running example: an aperiodic, ergodic 3-state weather chain. */
export const weather: ChainPreset = {
  key: "weather",
  label: "Weather (ergodic)",
  tag: "irreducible · aperiodic",
  P: [
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
  ],
  stateNames: ["sunny", "cloudy", "rainy"],
};

/** The canonical periodic counterexample: deterministic two-state flip. */
export const periodic2: ChainPreset = {
  key: "periodic2",
  label: "Periodic-2 (irreducible, periodic)",
  tag: "irreducible · period 2",
  P: [
    [0, 1],
    [1, 0],
  ],
};

/** Reducible 4-state chain: {0,1} recurrent, {2,3} transient (leaks into {0,1}). */
export const reducible: ChainPreset = {
  key: "reducible",
  label: "Reducible (multiple stationary distributions)",
  tag: "reducible · 2 classes",
  P: [
    [0.7, 0.3, 0, 0],
    [0.4, 0.6, 0, 0],
    [0.1, 0, 0.3, 0.6],
    [0, 0.2, 0.5, 0.3],
  ],
};

/** Sticky birth-death walk on 4 states: ergodic but λ⋆ near 1, so it mixes slowly. */
export const slowMixing: ChainPreset = {
  key: "slowMixing",
  label: "Slow-mixing (eigenvalue near 1)",
  tag: "irreducible · aperiodic · small gap",
  P: [
    [0.9, 0.1, 0, 0],
    [0.1, 0.8, 0.1, 0],
    [0, 0.1, 0.8, 0.1],
    [0, 0, 0.1, 0.9],
  ],
};

/** 3-state birth-death chain — reversible, used for the detailed-balance section. */
export const birthDeath: ChainPreset = {
  key: "birthDeath",
  label: "Birth-death (reversible)",
  tag: "reversible",
  P: [
    [0.5, 0.5, 0],
    [0.3, 0.4, 0.3],
    [0, 0.6, 0.4],
  ],
};

/** An asymmetric (irreversible) 3-cycle-ish chain — the V6 "not reversible" preset. */
export const asymmetric: ChainPreset = {
  key: "asymmetric",
  label: "Asymmetric (not reversible)",
  tag: "irreversible",
  P: [
    [0.1, 0.6, 0.3],
    [0.3, 0.2, 0.5],
    [0.6, 0.2, 0.2],
  ],
};

export const PRESETS: Record<string, ChainPreset> = {
  weather,
  periodic2,
  reducible,
  slowMixing,
  birthDeath,
  asymmetric,
};

/** The four chains offered in the Convergence Lab's selector. */
export const CONVERGENCE_LAB_PRESETS: ChainPreset[] = [weather, periodic2, reducible, slowMixing];
