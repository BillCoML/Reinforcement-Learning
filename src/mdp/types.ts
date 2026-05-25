/**
 * Core MDP types and the gridworld's index/action conventions, shared across
 * the whole lesson. State indexing is row-major: idx(r, c) = 3r + c, so the
 * 3×3 gridworld is states 0..8 with the pit at 4 and the goal at 8.
 */

export interface MDP {
  readonly nS: number;
  readonly nA: number;
  readonly gamma: number;
  /** P[s][a][s'] — transition probability. Rows sum to 1. */
  readonly P: number[][][];
  /** r[s][a] — expected immediate reward (collected on entry into s'). */
  readonly r: number[][];
  /** terminals[s] — true if s is terminal (absorbing, V = 0). */
  readonly terminals: boolean[];
}

export interface Policy {
  /** pi[s][a] — probability of taking action a in state s. Rows sum to 1. */
  readonly pi: number[][];
}

// Action indices. Order matters: it's the column order of pi and P, and the
// quadrant order in V5. Up / Right / Down / Left, clockwise from north.
export const UP = 0;
export const RIGHT = 1;
export const DOWN = 2;
export const LEFT = 3;

export const ACTIONS = [UP, RIGHT, DOWN, LEFT] as const;
export const ACTION_NAMES = ["Up", "Right", "Down", "Left"] as const;
/** Row/col deltas for each action. Row 0 is the top; row increases downward. */
export const ACTION_DELTAS: readonly [number, number][] = [
  [-1, 0], // Up
  [0, +1], // Right
  [+1, 0], // Down
  [0, -1], // Left
];

/** Gridworld geometry. Square grid of side `GRID_SIZE`. */
export const GRID_SIZE = 3;

/** Row-major state index for cell (r, c). */
export function idx(r: number, c: number): number {
  return r * GRID_SIZE + c;
}

/** Inverse of idx: state → (row, col). */
export function rc(s: number): { r: number; c: number } {
  return { r: Math.floor(s / GRID_SIZE), c: s % GRID_SIZE };
}
