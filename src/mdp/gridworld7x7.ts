/**
 * 7×7 gridworld for the function-approximation lesson.
 * Extends the 3×3 design to a larger grid with walls and two goals.
 *
 * Layout (7×7, row 0 = top):
 *   S . . . . . G2(+0.5)     row 0
 *   . . . W . . .             row 1  (W = wall: entry blocked)
 *   . . . W . . .             row 2
 *   . . . W . . .             row 3
 *   . . . W . . .             row 4
 *   . . . W . . .             row 5
 *   . . . . . . G1(+1.0)     row 6
 *
 * The wall at column 3, rows 1–5 forces agents to navigate around it
 * (via row 0 or row 6), creating a genuinely non-linear value function.
 */
import { ACTION_DELTAS, ACTIONS, type MDP, type Policy } from "./types";

export const GRID_SIZE_7 = 7;

export function idx7(r: number, c: number): number {
  return r * GRID_SIZE_7 + c;
}

export function rc7(s: number): { r: number; c: number } {
  return { r: Math.floor(s / GRID_SIZE_7), c: s % GRID_SIZE_7 };
}

/** Wall cells: transition INTO these states is blocked (agent bounces back). */
const WALL_CELLS_7x7 = new Set<number>([
  idx7(1, 3),
  idx7(2, 3),
  idx7(3, 3),
  idx7(4, 3),
  idx7(5, 3),
]);

/** Goals for the 7×7 gridworld. */
export const GOAL1_7x7: [number, number] = [6, 6]; // reward +1.0
export const GOAL2_7x7: [number, number] = [0, 6]; // reward +0.5
export const START_7x7: [number, number] = [0, 0];

function move7(r: number, c: number, dr: number, dc: number): number {
  const nr = r + dr;
  const nc = c + dc;
  if (nr < 0 || nr >= GRID_SIZE_7 || nc < 0 || nc >= GRID_SIZE_7) {
    return idx7(r, c); // out of bounds → stay
  }
  const sp = idx7(nr, nc);
  if (WALL_CELLS_7x7.has(sp)) return idx7(r, c); // wall → stay
  return sp;
}

export interface Gridworld7x7Opts {
  gamma?: number;
  /** Goal 1 position [r,c] and reward. Default (6,6), +1.0. */
  goal1?: [number, number];
  goal1Reward?: number;
  /** Goal 2 position [r,c] and reward. Default (0,6), +0.5. */
  goal2?: [number, number];
  goal2Reward?: number;
}

export function buildGridworld7x7(opts: Gridworld7x7Opts = {}): MDP {
  const {
    gamma = 0.95,
    goal1 = GOAL1_7x7,
    goal1Reward = 1.0,
    goal2 = GOAL2_7x7,
    goal2Reward = 0.5,
  } = opts;

  const nS = GRID_SIZE_7 * GRID_SIZE_7;
  const nA = ACTIONS.length;

  const goal1S = idx7(goal1[0], goal1[1]);
  const goal2S = idx7(goal2[0], goal2[1]);
  const terminals = Array.from({ length: nS }, (_, s) => s === goal1S || s === goal2S);

  const enterReward = (sp: number): number =>
    sp === goal1S ? goal1Reward : sp === goal2S ? goal2Reward : 0;

  const P: number[][][] = [];
  const r: number[][] = [];

  for (let s = 0; s < nS; s++) {
    P[s] = [];
    r[s] = [];
    const { r: row, c: col } = rc7(s);

    for (let a = 0; a < nA; a++) {
      const row_p = new Array<number>(nS).fill(0);

      if (terminals[s]) {
        row_p[s] = 1;
        P[s][a] = row_p;
        r[s][a] = 0;
        continue;
      }

      const [dr, dc] = ACTION_DELTAS[a];
      const sp = move7(row, col, dr, dc);
      row_p[sp] = 1;
      P[s][a] = row_p;
      r[s][a] = enterReward(sp);
    }
  }

  return { nS, nA, gamma, P, r, terminals };
}

/** Uniform random policy for 7×7 gridworld. */
export function uniformPolicy7x7(mdp: MDP): Policy {
  const pi = Array.from({ length: mdp.nS }, () =>
    new Array<number>(mdp.nA).fill(1 / mdp.nA),
  );
  return { pi };
}

/** Set of wall cell indices for display. */
export const wallCells7x7 = WALL_CELLS_7x7;
