/**
 * The 3×3 gridworld — the single running example of the lesson. One pit, one
 * goal, both terminal; reward is collected on the transition *into* a state
 * (+1 entering the goal, −1 entering the pit). Deterministic by default; a
 * "slippery" variant (80% intended, 10% each perpendicular) feeds the
 * stochastic mode of V1 and the centerpiece.
 */
import {
  ACTION_DELTAS,
  ACTIONS,
  GRID_SIZE,
  idx,
  rc,
  type MDP,
  type Policy,
} from "./types";

export interface GridworldOpts {
  /** Slippery transitions: 80% intended, 10% to each perpendicular. */
  slippery?: boolean;
  gamma?: number;
  /** Pit cell [row, col]; terminal, −1 on entry. Default (1,1). */
  pit?: [number, number];
  /** Goal cell [row, col]; terminal, +1 on entry. Default (2,2). */
  goal?: [number, number];
  /** Probability of the intended move under slippery dynamics. */
  slipMain?: number;
}

const DEFAULTS = {
  slippery: false,
  gamma: 0.9,
  pit: [1, 1] as [number, number],
  goal: [2, 2] as [number, number],
  slipMain: 0.8,
};

/** Clamp a move to the grid: walls bounce, so off-grid targets stay put. */
function move(r: number, c: number, dr: number, dc: number): number {
  let nr = r + dr;
  let nc = c + dc;
  if (nr < 0 || nr >= GRID_SIZE || nc < 0 || nc >= GRID_SIZE) {
    nr = r;
    nc = c;
  }
  return idx(nr, nc);
}

/** The two actions perpendicular to action a (for slip distribution). */
function perpendiculars(a: number): [number, number] {
  // Up(0)/Down(2) are vertical → slip left/right; Right(1)/Left(3) → up/down.
  return a === 0 || a === 2 ? [1, 3] : [0, 2];
}

export function buildGridworld(opts: GridworldOpts = {}): MDP {
  const { slippery, gamma, pit, goal, slipMain } = { ...DEFAULTS, ...opts };
  const nS = GRID_SIZE * GRID_SIZE;
  const nA = ACTIONS.length;

  const pitS = idx(pit[0], pit[1]);
  const goalS = idx(goal[0], goal[1]);
  const terminals = Array.from({ length: nS }, (_, s) => s === pitS || s === goalS);

  // Reward collected when *entering* state s'.
  const enterReward = (sp: number): number => (sp === goalS ? 1 : sp === pitS ? -1 : 0);
  const slipSide = (1 - slipMain) / 2;

  const P: number[][][] = [];
  const r: number[][] = [];

  for (let s = 0; s < nS; s++) {
    P[s] = [];
    r[s] = [];
    const { r: row, c: col } = rc(s);

    for (let a = 0; a < nA; a++) {
      const row_p = new Array<number>(nS).fill(0);

      if (terminals[s]) {
        // Absorbing: stay in place forever, no further reward.
        row_p[s] = 1;
        P[s][a] = row_p;
        r[s][a] = 0;
        continue;
      }

      const add = (action: number, prob: number) => {
        const [dr, dc] = ACTION_DELTAS[action];
        row_p[move(row, col, dr, dc)] += prob;
      };

      if (slippery) {
        add(a, slipMain);
        const [p1, p2] = perpendiculars(a);
        add(p1, slipSide);
        add(p2, slipSide);
      } else {
        add(a, 1);
      }

      P[s][a] = row_p;
      // Expected immediate reward = Σ_s' P(s'|s,a) · enterReward(s').
      let rsa = 0;
      for (let sp = 0; sp < nS; sp++) rsa += row_p[sp] * enterReward(sp);
      r[s][a] = rsa;
    }
  }

  return { nS, nA, gamma, P, r, terminals };
}

/** Uniform random policy: π(a|s) = 1/nA everywhere. */
export function uniformPolicy(mdp: MDP): Policy {
  const pi = Array.from({ length: mdp.nS }, () =>
    new Array<number>(mdp.nA).fill(1 / mdp.nA),
  );
  return { pi };
}

/**
 * Deterministic policy from a per-state action array. Terminal states get a
 * harmless uniform row (their actions never matter).
 */
export function deterministicPolicy(mdp: MDP, actions: number[]): Policy {
  const pi = Array.from({ length: mdp.nS }, (_, s) => {
    const row = new Array<number>(mdp.nA).fill(0);
    if (mdp.terminals[s]) row.fill(1 / mdp.nA);
    else row[actions[s]] = 1;
    return row;
  });
  return { pi };
}
