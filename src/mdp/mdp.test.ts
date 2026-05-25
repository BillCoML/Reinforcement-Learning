import { describe, expect, test } from "vitest";
import { buildGridworld, uniformPolicy, deterministicPolicy } from "./gridworld";
import { policyEvaluationExact, pPi, bellmanExpectationBackup } from "./policy-evaluation";
import { bellmanOptimalityBackup, optimalValue, greedyActions } from "./value-iteration";
import { qFromV, advantage } from "./q-and-advantage";
import { rollout } from "./rollout";
import { idx, UP, RIGHT, DOWN, LEFT } from "./types";

describe("gridworld construction", () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

  test("9 states, 4 actions, pit and goal terminal", () => {
    expect(mdp.nS).toBe(9);
    expect(mdp.nA).toBe(4);
    expect(mdp.terminals[idx(1, 1)]).toBe(true); // pit
    expect(mdp.terminals[idx(2, 2)]).toBe(true); // goal
    expect(mdp.terminals[idx(0, 0)]).toBe(false);
  });

  test("transition rows are probability distributions", () => {
    for (let s = 0; s < mdp.nS; s++)
      for (let a = 0; a < mdp.nA; a++)
        expect(mdp.P[s][a].reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12);
  });

  test("deterministic dynamics at (1,0) match the spec", () => {
    expect(mdp.P[idx(1, 0)][UP][idx(0, 0)]).toBe(1);
    expect(mdp.P[idx(1, 0)][RIGHT][idx(1, 1)]).toBe(1); // into the pit
    expect(mdp.P[idx(1, 0)][DOWN][idx(2, 0)]).toBe(1);
    expect(mdp.P[idx(1, 0)][LEFT][idx(1, 0)]).toBe(1); // wall bounce
  });

  test("reward is collected on entry into pit / goal", () => {
    expect(mdp.r[idx(1, 0)][RIGHT]).toBe(-1); // enters pit
    expect(mdp.r[idx(2, 1)][RIGHT]).toBe(1); // enters goal
    expect(mdp.r[idx(0, 0)][RIGHT]).toBe(0);
  });

  test("slippery rows still sum to 1 with 80/10/10 split", () => {
    const slip = buildGridworld({ slippery: true, gamma: 0.9 });
    const row = slip.P[idx(0, 0)][RIGHT];
    expect(row.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12);
    // intended Right (to (0,1)) gets 0.8; perpendiculars Up(bounce)/Down split 0.1 each.
    expect(row[idx(0, 1)]).toBeCloseTo(0.8, 12);
  });
});

describe("policy evaluation — uniform random", () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const uniform = uniformPolicy(mdp);
  const V = policyEvaluationExact(mdp, uniform);

  test("V^π(0,0) ≈ -0.4205", () => {
    expect(V[idx(0, 0)]).toBeCloseTo(-0.4205, 3);
  });

  test("V^π(0,1) ≈ -0.5139", () => {
    expect(V[idx(0, 1)]).toBeCloseTo(-0.5139, 3);
  });

  test("V^π(1,2) ≈ -0.069 (the §4 sanity-check cell)", () => {
    expect(V[idx(1, 2)]).toBeCloseTo(-0.069, 3);
  });

  test("terminals are exactly 0", () => {
    expect(V[idx(1, 1)]).toBe(0);
    expect(V[idx(2, 2)]).toBe(0);
  });

  test("P^π rows are stochastic", () => {
    const M = pPi(mdp, uniform);
    for (let s = 0; s < mdp.nS; s++) {
      let sum = 0;
      for (let sp = 0; sp < mdp.nS; sp++) sum += M.get(s, sp);
      expect(sum).toBeCloseTo(1, 12);
    }
  });

  test("iterative T^π converges to the exact solve", () => {
    let Vk = new Array(9).fill(0);
    for (let k = 0; k < 200; k++) Vk = bellmanExpectationBackup(mdp, uniform, Vk);
    for (let s = 0; s < 9; s++) expect(Vk[s]).toBeCloseTo(V[s], 5);
    // and the §6 trace value at k=∞
    expect(Vk[idx(0, 0)]).toBeCloseTo(-0.4205, 4);
  });
});

describe("Bellman optimality — V*, Q*, advantage", () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const Vstar = optimalValue(mdp, 100);

  test("V*(0,0) = γ^3 = 0.729 and V*(2,1) = 1.0", () => {
    expect(Vstar[idx(0, 0)]).toBeCloseTo(0.729, 4);
    expect(Vstar[idx(2, 1)]).toBeCloseTo(1.0, 4);
  });

  test("V* matches the full §7 table", () => {
    const expected = [
      [0.729, 0.81, 0.9],
      [0.81, 0.0, 1.0],
      [0.9, 1.0, 0.0],
    ];
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++)
        expect(Vstar[idx(r, c)]).toBeCloseTo(expected[r][c], 4);
  });

  test("Q*((1,0), right) = -1 (steps into pit), down = 0.81", () => {
    const Q = qFromV(mdp, Vstar);
    expect(Q[idx(1, 0)][RIGHT]).toBeCloseTo(-1.0, 6);
    expect(Q[idx(1, 0)][DOWN]).toBeCloseTo(0.81, 4);
  });

  test("advantage of optimal action is 0; A*((1,0),right) = -1.81", () => {
    const Q = qFromV(mdp, Vstar);
    const A = advantage(Q, Vstar);
    expect(A[idx(1, 0)][DOWN]).toBeCloseTo(0, 6);
    expect(A[idx(1, 0)][RIGHT]).toBeCloseTo(-1.81, 4);
  });

  test("V*(0,0) scales as γ^3 across discount factors", () => {
    for (const gamma of [0.5, 0.7, 0.9, 0.99]) {
      const m = buildGridworld({ slippery: false, gamma });
      const V = optimalValue(m, 300);
      expect(V[idx(0, 0)]).toBeCloseTo(Math.pow(gamma, 3), 4);
    }
  });

  test("(0,0) has two optimal actions: right and down", () => {
    const tied = greedyActions(mdp, Vstar, idx(0, 0));
    expect(tied.sort()).toEqual([RIGHT, DOWN].sort());
  });

  test("one backup is a single application of T^*", () => {
    const once = bellmanOptimalityBackup(mdp, new Array(9).fill(0));
    // After one step only neighbours of terminals get non-zero value.
    expect(once[idx(2, 1)]).toBeCloseTo(1.0, 6); // right into goal
    expect(once[idx(0, 0)]).toBeCloseTo(0, 6); // 3 steps away, still 0
  });
});

describe("rollout", () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

  test("optimal-ish path reaches the goal and collects +1 on entry", () => {
    // down, down, right, right from (0,0): (0,0)->(1,0)->(2,0)->(2,1)->(2,2 goal)
    const actions = new Array(9).fill(DOWN);
    actions[idx(2, 0)] = RIGHT;
    actions[idx(2, 1)] = RIGHT;
    const pol = deterministicPolicy(mdp, actions);
    const traj = rollout(mdp, pol, idx(0, 0), 50);
    expect(traj.length).toBe(4);
    expect(traj[traj.length - 1].sp).toBe(idx(2, 2));
    expect(traj[traj.length - 1].r).toBe(1);
  });
});
