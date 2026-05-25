import { describe, expect, test } from "vitest";
import { apply1D, iterate1D, opNormInfinity, banachBound, supDist } from "./ops";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { bellmanExpectationBackup } from "../mdp/policy-evaluation";

describe("apply1D", () => {
  test("T(x) = 0.5x + 1 at x=10", () => {
    expect(apply1D(0.5, 1, 10)).toBe(6);
  });
  test("fixed point at x=2", () => {
    expect(apply1D(0.5, 1, 2)).toBe(2);
  });
});

describe("iterate1D", () => {
  test("converges to fixed point of 0.5x+1 from x0=10", () => {
    const T = (x: number) => 0.5 * x + 1;
    const traj = iterate1D(T, 10, 20);
    expect(traj).toHaveLength(21);
    expect(traj[20]).toBeCloseTo(2, 4);
  });
  test("trajectory starts at x0", () => {
    const T = (x: number) => 0.5 * x + 1;
    expect(iterate1D(T, 10, 20)[0]).toBe(10);
  });
});

describe("opNormInfinity", () => {
  test("max absolute row sum", () => {
    const A = [[0.3, 0.4], [0.5, 0.2]];
    expect(opNormInfinity(A)).toBe(0.7);
  });
  test("identity matrix has norm 1", () => {
    expect(opNormInfinity([[1, 0], [0, 1]])).toBe(1);
  });
  test("row-stochastic matrix has norm 1", () => {
    expect(opNormInfinity([[0.3, 0.7], [0.6, 0.4]])).toBeCloseTo(1, 10);
  });
});

describe("banachBound", () => {
  test("exactly tracks T(x)=0.5x+1 from x0=10", () => {
    const c = 0.5;
    // d0 = |x_1 - x_0| = |T(10) - 10| = |6 - 10| = 4
    const d0 = Math.abs(apply1D(0.5, 1, 10) - 10);
    for (let k = 0; k <= 6; k++) {
      let xk = 10;
      for (let j = 0; j < k; j++) xk = 0.5 * xk + 1;
      expect(Math.abs(xk - 2)).toBeCloseTo(banachBound(c, k, d0), 4);
    }
  });
});

describe("Bellman T^π contraction on gridworld", () => {
  test("ratio ≤ γ for first step", () => {
    const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    const uniform = uniformPolicy(mdp);
    const V1 = new Array(9).fill(0);
    const V2 = new Array(9).fill(2);
    const d0 = supDist(V1, V2);
    const V1n = bellmanExpectationBackup(mdp, uniform, V1);
    const V2n = bellmanExpectationBackup(mdp, uniform, V2);
    const d1 = supDist(V1n, V2n);
    expect(d1).toBeLessThanOrEqual(0.9 * d0 + 1e-9);
    expect(d1 / d0).toBeCloseTo(0.9, 4);
  });
});
