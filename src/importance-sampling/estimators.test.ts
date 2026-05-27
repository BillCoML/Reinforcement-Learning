import { describe, expect, test } from "vitest";
import {
  importanceWeight,
  ordinaryIS,
  weightedIS,
  effectiveSampleSize,
  trajectoryISWeight,
  perDecisionIS,
} from "./estimators";
import { normalPdf, sampleNormal, mulberry32, boxMuller } from "./gaussian";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { optimalPolicy } from "../mdp/policies";
import { rollout } from "../mdp/rollout";
import { idx, RIGHT, DOWN } from "../mdp/types";

// ---------------------------------------------------------------------------
// Gaussian helpers
// ---------------------------------------------------------------------------

describe("normalPdf", () => {
  test("N(0,1) at x=0 ≈ 0.3989", () => {
    expect(normalPdf(0, 0, 1)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 10);
  });
  test("N(0,2) integrates to 1 via Riemann sum", () => {
    let s = 0;
    const dx = 0.001;
    for (let x = -20; x < 20; x += dx) s += normalPdf(x, 0, 2) * dx;
    expect(s).toBeCloseTo(1, 2);
  });
});

describe("Box-Muller correctness", () => {
  test("empirical mean ≈ 0 and variance ≈ 1 from N=10000 draws", () => {
    const rng = mulberry32(42);
    const samples = sampleNormal(10000, 0, 1, rng);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance =
      samples.reduce((a, x) => a + (x - mean) ** 2, 0) / samples.length;
    expect(mean).toBeGreaterThan(-0.05);
    expect(mean).toBeLessThan(0.05);
    expect(variance).toBeGreaterThan(0.95);
    expect(variance).toBeLessThan(1.05);
  });

  test("two pairs from distinct u1,u2 are not identical", () => {
    const rng = mulberry32(7);
    const [a1, b1] = boxMuller(rng);
    const [a2, b2] = boxMuller(rng);
    expect(a1).not.toBe(a2);
    expect(b1).not.toBe(b2);
  });
});

// ---------------------------------------------------------------------------
// Scalar IS estimators
// ---------------------------------------------------------------------------

describe("importanceWeight", () => {
  const p = (x: number) => normalPdf(x, 0, 1);
  const q2 = (x: number) => normalPdf(x, 0, 2);

  test("returns p/q for normal coverage", () => {
    const w = importanceWeight(p, q2, 0);
    expect(w).toBeCloseTo(normalPdf(0, 0, 1) / normalPdf(0, 0, 2), 10);
  });
  test("returns Infinity when q(x)=0 (coverage violated)", () => {
    const zeroPdf = () => 0;
    expect(importanceWeight(p, zeroPdf, 0)).toBe(Infinity);
  });
});

describe("ordinaryIS — Gaussian f(X)=X² unbiased at N=200000", () => {
  test("estimate ≈ 1.0 ± 0.1 (p=N(0,1), q=N(0,2), true E_p[X²]=1)", () => {
    const rng = mulberry32(123);
    const samples = sampleNormal(200_000, 0, 2, rng);
    const p = (x: number) => normalPdf(x, 0, 1);
    const q = (x: number) => normalPdf(x, 0, 2);
    const est = ordinaryIS(samples, (x) => x * x, (x) => p(x) / q(x));
    expect(est).toBeGreaterThan(0.9);
    expect(est).toBeLessThan(1.1);
  });
});

describe("weightedIS", () => {
  test("agrees with ordinaryIS when proposal = target", () => {
    const rng = mulberry32(5);
    const samples = sampleNormal(1000, 0, 1, rng);
    const w = () => 1;
    const f = (x: number) => x * x;
    const ord = ordinaryIS(samples, f, w);
    const wt = weightedIS(samples, f, w);
    expect(Math.abs(ord - wt)).toBeLessThan(0.01);
  });
  test("returns 0 when all weights are 0", () => {
    expect(weightedIS([1, 2, 3], () => 1, () => 0)).toBe(0);
  });
});

describe("effectiveSampleSize", () => {
  test("N_eff = N when all weights equal", () => {
    expect(effectiveSampleSize([2, 2, 2, 2, 2])).toBe(5);
  });
  test("N_eff = 1 when one weight dominates", () => {
    expect(effectiveSampleSize([256, 0, 0, 0, 0, 0, 0, 0])).toBe(1);
  });
  test("N_eff = 0 when all weights are 0", () => {
    expect(effectiveSampleSize([0, 0, 0])).toBe(0);
  });
  test("N_eff is between 1 and N for mixed weights", () => {
    const ess = effectiveSampleSize([10, 1, 1, 1]);
    expect(ess).toBeGreaterThan(1);
    expect(ess).toBeLessThan(4);
  });
});

// ---------------------------------------------------------------------------
// Trajectory IS
// ---------------------------------------------------------------------------

describe("trajectoryISWeight", () => {
  // Optimal path: s0→a1=RIGHT→s1→a1=RIGHT→s2→a2=DOWN→s5→a2=DOWN→goal
  const optTraj = [
    { s: idx(0, 0), a: RIGHT },
    { s: idx(0, 1), a: RIGHT },
    { s: idx(0, 2), a: DOWN },
    { s: idx(1, 2), a: DOWN },
  ];

  const piT = (s: number, a: number): number => {
    const opt: Record<number, number> = {
      [idx(0, 0)]: RIGHT,
      [idx(0, 1)]: RIGHT,
      [idx(0, 2)]: DOWN,
      [idx(1, 2)]: DOWN,
    };
    return opt[s] === a ? 1 : 0;
  };
  const piB = (): number => 0.25;

  test("weight is exactly 256 for length-4 matching trajectory (4^4)", () => {
    expect(trajectoryISWeight(optTraj, piT, piB)).toBe(256);
  });

  test("weight is 0 on any single mismatch", () => {
    const badTraj = [
      { s: idx(0, 0), a: RIGHT },
      { s: idx(0, 1), a: DOWN }, // WRONG — target wants RIGHT
      { s: idx(0, 2), a: DOWN },
    ];
    expect(trajectoryISWeight(badTraj, piT, piB)).toBe(0);
  });

  test("weight is 0 when piB returns 0 defensively", () => {
    const zeroBehavior = (): number => 0;
    expect(trajectoryISWeight(optTraj, piT, zeroBehavior)).toBe(0);
  });

  test("empty trajectory weight is 1 (empty product)", () => {
    expect(trajectoryISWeight([], piT, piB)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Gridworld sampling — ~38 non-zero contributions at N=10000
// ---------------------------------------------------------------------------

describe("gridworld trajectory IS sampling", () => {
  test("non-zero weight count ≈ 38 out of 10000 trajectories (±10 slack)", () => {
    const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    const piTarget = optimalPolicy(mdp);
    const piB = uniformPolicy(mdp);
    const piT_fn = (s: number, a: number) => piTarget.pi[s][a];
    const piB_fn = (s: number, a: number) => piB.pi[s][a];

    const rng = mulberry32(999);
    let nonZero = 0;
    for (let i = 0; i < 10_000; i++) {
      const steps = rollout(mdp, piB, 0, 20, rng);
      const traj = steps.map(({ s, a }) => ({ s, a }));
      const w = trajectoryISWeight(traj, piT_fn, piB_fn);
      if (w > 0) nonZero++;
    }
    // Expected ≈ 10000/256 ≈ 39.06
    expect(nonZero).toBeGreaterThanOrEqual(28);
    expect(nonZero).toBeLessThanOrEqual(52);
  });
});

// ---------------------------------------------------------------------------
// Per-decision IS
// ---------------------------------------------------------------------------

describe("perDecisionIS", () => {
  const piT = (s: number, a: number): number => {
    const opt: Record<number, number> = {
      [idx(0, 0)]: RIGHT,
      [idx(0, 1)]: RIGHT,
      [idx(0, 2)]: DOWN,
      [idx(1, 2)]: DOWN,
    };
    return opt[s] === a ? 1 : 0;
  };
  const piB = (): number => 0.25;

  test("sparse-reward matching trajectory: per-decision == trajectory IS", () => {
    // Sparse terminal reward → ρ_{0:t} for each step → at terminal step weights are
    // identical to the full trajectory weight. Per-decision collapses to trajectory IS.
    const gamma = 0.9;
    const traj = [
      { s: idx(0, 0), a: RIGHT, r: 0 },
      { s: idx(0, 1), a: RIGHT, r: 0 },
      { s: idx(0, 2), a: DOWN, r: 0 },
      { s: idx(1, 2), a: DOWN, r: 1 }, // reward on entry into goal
    ];
    const pdIS = perDecisionIS(traj, piT, piB, gamma);
    // trajectory IS: ρ_{0:3} * G0 = 256 * γ^3 * 1 = 186.624
    const trajIS =
      trajectoryISWeight(
        traj.map(({ s, a }) => ({ s, a })),
        piT,
        piB,
      ) *
      Math.pow(gamma, 3) *
      1;
    expect(pdIS).toBeCloseTo(trajIS, 8);
  });

  test("mismatch step: reward before mismatch is weighted correctly, mismatch stops", () => {
    const traj = [
      { s: idx(0, 0), a: RIGHT, r: 1 }, // rho updates to 4 first, then contributes 4*1 = 4
      { s: idx(0, 1), a: DOWN, r: 5 },  // target says RIGHT → piT=0 → rho=0, return
    ];
    const pdIS = perDecisionIS(traj, piT, piB, 1.0);
    // t=0: rho = 1*(1/0.25)=4; total += 1^0 * 1 * 4 = 4
    // t=1: rho = 4*(0/0.25)=0; if rho===0 return total=4
    expect(pdIS).toBeCloseTo(4, 8);
  });
});
