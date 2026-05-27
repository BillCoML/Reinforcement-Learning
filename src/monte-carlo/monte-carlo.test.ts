import { describe, expect, test } from "vitest";
import { mulberry32 } from "../importance-sampling/gaussian";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { valueIteration } from "../dp/algorithms";
import { idx } from "../mdp/types";
import { computeReturns } from "./returns";
import { mcPolicyEvaluation } from "./policy-evaluation";
import { mcOffPolicyEvaluation } from "./off-policy";
import { mcControlExploringStarts, mcControlEpsGreedy } from "./control";

const mdp = buildGridworld();
const uniform = uniformPolicy(mdp);
const { policy: detOpt } = valueIteration(mdp);

// 50-trial standard deviation of V^π(0,0) at given N
function measureStd(N: number, trials = 50): number {
  const estimates: number[] = [];
  for (let i = 0; i < trials; i++) {
    const rng = mulberry32(((42 * 1_000_003 + i * 97_001) >>> 0));
    const { V } = mcPolicyEvaluation(mdp, uniform, N, { rng });
    estimates.push(V[idx(0, 0)]);
  }
  const mean = estimates.reduce((s, x) => s + x, 0) / trials;
  return Math.sqrt(estimates.reduce((s, x) => s + (x - mean) ** 2, 0) / (trials - 1));
}

describe("computeReturns", () => {
  test("backward sweep includes terminal reward", () => {
    const G = computeReturns([0, 0, 0, 1], 0.9);
    expect(G[0]).toBeCloseTo(0.729, 4);
    expect(G[3]).toBeCloseTo(1.0, 10);
  });

  test("single-step episode", () => {
    const G = computeReturns([1], 0.9);
    expect(G[0]).toBeCloseTo(1.0, 10);
  });
});

describe("mcPolicyEvaluation", () => {
  test("first-visit MC converges to V^π(0,0) ≈ -0.4205 at N=50000", () => {
    const rng = mulberry32(0);
    const { V } = mcPolicyEvaluation(mdp, uniform, 50_000, { rng });
    expect(V[idx(0, 0)]).toBeCloseTo(-0.4205, 1);
  });

  test("first-visit convergence rate: std ≈ 0.044 / 0.013 / 0.0036", () => {
    const stds = [100, 1000, 10_000].map(N => measureStd(N));
    // Within 30% of theoretical 1/√N rate
    expect(stds[0]).toBeGreaterThan(0.044 * 0.7);
    expect(stds[0]).toBeLessThan(0.044 * 1.3);
    expect(stds[1]).toBeGreaterThan(0.013 * 0.7);
    expect(stds[1]).toBeLessThan(0.013 * 1.3);
    expect(stds[2]).toBeGreaterThan(0.0036 * 0.7);
    expect(stds[2]).toBeLessThan(0.0036 * 1.3);
  });

  test("every-visit MC also converges near -0.4205", () => {
    const rng = mulberry32(99);
    const { V } = mcPolicyEvaluation(mdp, uniform, 20_000, { firstVisit: false, rng });
    expect(V[idx(0, 0)]).toBeCloseTo(-0.4205, 1);
  });
});

describe("mcOffPolicyEvaluation", () => {
  test("weighted IS converges to exactly 0.7290 at N=10000", () => {
    for (let trial = 0; trial < 10; trial++) {
      const rng = mulberry32(((trial * 997 + 1) >>> 0));
      const { V } = mcOffPolicyEvaluation(mdp, detOpt, uniform, 10_000, {
        weighted: true,
        rng,
      });
      expect(V[idx(0, 0)]).toBeCloseTo(0.7290, 4);
    }
  });

  test("ESS at N=10000 ≈ 39 (matches non-zero trajectory count)", () => {
    const rng = mulberry32(42);
    const { ess } = mcOffPolicyEvaluation(mdp, detOpt, uniform, 10_000, {
      weighted: true,
      rng,
    });
    expect(ess[idx(0, 0)]).toBeGreaterThan(20);
    expect(ess[idx(0, 0)]).toBeLessThan(80);
  });

  test("zero-weight result is 0, not NaN", () => {
    // Behavior = all-down policy; target = optimal (which goes RIGHT first)
    // State (0,0): behavior goes down, target goes right → ρ=0 for every episode
    const allDown = {
      pi: Array.from({ length: mdp.nS }, (_, s) =>
        mdp.terminals[s]
          ? new Array(mdp.nA).fill(0.25)
          : [0, 0, 1, 0].map(Number), // always DOWN
      ),
    };
    const rng = mulberry32(1);
    const { V } = mcOffPolicyEvaluation(mdp, detOpt, allDown, 100, {
      weighted: true,
      rng,
    });
    expect(isNaN(V[idx(0, 0)])).toBe(false);
    expect(V[idx(0, 0)]).toBe(0);
  });
});

describe("mcControlExploringStarts", () => {
  test("converges to Q*(0,0,right) ≈ 0.7290 at N=50000", () => {
    const rng = mulberry32(0);
    const { Q, policy } = mcControlExploringStarts(mdp, 50_000, { rng });
    // Q*(0,0,right) = Q*(0,0,down) = 0.7290 — both are optimal, accept either
    expect(Q[idx(0, 0) * 4 + 1]).toBeCloseTo(0.7290, 1);
    expect(Q[idx(0, 0) * 4 + 2]).toBeCloseTo(0.7290, 1);
    expect([1, 2]).toContain(policy[idx(0, 0)]);
  });
});

describe("mcControlEpsGreedy", () => {
  test("ε=0.1 converges to Q^ε-soft ≈ 0.6307 NOT Q*=0.7290", () => {
    const rng = mulberry32(0);
    const { Q } = mcControlEpsGreedy(mdp, 50_000, 0.1, { rng });
    const q = Q[idx(0, 0) * 4 + 1];
    // Within 0.10 of the ε-soft optimal 0.6307
    expect(q).toBeGreaterThan(0.6307 - 0.10);
    expect(q).toBeLessThan(0.6307 + 0.10);
    // Critically: more than 0.05 away from Q* = 0.7290
    expect(Math.abs(q - 0.7290)).toBeGreaterThan(0.05);
  });

  test("GLIE schedule ε(n)=1/√n converges to Q* ≈ 0.7290 at N=100000", () => {
    const glie = (ep: number) => 1 / Math.sqrt(ep + 1);
    const rng = mulberry32(0);
    const { Q } = mcControlEpsGreedy(mdp, 100_000, glie, { rng });
    expect(Q[idx(0, 0) * 4 + 1]).toBeCloseTo(0.7290, 1);
  });
});
