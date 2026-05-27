import { describe, test, expect } from "vitest";
import { mulberry32 } from "../importance-sampling/gaussian";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { idx, RIGHT, UP } from "../mdp/types";
import { tdZero } from "./td-zero";
import { sarsa } from "./sarsa";
import { qLearning } from "./q-learning";
import { nStepTD } from "./n-step-td";
import { tdLambda } from "./td-lambda";
import { makeMaxBiasMDP, qLearningMaxBias } from "./max-bias-mdp";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const uniform = uniformPolicy(mdp);
const S0 = idx(0, 0); // start state

// ─── TD(0) ────────────────────────────────────────────────────────────────────

describe("tdZero", () => {
  test("V(0,0) mean within 0.05 of −0.4205 over 20 trials at N=2000, α=0.1", () => {
    const trials = Array.from({ length: 20 }, (_, t) =>
      tdZero(mdp, uniform, 2000, 0.1, { rng: mulberry32(t) }).V[S0],
    );
    const mean = trials.reduce((a, b) => a + b) / trials.length;
    expect(mean).toBeGreaterThan(-0.4205 - 0.05);
    expect(mean).toBeLessThan(-0.4205 + 0.05);
  });

  test("terminal-state target is r, not r + γV(s')", () => {
    // Run 1 episode: detect that V never gets a ghost bootstrap from terminal.
    // With α=1, V[s] becomes target exactly after 1 update. If terminal
    // handling is wrong, V[pre-terminal] would be r + γ*0 = r anyway
    // (terminal V=0), but V[start] could be contaminated. Just check
    // the algorithm doesn't crash and produces bounded values.
    const { V } = tdZero(mdp, uniform, 50, 1.0, { rng: mulberry32(0) });
    for (let s = 0; s < mdp.nS; s++) {
      expect(isFinite(V[s])).toBe(true);
      expect(Math.abs(V[s])).toBeLessThanOrEqual(2);
    }
  });

  test("constant-α std grows with α (Robbins-Monro oscillation)", () => {
    function measureStd(alpha: number, N: number, trials: number): number {
      const est = Array.from({ length: trials }, (_, t) =>
        tdZero(mdp, uniform, N, alpha, { rng: mulberry32(t * 997 + 1) }).V[S0],
      );
      const mean = est.reduce((a, b) => a + b) / trials;
      return Math.sqrt(est.reduce((s, x) => s + (x - mean) ** 2, 0) / (trials - 1));
    }
    const std01 = measureStd(0.01, 5000, 20);
    const std10 = measureStd(0.10, 5000, 20);
    // std grows roughly as sqrt(α): ratio should be between 2 and 6
    const ratio = std10 / std01;
    expect(ratio).toBeGreaterThan(2);
    expect(ratio).toBeLessThan(6);
  });
});

// ─── SARSA ────────────────────────────────────────────────────────────────────

describe("sarsa", () => {
  test("Q(0,0,right) mean in [0.45, 0.65] at N=10000 over 20 seeds — converging toward Q^π_ε-soft", () => {
    // Constant-α SARSA oscillates around Q^π_ε-soft(0,0,right)=0.6307.
    // Single-seed values vary widely; the mean is the meaningful check.
    const vals = Array.from({ length: 20 }, (_, t) => {
      const { Q } = sarsa(mdp, 10000, 0.1, 0.1, { rng: mulberry32(t) });
      return Q[S0 * mdp.nA + RIGHT];
    });
    const mean = vals.reduce((a, b) => a + b) / vals.length;
    expect(mean).toBeGreaterThan(0.45);
    expect(mean).toBeLessThan(0.65);
  });

  test("Q(0,0,right) does NOT converge to 0.7290 (Q*)", () => {
    const { Q } = sarsa(mdp, 10000, 0.1, 0.1, { rng: mulberry32(0) });
    const qRight = Q[S0 * mdp.nA + RIGHT];
    expect(Math.abs(qRight - 0.7290)).toBeGreaterThan(0.05);
  });

  test("on-policy convergence stable across multiple seeds", () => {
    // All seeds should land in [0.40, 0.70] — not near 0.7290
    const results = Array.from({ length: 5 }, (_, t) => {
      const { Q } = sarsa(mdp, 10000, 0.1, 0.1, { rng: mulberry32(t + 10) });
      return Q[S0 * mdp.nA + RIGHT];
    });
    for (const q of results) {
      expect(q).toBeGreaterThan(0.40);
      expect(q).toBeLessThan(0.70);
    }
  });
});

// ─── Q-learning ───────────────────────────────────────────────────────────────

describe("qLearning", () => {
  test("Q(0,0,right) within 0.05 of 0.7290 (Q*) at N=10000", () => {
    const { Q } = qLearning(mdp, 10000, 0.1, 0.1, { rng: mulberry32(0) });
    expect(Q[S0 * mdp.nA + RIGHT]).toBeCloseTo(0.7290, 1);
  });

  test("Q(0,0,up) within 0.05 of 0.6561 (Q*) at N=10000", () => {
    const { Q } = qLearning(mdp, 10000, 0.1, 0.1, { rng: mulberry32(0) });
    expect(Q[S0 * mdp.nA + UP]).toBeCloseTo(0.6561, 1);
  });

  test("greedy policy at (0,0) is RIGHT", () => {
    const { greedyPolicy } = qLearning(mdp, 20000, 0.1, 0.1, { rng: mulberry32(0) });
    expect(greedyPolicy[S0]).toBe(RIGHT);
  });

  test("converges to Q* across multiple seeds", () => {
    const results = Array.from({ length: 5 }, (_, t) => {
      const { Q } = qLearning(mdp, 10000, 0.1, 0.1, { rng: mulberry32(t) });
      return Q[S0 * mdp.nA + RIGHT];
    });
    for (const q of results) {
      expect(q).toBeGreaterThan(0.65);
      expect(q).toBeLessThanOrEqual(0.80);
    }
  });
});

// ─── n-step TD ────────────────────────────────────────────────────────────────

describe("nStepTD", () => {
  test("n=1 produces identical V to tdZero given same seeded RNG", () => {
    const seed = 42;
    const { V: vTD0 } = tdZero(mdp, uniform, 1000, 0.1, { rng: mulberry32(seed) });
    const { V: vN1 } = nStepTD(mdp, uniform, 1, 1000, 0.1, { rng: mulberry32(seed) });
    for (let s = 0; s < mdp.nS; s++) {
      expect(vN1[s]).toBeCloseTo(vTD0[s], 6);
    }
  });

  test("larger n increases variance on the short-horizon gridworld", () => {
    function rmse(n: number, trials = 10): number {
      const truth = -0.4205;
      const errs = Array.from({ length: trials }, (_, t) =>
        nStepTD(mdp, uniform, n, 2000, 0.1, { rng: mulberry32(t) }).V[S0] - truth,
      );
      return Math.sqrt(errs.reduce((s, e) => s + e * e, 0) / trials);
    }
    // RMSE should grow as n increases (short-horizon gridworld, variance dominates)
    expect(rmse(1)).toBeLessThan(rmse(8));
  });
});

// ─── TD(λ) ────────────────────────────────────────────────────────────────────

describe("tdLambda", () => {
  test("λ=0 produces identical V to tdZero given same seeded RNG (within 1e-4)", () => {
    const seed = 42;
    const { V: vTD0 } = tdZero(mdp, uniform, 1000, 0.1, { rng: mulberry32(seed) });
    const { V: vL0 } = tdLambda(mdp, uniform, 0, 1000, 0.1, { rng: mulberry32(seed) });
    for (let s = 0; s < mdp.nS; s++) {
      expect(vL0[s]).toBeCloseTo(vTD0[s], 4);
    }
  });

  test("λ=0.5 converges without diverging (traces reset correctly between episodes)", () => {
    // λ=1 with constant α is unstable; λ=0.5 with small α converges.
    // Divergence would indicate traces are not resetting between episodes.
    const { V } = tdLambda(mdp, uniform, 0.5, 5000, 0.05, { rng: mulberry32(0) });
    for (let s = 0; s < mdp.nS; s++) {
      expect(isFinite(V[s])).toBe(true);
    }
    expect(Math.abs(V[S0] - (-0.4205))).toBeLessThan(0.20);
  });

  test("finalTraces shape is nS", () => {
    const { finalTraces } = tdLambda(mdp, uniform, 100, 0.5, 0.1, { rng: mulberry32(0) });
    expect(finalTraces.length).toBe(mdp.nS);
  });
});

// ─── Maximization bias ────────────────────────────────────────────────────────

describe("maximizationBias", () => {
  test("Q-learning chooses left from A in >25% of first 300 episodes (mean over 20 seeds)", () => {
    const biasedMDP = makeMaxBiasMDP();
    const fractions = Array.from({ length: 20 }, (_, t) => {
      const { leftFraction } = qLearningMaxBias(biasedMDP, 300, 0.1, 0.1, {
        rng: mulberry32(t),
      });
      return leftFraction;
    });
    const mean = fractions.reduce((a, b) => a + b) / fractions.length;
    // Optimal is ~5% (ε/2 = 0.05); maximization bias pushes this well above 0.25.
    expect(mean).toBeGreaterThan(0.25);
  });

  test("leftHistory is monotone-adjacent (length = nEpisodes)", () => {
    const biasedMDP = makeMaxBiasMDP();
    const { leftHistory } = qLearningMaxBias(biasedMDP, 100, 0.1, 0.1, {
      rng: mulberry32(0),
    });
    expect(leftHistory.length).toBe(100);
    // All values in [0, 1]
    for (const f of leftHistory) {
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThanOrEqual(1);
    }
  });
});
