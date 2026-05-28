import { describe, test, expect } from "vitest";
import { mulberry32 } from "../importance-sampling/gaussian";
import { buildGridworld } from "../mdp/gridworld";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { SoftmaxPolicy } from "./softmax-policy";
import { gaussianScoreEstimator, gaussianTrueGradient } from "./gaussian-score";
import { reinforce } from "./reinforce";
import { actorCritic } from "./actor-critic";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const S0 = 0; // start state (0,0)

// ─── SoftmaxPolicy ────────────────────────────────────────────────────────────

describe("SoftmaxPolicy.probs", () => {
  test("sums to 1 for any theta", () => {
    const pol = new SoftmaxPolicy(mdp.nS, mdp.nA);
    const rng = mulberry32(42);
    for (let s = 0; s < mdp.nS; s++) {
      for (let a = 0; a < mdp.nA; a++) {
        pol.theta[s * mdp.nA + a] = (rng() - 0.5) * 4;
      }
      const p = pol.probs(s);
      const sum = p.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 10);
    }
  });

  test("uniform theta yields uniform probs", () => {
    const pol = new SoftmaxPolicy(mdp.nS, mdp.nA);
    for (let s = 0; s < mdp.nS; s++) {
      const p = pol.probs(s);
      for (let a = 0; a < mdp.nA; a++) {
        expect(p[a]).toBeCloseTo(1 / mdp.nA, 10);
      }
    }
  });

  test("stable with large theta values (no NaN/Inf)", () => {
    const pol = new SoftmaxPolicy(1, 4);
    pol.theta[0] = 1000;
    pol.theta[1] = 900;
    pol.theta[2] = 800;
    pol.theta[3] = 700;
    const p = pol.probs(0);
    for (let a = 0; a < 4; a++) {
      expect(isFinite(p[a])).toBe(true);
      expect(p[a]).toBeGreaterThanOrEqual(0);
    }
    const sum = p.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });
});

describe("SoftmaxPolicy.scoreFunction", () => {
  test("sums to zero across all actions (unit sum-zero property)", () => {
    const pol = new SoftmaxPolicy(1, 4);
    pol.theta.set([0.1, 0.5, -0.3, 0.0]);
    for (let a = 0; a < 4; a++) {
      const score = pol.scoreFunction(0, a);
      const sum = score.reduce((acc, v) => acc + v, 0);
      expect(sum).toBeCloseTo(0, 10);
    }
  });

  test("taken-action component equals 1 minus its probability", () => {
    const pol = new SoftmaxPolicy(1, 4);
    pol.theta.set([0.2, -0.1, 0.5, 0.3]);
    const p = pol.probs(0);
    for (let a = 0; a < 4; a++) {
      const score = pol.scoreFunction(0, a);
      expect(score[a]).toBeCloseTo(1 - p[a], 10);
      for (let ap = 0; ap < 4; ap++) {
        if (ap !== a) expect(score[ap]).toBeCloseTo(-p[ap], 10);
      }
    }
  });
});

describe("SoftmaxPolicy.toPolicy", () => {
  test("pi rows sum to 1 and match probs()", () => {
    const pol = new SoftmaxPolicy(mdp.nS, mdp.nA);
    const rng = mulberry32(7);
    for (let i = 0; i < pol.theta.length; i++) pol.theta[i] = rng() - 0.5;
    const policy = pol.toPolicy();
    for (let s = 0; s < mdp.nS; s++) {
      const row = policy.pi[s];
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 10);
      const p = pol.probs(s);
      for (let a = 0; a < mdp.nA; a++) {
        expect(row[a]).toBeCloseTo(p[a], 10);
      }
    }
  });
});

// ─── Gaussian score function estimator ────────────────────────────────────────

describe("gaussianScoreEstimator", () => {
  test("unbiased at theta=1.0, N=10000: estimate within 0.5 of true 2.0", () => {
    const rng = mulberry32(0);
    const estimate = gaussianScoreEstimator(1.0, 10000, rng);
    expect(estimate).toBeCloseTo(gaussianTrueGradient(1.0), 0); // within 0.5
  });

  test("true gradient is 2*theta", () => {
    expect(gaussianTrueGradient(0)).toBe(0);
    expect(gaussianTrueGradient(1)).toBe(2);
    expect(gaussianTrueGradient(-2)).toBe(-4);
  });

  test("estimate sign matches true gradient direction", () => {
    const rng = mulberry32(1);
    const pos = gaussianScoreEstimator(2.0, 5000, rng);
    expect(pos).toBeGreaterThan(0);
    const rng2 = mulberry32(1);
    const neg = gaussianScoreEstimator(-2.0, 5000, rng2);
    expect(neg).toBeLessThan(0);
  });
});

// ─── REINFORCE ────────────────────────────────────────────────────────────────

describe("reinforce (vanilla)", () => {
  test("exact V(0,0) in [0.70, 0.74] after N=2000, alpha=0.1, seed=0", () => {
    const { policy } = reinforce(mdp, 2000, 0.1, { rng: mulberry32(0) });
    const v = policyEvaluationExact(mdp, policy.toPolicy())[S0];
    expect(v).toBeGreaterThan(0.70);
    expect(v).toBeLessThan(0.74);
  });

  test("returns converge upward vs random baseline", () => {
    const { history } = reinforce(mdp, 500, 0.05, { rng: mulberry32(0) });
    const early = history.slice(0, 50).reduce((a, b) => a + b) / 50;
    const late = history.slice(-50).reduce((a, b) => a + b) / 50;
    expect(late).toBeGreaterThan(early);
  });
});

describe("reinforce with baseline", () => {
  test("exact V(0,0) in [0.70, 0.74] after N=2000, alpha=0.1, seed=0", () => {
    const { policy } = reinforce(mdp, 2000, 0.1, {
      rng: mulberry32(0),
      useBaseline: true,
    });
    const v = policyEvaluationExact(mdp, policy.toPolicy())[S0];
    expect(v).toBeGreaterThan(0.70);
    expect(v).toBeLessThan(0.74);
  });

  test("baseline reduces variance: std < 0.5 × vanilla std across 10 seeds (exact V)", () => {
    const vanilla: number[] = [];
    const baseline: number[] = [];
    for (let s = 0; s < 10; s++) {
      const { policy: pv } = reinforce(mdp, 2000, 0.1, { rng: mulberry32(s) });
      vanilla.push(policyEvaluationExact(mdp, pv.toPolicy())[S0]);
      const { policy: pb } = reinforce(mdp, 2000, 0.1, {
        rng: mulberry32(s),
        useBaseline: true,
      });
      baseline.push(policyEvaluationExact(mdp, pb.toPolicy())[S0]);
    }
    const std = (arr: number[]) => {
      const mean = arr.reduce((a, b) => a + b) / arr.length;
      return Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
    };
    expect(std(baseline)).toBeLessThan(std(vanilla) * 0.5);
  });
});

// ─── Actor-Critic ─────────────────────────────────────────────────────────────

describe("actorCritic", () => {
  test("exact V(0,0) in [0.70, 0.74] after N=2000, alpha_a=0.1, alpha_c=0.2, seed=0", () => {
    const { policy } = actorCritic(mdp, 2000, 0.1, 0.2, {
      rng: mulberry32(0),
    });
    const v = policyEvaluationExact(mdp, policy.toPolicy())[S0];
    expect(v).toBeGreaterThan(0.70);
    expect(v).toBeLessThan(0.74);
  });

  test("critic approximates V^pi after 5000 episodes (within 0.12 at each state)", () => {
    const { policy, critic } = actorCritic(mdp, 5000, 0.1, 0.2, {
      rng: mulberry32(0),
    });
    const truePi = policyEvaluationExact(mdp, policy.toPolicy());
    for (let s = 0; s < mdp.nS; s++) {
      if (mdp.terminals[s]) continue;
      expect(Math.abs(critic[s] - truePi[s])).toBeLessThan(0.12);
    }
  });
});
