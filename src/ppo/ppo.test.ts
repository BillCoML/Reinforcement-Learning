import { describe, test, expect } from "vitest";
import { buildGridworld } from "../mdp/gridworld";
import { gaeAdvantages } from "./gae";
import { softmax, policyKL, runPPO } from "./ppo";
import { runVanilla } from "./vanilla";
import type { Trajectory } from "./types";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

// ─── softmax ─────────────────────────────────────────────────────────────────

describe("softmax row sums to 1", () => {
  test("uniform logits", () => {
    const logits = new Float64Array([0, 0, 0, 0]);
    const p = softmax(logits, 4);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    for (let a = 0; a < 4; a++) expect(p[a]).toBeCloseTo(0.25, 10);
  });

  test("arbitrary logits sum to 1", () => {
    const logits = new Float64Array([1.5, -0.3, 0.8, -1.2]);
    const p = softmax(logits, 4);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  test("large logit values stay stable (no NaN)", () => {
    const logits = new Float64Array([1000, 900, 800, 700]);
    const p = softmax(logits, 4);
    for (let a = 0; a < 4; a++) {
      expect(isFinite(p[a])).toBe(true);
      expect(p[a]).toBeGreaterThanOrEqual(0);
    }
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });
});

// ─── policyKL ────────────────────────────────────────────────────────────────

describe("policyKL", () => {
  test("is non-negative", () => {
    const thetaOld = new Float64Array(mdp.nS * mdp.nA).map(() => Math.random() - 0.5);
    const thetaNew = new Float64Array(mdp.nS * mdp.nA).map(() => Math.random() - 0.5);
    const states = new Set([0, 1, 2]);
    const kl = policyKL(thetaOld, thetaNew, states, mdp.nA);
    expect(kl).toBeGreaterThanOrEqual(0);
  });

  test("equals zero when policies are identical", () => {
    const theta = new Float64Array([0.5, -0.3, 0.1, 0.2]);
    const states = new Set([0]);
    const kl = policyKL(theta, theta.slice(), states, 4);
    expect(kl).toBeCloseTo(0, 10);
  });
});

// ─── gaeAdvantages ────────────────────────────────────────────────────────────

describe("gaeAdvantages collapses to TD residual at lambda=0", () => {
  test("single-step trajectory: Â_0 = r + γV(s') - V(s)", () => {
    const V = new Float64Array(mdp.nS).fill(0);
    V[0] = 0.5;
    V[1] = 0.3;
    const traj: Trajectory = {
      states: [0, 1],
      actions: [1],
      rewards: [0.1],
    };
    const adv = gaeAdvantages(traj, V, mdp.gamma, 0, mdp);
    const expected = 0.1 + mdp.gamma * V[1] - V[0];
    expect(adv[0]).toBeCloseTo(expected, 10);
  });
});

describe("gaeAdvantages collapses to MC at lambda=1", () => {
  test("5-step trajectory with V=0: Â_t = G_t", () => {
    const V = new Float64Array(mdp.nS).fill(0);
    const rewards = [0.1, -0.2, 0.3, -0.1, 0.5];
    const states = [0, 1, 2, 3, 5, 8]; // 8 = goal (terminal)
    const traj: Trajectory = {
      states,
      actions: [0, 1, 2, 3, 0],
      rewards,
    };
    const adv = gaeAdvantages(traj, V, mdp.gamma, 1, mdp);
    // GAE(1) = G_t - V(s_t) = G_t (since V=0)
    const gamma = mdp.gamma;
    const G = [
      rewards[0] + gamma * (rewards[1] + gamma * (rewards[2] + gamma * (rewards[3] + gamma * rewards[4]))),
      rewards[1] + gamma * (rewards[2] + gamma * (rewards[3] + gamma * rewards[4])),
      rewards[2] + gamma * (rewards[3] + gamma * rewards[4]),
      rewards[3] + gamma * rewards[4],
      rewards[4],
    ];
    for (let t = 0; t < 5; t++) {
      expect(adv[t]).toBeCloseTo(G[t], 6);
    }
  });
});

// ─── runPPO ───────────────────────────────────────────────────────────────────

describe("runPPO matches numerical anchor", () => {
  test("200 iters lr=0.5 batch=20 seed=0 → vStart > 0.71", () => {
    const { logs } = runPPO(
      {
        lrPolicy: 0.5,
        lrValue: 0.5,
        clipEps: 0.2,
        gaeLambda: 0.95,
        epochs: 4,
        batchEpisodes: 20,
        entropyCoef: 0.0,
        valueCoef: 0.5,
        normalizeAdvantages: true,
      },
      200,
      0,
      mdp,
    );
    const finalV = logs[logs.length - 1].vStart;
    // Seed-0 single trial should be near the 10-seed mean of 0.7267
    expect(finalV).toBeGreaterThan(0.71);
    expect(finalV).toBeLessThan(0.73);
  }, 30000);
});

describe("runVanilla diverges quietly at lr=0.1", () => {
  test("200 iters lr=0.1 batch=10 seed=0 → final vStart < 0", () => {
    const { logs } = runVanilla(
      {
        lrPolicy: 0.1,
        lrValue: 0.1,
        gaeLambda: 0.95,
        normalizeAdvantages: true,
        batchEpisodes: 10,
      },
      200,
      0,
      mdp,
    );
    const finalV = logs[logs.length - 1].vStart;
    expect(finalV).toBeLessThan(0.0);
  }, 30000);
});

describe("runPPO ratio histogram bounded", () => {
  test("max ratio bounded loosely at any iteration", () => {
    const config = {
      lrPolicy: 0.5,
      lrValue: 0.5,
      clipEps: 0.2,
      gaeLambda: 0.95,
      epochs: 4,
      batchEpisodes: 20,
      entropyCoef: 0.0,
      valueCoef: 0.5,
      normalizeAdvantages: true,
    };
    const { logs } = runPPO(config, 50, 42, mdp);
    for (const log of logs) {
      // Max ratio should be finite and not blow up catastrophically.
      expect(isFinite(log.maxRatio)).toBe(true);
      expect(log.maxRatio).toBeLessThan(20);
    }
  }, 20000);
});

describe("ppoUpdate sanity", () => {
  test("theta changes after one update on a non-trivial batch", () => {
    const { logs, finalState } = runPPO(
      {
        lrPolicy: 1.0,
        lrValue: 0.5,
        clipEps: 0.2,
        gaeLambda: 0.95,
        epochs: 4,
        batchEpisodes: 5,
        entropyCoef: 0.0,
        valueCoef: 0.5,
        normalizeAdvantages: true,
      },
      1,
      123,
      mdp,
    );
    // After one update, theta should have changed from zero.
    const anyNonZero = Array.from(finalState.theta).some((v) => v !== 0);
    expect(anyNonZero).toBe(true);
    expect(logs.length).toBe(1);
  }, 10000);
});
