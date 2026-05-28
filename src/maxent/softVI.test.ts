import { describe, it, expect } from "vitest";
import { buildGridworld } from "../mdp/gridworld";
import { softValueIteration } from "./softVI";
import { softEvaluate } from "./softEval";
import { monteCarloDiagnostic } from "./diagnostic";
import { logSumExp } from "./logsumexp";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const START = 0;

describe("softValueIteration", () => {
  it("alpha=0.0001 recovers hard VI: V_soft[start] ≈ V* = 0.7290", () => {
    const result = softValueIteration(0.0001, mdp);
    expect(result.converged).toBe(true);
    expect(result.V[START]).toBeCloseTo(0.7290, 2);
  });

  it("alpha=0.02 hits L10 softmax cap: V_pi[start] ≈ 0.7217", () => {
    const result = softValueIteration(0.02, mdp);
    expect(result.converged).toBe(true);
    const eval_ = softEvaluate(result.pi, 0.02, mdp);
    // V_soft is the fixed-point (includes entropy); V_pi is the true return.
    expect(eval_.V_pi[START]).toBeCloseTo(0.7217, 2);
    // V_soft > V* at this alpha (entropy bonus inflates it).
    expect(eval_.V_soft[START]).toBeGreaterThan(0.729);
  });

  it("alpha=0.2 failure: goal reach prob ≤ 0.10", () => {
    const result = softValueIteration(0.2, mdp);
    const diag = monteCarloDiagnostic(result.pi, mdp, { nRollouts: 2000, maxSteps: 500, seed: 42 });
    expect(diag.goalReachProb).toBeLessThanOrEqual(0.10);
  });

  it("alpha=0.5 deep failure: goal reach prob ≤ 0.01", () => {
    const result = softValueIteration(0.5, mdp);
    const diag = monteCarloDiagnostic(result.pi, mdp, { nRollouts: 2000, maxSteps: 500, seed: 42 });
    expect(diag.goalReachProb).toBeLessThanOrEqual(0.01);
  });

  it("Boltzmann policy rows all sum to 1 ± 1e-10", () => {
    const result = softValueIteration(0.05, mdp);
    const { pi } = result;
    const nA = mdp.nA;
    for (let s = 0; s < mdp.nS; s++) {
      let sum = 0;
      for (let a = 0; a < nA; a++) sum += pi[s * nA + a];
      expect(Math.abs(sum - 1)).toBeLessThan(1e-10);
    }
  });

  it("V_soft[start] is monotonically non-decreasing in alpha", () => {
    const alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0];
    let prevV = -Infinity;
    for (const alpha of alphas) {
      const result = softValueIteration(alpha, mdp);
      expect(result.V[START]).toBeGreaterThanOrEqual(prevV - 1e-6);
      prevV = result.V[START];
    }
  });

  it("logSumExp matches analytic formula at alpha=0.05", () => {
    const result = softValueIteration(0.05, mdp);
    const Q_s0 = result.Q.subarray(START * mdp.nA, (START + 1) * mdp.nA);
    const lse = logSumExp(Q_s0, 0.05);
    // Cross-check: manually compute shifted log-sum-exp
    let maxQ = -Infinity;
    for (let a = 0; a < mdp.nA; a++) if (Q_s0[a] > maxQ) maxQ = Q_s0[a];
    let sumExp = 0;
    for (let a = 0; a < mdp.nA; a++) sumExp += Math.exp((Q_s0[a] - maxQ) / 0.05);
    const expected = 0.05 * Math.log(sumExp) + maxQ;
    expect(Math.abs(lse - expected)).toBeLessThan(1e-10);
    // Should match V[start] from the converged result (within tol).
    expect(Math.abs(lse - result.V[START])).toBeLessThan(1e-6);
  });
});
