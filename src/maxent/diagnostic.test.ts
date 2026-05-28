import { describe, it, expect } from "vitest";
import { buildGridworld } from "../mdp/gridworld";
import { softValueIteration } from "./softVI";
import { monteCarloDiagnostic } from "./diagnostic";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

describe("monteCarloDiagnostic", () => {
  it("goalReachProb + pitReachProb + timeoutProb ≈ 1", () => {
    const { pi } = softValueIteration(0.05, mdp);
    const d = monteCarloDiagnostic(pi, mdp, { nRollouts: 500, maxSteps: 200, seed: 7 });
    const total = d.goalReachProb + d.pitReachProb + d.timeoutProb;
    expect(Math.abs(total - 1)).toBeLessThan(1e-10);
  });

  it("same seed produces identical results", () => {
    const { pi } = softValueIteration(0.1, mdp);
    const d1 = monteCarloDiagnostic(pi, mdp, { nRollouts: 500, maxSteps: 200, seed: 99 });
    const d2 = monteCarloDiagnostic(pi, mdp, { nRollouts: 500, maxSteps: 200, seed: 99 });
    expect(d1.goalReachProb).toBe(d2.goalReachProb);
    expect(d1.meanStepsToTerminal).toBe(d2.meanStepsToTerminal);
  });

  it("histogram sums to 1", () => {
    const { pi } = softValueIteration(0.05, mdp);
    const d = monteCarloDiagnostic(pi, mdp, { nRollouts: 200, maxSteps: 100, seed: 1 });
    let total = 0;
    for (let i = 0; i < d.lengthHistogram.length; i++) total += d.lengthHistogram[i];
    expect(Math.abs(total - 1)).toBeLessThan(1e-10);
  });
});
