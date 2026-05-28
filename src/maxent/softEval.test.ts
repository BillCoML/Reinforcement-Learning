import { describe, it, expect } from "vitest";
import { buildGridworld } from "../mdp/gridworld";
import { softValueIteration } from "./softVI";
import { softEvaluate } from "./softEval";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const START = 0;

describe("softEvaluate", () => {
  it("V_soft and V_pi are distinct quantities", () => {
    const { pi } = softValueIteration(0.02, mdp);
    const ev = softEvaluate(pi, 0.02, mdp);
    // V_soft includes entropy bonus (> V* = 0.729); V_pi is the true return (< V*).
    expect(ev.V_soft[START]).toBeGreaterThan(0.729);
    expect(ev.V_pi[START]).toBeLessThan(0.729);
    expect(ev.V_soft[START]).toBeGreaterThan(ev.V_pi[START]);
  });

  it("V_pi at alpha=0.02 reproduces L10 softmax cap (≈ 0.7217)", () => {
    const { pi } = softValueIteration(0.02, mdp);
    const ev = softEvaluate(pi, 0.02, mdp);
    expect(ev.V_pi[START]).toBeCloseTo(0.7217, 2);
  });

  it("terminal states have V_soft = V_pi = 0", () => {
    const { pi } = softValueIteration(0.1, mdp);
    const ev = softEvaluate(pi, 0.1, mdp);
    for (let s = 0; s < mdp.nS; s++) {
      if (mdp.terminals[s]) {
        expect(Math.abs(ev.V_soft[s])).toBeLessThan(1e-10);
        expect(Math.abs(ev.V_pi[s])).toBeLessThan(1e-10);
      }
    }
  });
});
