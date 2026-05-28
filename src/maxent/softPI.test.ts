import { describe, it, expect } from "vitest";
import { buildGridworld } from "../mdp/gridworld";
import { softValueIteration } from "./softVI";
import { softPolicyIteration } from "./softPI";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

describe("softPolicyIteration", () => {
  it.each([0.05, 0.1, 0.5])("softPI == softVI fixed point at alpha=%f", (alpha) => {
    const vi = softValueIteration(alpha, mdp, { tol: 1e-10 });
    const pi = softPolicyIteration(alpha, mdp, { tol: 1e-10 });
    const nSA = mdp.nS * mdp.nA;
    let maxDiff = 0;
    for (let i = 0; i < nSA; i++) {
      const diff = Math.abs(vi.pi[i] - pi.pi[i]);
      if (diff > maxDiff) maxDiff = diff;
    }
    expect(maxDiff).toBeLessThan(1e-6);
  });
});
