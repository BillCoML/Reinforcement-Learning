import { expect, test, describe } from "vitest";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { bellmanExpectationBackup } from "../mdp/policy-evaluation";
import { bellmanOptimalityBackup } from "../mdp/value-iteration";
import { supDist } from "./ops";

describe("§5 spec table — T^π on 3×3 gridworld, γ=0.9", () => {
  test("sup-distances match spec to 3 decimal places", () => {
    const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    const uniform = uniformPolicy(mdp);
    let V1 = new Array(9).fill(0);
    let V2 = new Array(9).fill(2);

    const expected = [2.0000, 1.8000, 1.6200, 1.2758, 1.0252, 0.8119];
    const dists = [supDist(V1, V2)];
    for (let k = 0; k < 10; k++) {
      V1 = bellmanExpectationBackup(mdp, uniform, V1);
      V2 = bellmanExpectationBackup(mdp, uniform, V2);
      dists.push(supDist(V1, V2));
    }

    for (let k = 0; k < 6; k++) {
      expect(dists[k]).toBeCloseTo(expected[k], 3);
    }
    expect(dists[10]).toBeCloseTo(0.2305, 3);

    // Empirical ratios ≤ γ at every step
    for (let k = 1; k <= 10; k++) {
      expect(dists[k] / dists[k - 1]).toBeLessThanOrEqual(0.9 + 1e-9);
    }
  });

  test("T^* — ratios ≤ γ for all 10 steps", () => {
    const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    let V1 = new Array(9).fill(0);
    let V2 = new Array(9).fill(2);
    let prev = supDist(V1, V2);
    for (let k = 0; k < 10; k++) {
      V1 = bellmanOptimalityBackup(mdp, V1);
      V2 = bellmanOptimalityBackup(mdp, V2);
      const curr = supDist(V1, V2);
      expect(curr / prev).toBeLessThanOrEqual(0.9 + 1e-9);
      prev = curr;
    }
  });

  test("actual curve ≤ γ^k * d0 bound at every k", () => {
    const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    const uniform = uniformPolicy(mdp);
    let V1 = new Array(9).fill(0);
    let V2 = new Array(9).fill(2);
    const d0 = supDist(V1, V2);
    for (let k = 1; k <= 15; k++) {
      V1 = bellmanExpectationBackup(mdp, uniform, V1);
      V2 = bellmanExpectationBackup(mdp, uniform, V2);
      const actual = supDist(V1, V2);
      const bound = Math.pow(0.9, k) * d0;
      expect(actual).toBeLessThanOrEqual(bound + 1e-9);
    }
  });
});
