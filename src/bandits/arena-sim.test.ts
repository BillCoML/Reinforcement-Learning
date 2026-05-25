import { describe, expect, test } from "vitest";
import { ArenaSim, type ArenaConfig } from "./arena-sim";

function defaultConfig(seeds: number): ArenaConfig {
  return {
    K: 3,
    means: [0.3, 0.5, 0.7],
    family: "bernoulli",
    gaussianSigma: 0.3,
    betaConcentration: 6,
    T: 5000,
    seeds,
    baseSeed: 1,
    algos: [
      { key: "random", kind: "random", label: "Random" },
      { key: "eps01", kind: "epsilon", label: "ε-greedy(0.10)", epsilon: 0.1 },
      { key: "ucb1", kind: "ucb1", label: "UCB1" },
      { key: "thompson", kind: "thompson", label: "Thompson" },
    ],
  };
}

describe("ArenaSim", () => {
  test("in-browser sim reproduces the offline final regrets to ~1.5%", () => {
    const sim = new ArenaSim(defaultConfig(200));
    sim.advance(5000, 1e9); // run fully, no budget cutoff
    expect(sim.done).toBe(true);
    const byKey = Object.fromEntries(sim.results().map((r) => [r.key, r]));

    // Targets from the §7 table / offline sim.
    expect(byKey.random.finalRegret).toBeGreaterThan(900);
    expect(byKey.random.finalRegret).toBeLessThan(1100);
    expect(byKey.eps01.finalRegret).toBeGreaterThan(95);
    expect(byKey.eps01.finalRegret).toBeLessThan(130);
    expect(byKey.ucb1.finalRegret).toBeGreaterThan(70);
    expect(byKey.ucb1.finalRegret).toBeLessThan(98);
    expect(byKey.thompson.finalRegret).toBeGreaterThan(10);
    expect(byKey.thompson.finalRegret).toBeLessThan(26);

    // Thompson should concentrate the most on the optimal arm.
    expect(byKey.thompson.pctOptimal).toBeGreaterThan(byKey.ucb1.pctOptimal);
    expect(byKey.ucb1.pctOptimal).toBeGreaterThan(byKey.random.pctOptimal);
  });

  test("frontier advances incrementally and budget caps a frame", () => {
    const sim = new ArenaSim(defaultConfig(50));
    sim.advance(100, 1e9);
    expect(sim.frontier).toBe(100);
    const r = sim.results();
    expect(r[0].mean.length).toBe(100);
  });
});
