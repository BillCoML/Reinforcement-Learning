import { describe, expect, test } from "vitest";
import { MarkovChain, totalVariation, gcd } from "./chain";
import { weather, periodic2, reducible, birthDeath } from "./presets";
import { mulberry32 } from "../bandits/stats";

describe("MarkovChain — construction", () => {
  test("rejects non-row-stochastic matrices", () => {
    expect(() => new MarkovChain([[0.5, 0.4]])).toThrow();
    expect(() => new MarkovChain([[0.5, 0.5, 0.1]])).toThrow(); // non-square
  });
});

describe("MarkovChain — weather chain", () => {
  const chain = new MarkovChain(weather.P);

  test("stationary distribution π = (21/46, 13/46, 12/46)", () => {
    const pi = chain.stationary();
    expect(pi[0]).toBeCloseTo(21 / 46, 6);
    expect(pi[1]).toBeCloseTo(13 / 46, 6);
    expect(pi[2]).toBeCloseTo(12 / 46, 6);
  });

  test("second eigenvalue magnitude λ⋆ = 0.473205", () => {
    expect(chain.lambdaStar()).toBeCloseTo(0.473205, 5);
  });

  test("(P²)₀₀ = 0.57 — the §1 hand calculation", () => {
    expect(chain.pPower(2).get(0, 0)).toBeCloseTo(0.57, 6);
  });

  test("P⁵ and P²⁰ match the §2 table", () => {
    const p5 = chain.pPower(5);
    expect(p5.get(0, 0)).toBeCloseTo(0.468, 3);
    expect(p5.get(0, 1)).toBeCloseTo(0.279, 3);
    expect(p5.get(0, 2)).toBeCloseTo(0.252, 3);
    const p20 = chain.pPower(20);
    // Rows of P²⁰ have collapsed to π.
    for (let i = 0; i < 3; i++) {
      expect(p20.get(i, 0)).toBeCloseTo(0.457, 3);
      expect(p20.get(i, 1)).toBeCloseTo(0.283, 3);
      expect(p20.get(i, 2)).toBeCloseTo(0.261, 3);
    }
  });

  test("is irreducible and aperiodic (ergodic)", () => {
    expect(chain.isIrreducible()).toBe(true);
    expect(chain.isAperiodic()).toBe(true);
    expect(chain.communicatingClasses().length).toBe(1);
  });

  test("convergence rate: consecutive TV ratio → λ⋆ = 0.4732", () => {
    // The ratio approaches λ⋆ from below as the subdominant (0.1268) mode
    // dies off; by n≈7 it has converged to the spectral rate.
    const tv7 = chain.tvToStationary([1, 0, 0], 7);
    const tv8 = chain.tvToStationary([1, 0, 0], 8);
    expect(tv8 / tv7).toBeCloseTo(0.4732, 3);
  });

  test("TV distance from sunny matches §5 table", () => {
    // Row 0 of the pre-verified table.
    expect(chain.tvToStationary([1, 0, 0], 0)).toBeCloseTo(0.5435, 4);
    expect(chain.tvToStationary([1, 0, 0], 1)).toBeCloseTo(0.2435, 4);
    expect(chain.tvToStationary([1, 0, 0], 2)).toBeCloseTo(0.1135, 4);
    expect(chain.tvToStationary([1, 0, 0], 3)).toBeCloseTo(0.0535, 4);
    expect(chain.tvToStationary([1, 0, 0], 5)).toBeCloseTo(0.012, 3);
  });

  test("mixing time bound at ε=0.01 is ≈ 8.74 steps", () => {
    expect(chain.mixingTimeBound(0.01)).toBeCloseTo(8.74, 1);
  });
});

describe("MarkovChain — periodic chain", () => {
  const chain = new MarkovChain(periodic2.P);

  test("period of state 0 is 2 and chain is not aperiodic", () => {
    expect(chain.period(0)).toBe(2);
    expect(chain.isAperiodic()).toBe(false);
  });

  test("is irreducible with π = (0.5, 0.5)", () => {
    expect(chain.isIrreducible()).toBe(true);
    const pi = chain.stationary();
    expect(pi[0]).toBeCloseTo(0.5, 6);
    expect(pi[1]).toBeCloseTo(0.5, 6);
  });

  test("λ⋆ = 1 (eigenvalue on the unit circle → no convergence)", () => {
    expect(chain.lambdaStar()).toBeCloseTo(1, 6);
  });
});

describe("MarkovChain — reducible chain", () => {
  const chain = new MarkovChain(reducible.P);

  test("has two communicating classes and is not irreducible", () => {
    const classes = chain.communicatingClasses();
    expect(classes.length).toBe(2);
    expect(chain.isIrreducible()).toBe(false);
  });

  test("classify: {0,1} recurrent, {2,3} transient", () => {
    const { classes, stateToClass } = chain.classify();
    const classOf01 = classes[stateToClass[0]];
    const classOf23 = classes[stateToClass[2]];
    expect(classOf01.members).toEqual([0, 1]);
    expect(classOf01.recurrent).toBe(true);
    expect(classOf23.members).toEqual([2, 3]);
    expect(classOf23.recurrent).toBe(false);
  });

  test("P⁵⁰ from transient state 2 has leaked into {0,1}", () => {
    const dist = chain.distributionAfter([0, 0, 1, 0], 50);
    expect(dist[0]).toBeCloseTo(0.571, 2);
    expect(dist[1]).toBeCloseTo(0.428, 2);
    // Transient mass has all but vanished (≈1e-4 at n=50; the §3 table rounds to 0.000).
    expect(dist[2]).toBeLessThan(1e-3);
    expect(dist[3]).toBeLessThan(1e-3);
  });
});

describe("MarkovChain — detailed balance", () => {
  test("birth-death chain is reversible (residual < 1e-10)", () => {
    const chain = new MarkovChain(birthDeath.P);
    const pi = [6 / 21, 10 / 21, 5 / 21];
    expect(chain.detailedBalanceResidual(pi)).toBeLessThan(1e-10);
  });

  test("birth-death stationary distribution = (6/21, 10/21, 5/21)", () => {
    const chain = new MarkovChain(birthDeath.P);
    const pi = chain.stationary();
    expect(pi[0]).toBeCloseTo(6 / 21, 6);
    expect(pi[1]).toBeCloseTo(10 / 21, 6);
    expect(pi[2]).toBeCloseTo(5 / 21, 6);
  });

  test("weather chain is NOT reversible (residual well above tolerance)", () => {
    const chain = new MarkovChain(weather.P);
    const pi = chain.stationary();
    expect(chain.detailedBalanceResidual(pi)).toBeGreaterThan(1e-3);
  });
});

describe("MarkovChain — sampling", () => {
  test("empirical visit frequencies converge to π for the weather chain", () => {
    const chain = new MarkovChain(weather.P);
    const rng = mulberry32(42);
    const path = chain.sampleTrajectory(0, 20000, rng);
    const counts = [0, 0, 0];
    for (const s of path) counts[s]++;
    const freq = counts.map((c) => c / path.length);
    const pi = chain.stationary();
    for (let i = 0; i < 3; i++) expect(freq[i]).toBeCloseTo(pi[i], 1);
  });

  test("trajectory respects the deterministic periodic chain", () => {
    const chain = new MarkovChain(periodic2.P);
    const path = chain.sampleTrajectory(0, 6, mulberry32(1));
    expect(path).toEqual([0, 1, 0, 1, 0, 1]);
  });
});

describe("helpers", () => {
  test("gcd", () => {
    expect(gcd(0, 5)).toBe(5);
    expect(gcd(12, 18)).toBe(6);
    expect(gcd(2, 3)).toBe(1);
  });

  test("totalVariation is half the L1 distance", () => {
    expect(totalVariation([1, 0], [0, 1])).toBeCloseTo(1, 6);
    expect(totalVariation([0.5, 0.5], [0.5, 0.5])).toBeCloseTo(0, 6);
  });
});
