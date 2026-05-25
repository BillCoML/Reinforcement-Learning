import { describe, expect, test } from "vitest";
import { ThompsonBetaBernoulli, UCB1, RandomPolicy } from "./algorithms";
import {
  bernoulliKL,
  betaMoments,
  hoeffdingBound,
  laiRobbinsConstant,
  mulberry32,
  sampleBeta,
} from "./stats";
import { BernoulliBandit } from "./env";
import { RegretTracker } from "./regret";

describe("UCB1", () => {
  test("trace matches hand calculation at t=4", () => {
    // After pulls (0,1,0) on arms 1,2,3, UCB1 picks arm 2 at the 4th pull,
    // because UCB = (1.6651, 2.6651, 1.6651).
    const algo = new UCB1();
    algo.reset(3);
    algo.update(0, 0);
    algo.update(1, 1);
    algo.update(2, 0);
    expect(algo.selectArm(3)).toBe(1); // 0-indexed → arm 2
  });

  test("bonus and UCB values match the §5 table", () => {
    const algo = new UCB1();
    algo.reset(3);
    algo.update(0, 0);
    algo.update(1, 1);
    algo.update(2, 0);
    const ucb4 = algo.ucb(3); // 4th pull → ln 4
    expect(algo.bonuses(3)[0]).toBeCloseTo(1.6651, 4);
    expect(ucb4[1]).toBeCloseTo(2.6651, 4);

    // Pull arm 2 again, observe reward 1 → N_2=2, μ̂_2=1.
    algo.update(1, 1);
    const ucb5 = algo.ucb(4); // 5th pull → ln 5
    expect(algo.bonuses(4)[0]).toBeCloseTo(1.7941, 4);
    expect(ucb5[1]).toBeCloseTo(2.2686, 4);
    expect(algo.selectArm(4)).toBe(1);
  });
});

describe("Thompson sampling", () => {
  test("Beta-Bernoulli posterior parameters after 7s/3f", () => {
    const algo = new ThompsonBetaBernoulli();
    algo.reset(1);
    for (let i = 0; i < 7; i++) algo.update(0, 1);
    for (let i = 0; i < 3; i++) algo.update(0, 0);
    expect(algo.state().extra!.alpha).toStrictEqual([8]);
    expect(algo.state().extra!.beta).toStrictEqual([4]);
  });
});

describe("stats", () => {
  test("Lai-Robbins constant for running example", () => {
    expect(laiRobbinsConstant([0.3, 0.5, 0.7])).toBeCloseTo(3.4744, 3);
  });

  test("Bernoulli KL matches §2 table", () => {
    expect(bernoulliKL(0.3, 0.7)).toBeCloseTo(0.338919, 5);
    expect(bernoulliKL(0.5, 0.7)).toBeCloseTo(0.087177, 5);
  });

  test("Hoeffding bound", () => {
    expect(hoeffdingBound(100, 0.1)).toBeCloseTo(0.270671, 5);
  });

  test("Beta(8,4) moments match §6 table", () => {
    const { mean, std } = betaMoments(8, 4);
    expect(mean).toBeCloseTo(0.667, 3);
    expect(std).toBeCloseTo(0.131, 3);
  });

  test("sampleBeta is in (0,1) and roughly tracks its mean", () => {
    const rng = mulberry32(7);
    let sum = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
      const x = sampleBeta(8, 4, rng);
      expect(x).toBeGreaterThan(0);
      expect(x).toBeLessThan(1);
      sum += x;
    }
    expect(sum / n).toBeCloseTo(8 / 12, 1); // mean ≈ 0.667
  });
});

describe("regret tracking", () => {
  test("gap decomposition reproduces the §2 numerical example", () => {
    // Pulled arm 1 ×50, arm 2 ×30, arm 3 ×420 → R_500 = 26.0
    const env = new BernoulliBandit([0.3, 0.5, 0.7]);
    const tracker = new RegretTracker();
    for (let i = 0; i < 50; i++) tracker.observe(env, 0);
    for (let i = 0; i < 30; i++) tracker.observe(env, 1);
    for (let i = 0; i < 420; i++) tracker.observe(env, 2);
    expect(tracker.current()).toBeCloseTo(26.0, 6);
  });
});

describe("end-to-end sanity", () => {
  test("Thompson beats random on the running example over a long horizon", () => {
    const means = [0.3, 0.5, 0.7];
    function finalRegret(algoFactory: () => RandomPolicy | ThompsonBetaBernoulli): number {
      const rng = mulberry32(123);
      const env = new BernoulliBandit(means, rng);
      const algo = algoFactory();
      algo.rng = rng;
      algo.reset(3);
      const tracker = new RegretTracker();
      for (let t = 0; t < 3000; t++) {
        const arm = algo.selectArm(t);
        const r = env.pull(arm);
        algo.update(arm, r);
        tracker.observe(env, arm);
      }
      return tracker.current();
    }
    const ts = finalRegret(() => new ThompsonBetaBernoulli());
    const rand = finalRegret(() => new RandomPolicy());
    expect(ts).toBeLessThan(rand);
    expect(ts).toBeLessThan(60); // log-shaped, comfortably sub-linear
  });
});
