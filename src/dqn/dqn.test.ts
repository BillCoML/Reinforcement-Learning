import { describe, expect, test } from "vitest";
import { mulberry32 } from "../importance-sampling/gaussian";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { linearTDPrediction, featurizeRowColBias, featurizeOneHot } from "./linear-fa";
import { bairdCounterexample } from "./baird";
import { runMaxBias } from "./max-bias";
import { policyEvaluationExact } from "../mdp/policy-evaluation";

function seeded(s: number) {
  return mulberry32(s);
}

// ---------------------------------------------------------------------------
// Linear FA convergence
// ---------------------------------------------------------------------------

describe("linearTDPrediction", () => {
  test("bias term θ₂ converges to V^π(0,0) with (r,c,1) features", () => {
    // phi(0,0) = [0,0,1] so V_θ(0,0) = θ_bias; must match true V^π(0,0)
    const mdp = buildGridworld({ gamma: 0.9 });
    const policy = uniformPolicy(mdp);
    const trueV = policyEvaluationExact(mdp, policy);
    const { theta } = linearTDPrediction(
      mdp,
      policy,
      (s) => featurizeRowColBias(3, s),
      10_000,
      0.05,
      { rng: seeded(0) },
    );
    // θ₂ ≈ V^π(0,0) ≈ −0.42 (state 0 is always visited; well-converged)
    expect(theta[2]).toBeCloseTo(trueV[0], 1);
    // θ₁ (col coeff) should not be strongly negative
    expect(theta[1]).toBeGreaterThan(-0.15);
    // Approximation at start state should be close to true value
    const vApprox = theta[2]; // phi(0,0) = [0,0,1]
    expect(vApprox).toBeCloseTo(trueV[0], 1);
  });

  test("one-hot features converge closer to true V^π than row-col-bias", () => {
    // One-hot FA = tabular; should approximate V^π much better than linear
    const mdp = buildGridworld({ gamma: 0.9 });
    const policy = uniformPolicy(mdp);
    const trueV = policyEvaluationExact(mdp, policy);

    const { theta: thetaOneHot } = linearTDPrediction(
      mdp, policy,
      (s) => featurizeOneHot(mdp.nS, s),
      30_000, 0.01,
      { rng: seeded(1), randomStart: true },
    );
    const { theta: thetaLinear } = linearTDPrediction(
      mdp, policy,
      (s) => featurizeRowColBias(3, s),
      30_000, 0.01,
      { rng: seeded(1), randomStart: true },
    );

    // Compute WMSE for both
    let mseOneHot = 0, mseLinear = 0, n = 0;
    for (let s = 0; s < mdp.nS; s++) {
      if (mdp.terminals[s]) continue;
      // one-hot: V_θ(s) = theta[s]
      // one-hot: V_θ(s) = theta[s]
      mseOneHot += (thetaOneHot[s] - trueV[s]) ** 2;
      const r = Math.floor(s / 3), c = s % 3;
      const vLinear = thetaLinear[0] * r + thetaLinear[1] * c + thetaLinear[2];
      mseLinear += (vLinear - trueV[s]) ** 2;
      n++;
    }
    // One-hot should have lower error than row-col-bias
    expect(mseOneHot / n).toBeLessThan(mseLinear / n);
    // State 0 is always well-converged for both
    expect(thetaOneHot[0]).toBeCloseTo(trueV[0], 1);
  });
});

// ---------------------------------------------------------------------------
// Baird's counterexample
// ---------------------------------------------------------------------------

describe("bairdCounterexample", () => {
  test("initial ‖θ‖ is approximately 10.3", () => {
    const { normHistory } = bairdCounterexample(0, { rng: seeded(0) });
    expect(normHistory[0]).toBeCloseTo(10.3, 0);
  });

  test("diverges with full deadly triad: ‖θ‖ at 2000 > 2000", () => {
    const { normHistory } = bairdCounterexample(2000, { alpha: 0.01, rng: seeded(0) });
    expect(normHistory[100]).toBeLessThan(50);
    expect(normHistory[1000]).toBeGreaterThan(200);
    expect(normHistory[2000]).toBeGreaterThan(2000);
  });

  test("bounded without off-policy (onPolicy=true): ‖θ‖ at 2000 < 50", () => {
    const { normHistory } = bairdCounterexample(2000, {
      alpha: 0.01,
      onPolicy: true,
      rng: seeded(0),
    });
    expect(normHistory[2000]).toBeLessThan(50);
  });

  test("converges without bootstrapping (MC): ‖θ‖ at 2000 < 15", () => {
    const { normHistory } = bairdCounterexample(2000, {
      alpha: 0.01,
      bootstrap: false,
      rng: seeded(0),
    });
    expect(normHistory[2000]).toBeLessThan(15);
  });

  test("divergence is reproducible across seeds", () => {
    // Run 5 seeds; all should diverge
    for (let seed = 0; seed < 5; seed++) {
      const { normHistory } = bairdCounterexample(2000, {
        alpha: 0.01,
        rng: seeded(seed),
      });
      expect(normHistory[2000]).toBeGreaterThan(500);
    }
  });
});

// ---------------------------------------------------------------------------
// Double DQN max-bias reduction
// ---------------------------------------------------------------------------

describe("maxBias", () => {
  test("DQN shows systematic overestimation at episode 300 (>20% wrong)", () => {
    const result = runMaxBias({
      nEpisodes: 300,
      nSeeds: 500,
      masterSeed: 42,
      useDoubleDqn: false,
    });
    const lastFrac = result.leftFractionPerEpisode.slice(-10).reduce((a, b) => a + b, 0) / 10;
    expect(lastFrac).toBeGreaterThan(0.20);
  });

  test("Double DQN reduces bias at episode 300 (<15% wrong)", () => {
    const result = runMaxBias({
      nEpisodes: 300,
      nSeeds: 500,
      masterSeed: 42,
      useDoubleDqn: true,
    });
    const lastFrac = result.leftFractionPerEpisode.slice(-10).reduce((a, b) => a + b, 0) / 10;
    expect(lastFrac).toBeLessThan(0.15);
  });

  test("Double DQN is less biased than DQN at episode 300", () => {
    const dqn = runMaxBias({ nEpisodes: 300, nSeeds: 200, masterSeed: 7 });
    const ddqn = runMaxBias({ nEpisodes: 300, nSeeds: 200, masterSeed: 7, useDoubleDqn: true });
    const dqnFrac = dqn.leftFractionPerEpisode.slice(-10).reduce((a, b) => a + b, 0) / 10;
    const ddqnFrac = ddqn.leftFractionPerEpisode.slice(-10).reduce((a, b) => a + b, 0) / 10;
    expect(ddqnFrac).toBeLessThan(dqnFrac);
  });
});
