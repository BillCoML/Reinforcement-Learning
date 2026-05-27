import { describe, test, expect } from 'vitest';
import { buildGridworld, uniformPolicy } from '../mdp/gridworld';
import { policyEvaluationExact } from '../mdp/policy-evaluation';
import { idx } from '../mdp/types';
import {
  policyEvaluationIterative,
  policyImprovement,
  policyIteration,
  valueIteration,
  modifiedPolicyIteration,
  asyncValueIteration,
  supDist,
} from './algorithms';

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

describe('policyEvaluationIterative', () => {
  test('matches exact PE for uniform policy', () => {
    const uniform = uniformPolicy(mdp);
    const exact = policyEvaluationExact(mdp, uniform);
    const { V } = policyEvaluationIterative(mdp, uniform, 1e-10);
    for (let s = 0; s < 9; s++) expect(V[s]).toBeCloseTo(exact[s], 5);
  });

  test('trace has correct length and starts at zero', () => {
    const { trace } = policyEvaluationIterative(mdp, uniformPolicy(mdp));
    expect(trace[0].every(v => v === 0)).toBe(true);
    expect(trace.length).toBeGreaterThan(1);
  });

  test('V_50 at (0,0) matches spec table', () => {
    const uniform = uniformPolicy(mdp);
    const { trace } = policyEvaluationIterative(mdp, uniform, 1e-12);
    expect(trace[50][idx(0, 0)]).toBeCloseTo(-0.4205, 3);
  });
});

describe('policyIteration', () => {
  test('converges in 3 outer steps', () => {
    const { iterations } = policyIteration(mdp);
    expect(iterations).toBe(3);
  });

  test('history has 3 entries with correct V(0,0) sequence', () => {
    const { history } = policyIteration(mdp);
    expect(history).toHaveLength(3);
    expect(history[0].V[idx(0, 0)]).toBeCloseTo(-0.4205, 3);
    expect(history[1].V[idx(0, 0)]).toBeCloseTo(0.0000, 3);
    expect(history[2].V[idx(0, 0)]).toBeCloseTo(0.7290, 3);
  });

  test('first improvement picks Up at (0,0)', () => {
    const { history } = policyIteration(mdp);
    // π₁ is built from history[0].policy improvement → history[1].policy
    // After first improvement (from π₀=uniform evaluated to history[0].V),
    // the new policy (π₁) has action UP at (0,0).
    const V0 = history[0].V;
    const pi1 = policyImprovement(mdp, V0);
    expect(pi1.pi[idx(0, 0)][0]).toBe(1); // UP = index 0
  });

  test('final V matches V* to 5 decimals', () => {
    const pi = policyIteration(mdp);
    const vi = valueIteration(mdp, 1e-12);
    for (let s = 0; s < 9; s++) expect(pi.V[s]).toBeCloseTo(vi.V[s], 5);
  });
});

describe('valueIteration', () => {
  test('converges in 5 iterations (or very close) on gridworld', () => {
    const { iterations } = valueIteration(mdp, 1e-8);
    expect(iterations).toBeLessThanOrEqual(6);
    expect(iterations).toBeGreaterThanOrEqual(4);
  });

  test('V(0,0) ≈ 0.729 at convergence', () => {
    const { V } = valueIteration(mdp, 1e-8);
    expect(V[idx(0, 0)]).toBeCloseTo(0.729, 4);
  });

  test('stopping criterion guarantees ε-optimality', () => {
    const epsilon = 0.01;
    const { V } = valueIteration(mdp, epsilon);
    const optimal = valueIteration(mdp, 1e-12).V;
    for (let s = 0; s < 9; s++) {
      expect(Math.abs(V[s] - optimal[s])).toBeLessThan(epsilon);
    }
  });

  test('trace starts at zero and has correct shape', () => {
    const { trace } = valueIteration(mdp, 1e-8);
    expect(trace[0].every(v => v === 0)).toBe(true);
    expect(trace.length).toBeGreaterThan(1);
  });

  test('k=1 trace: only (1,2) and (2,1) are non-zero', () => {
    const { trace } = valueIteration(mdp, 1e-8);
    expect(trace[1][idx(1, 2)]).toBeCloseTo(1.0, 5);
    expect(trace[1][idx(2, 1)]).toBeCloseTo(1.0, 5);
    expect(trace[1][idx(0, 0)]).toBe(0);
    expect(trace[1][idx(0, 1)]).toBe(0);
    expect(trace[1][idx(1, 0)]).toBe(0);
  });
});

describe('modifiedPolicyIteration', () => {
  test('m=1 converges (VI-style)', () => {
    const { V } = modifiedPolicyIteration(mdp, 1);
    expect(V[idx(0, 0)]).toBeCloseTo(0.729, 3);
  });

  test('m=Infinity converges like full PI', () => {
    const { iterations } = modifiedPolicyIteration(mdp, Infinity);
    expect(iterations).toBe(3);
  });

  test('m=5 intermediate: V matches V* approximately', () => {
    const { V } = modifiedPolicyIteration(mdp, 5);
    const exact = valueIteration(mdp, 1e-12).V;
    for (let s = 0; s < 9; s++) expect(Math.abs(V[s] - exact[s])).toBeLessThan(0.01);
  });
});

describe('asyncValueIteration', () => {
  test('reverse sweep converges faster than forward', () => {
    const fwd = asyncValueIteration(mdp, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const rev = asyncValueIteration(mdp, [8, 7, 6, 5, 4, 3, 2, 1, 0]);
    expect(rev.iterations).toBeLessThan(fwd.iterations);
    expect(rev.iterations).toBeLessThanOrEqual(3);
  });

  test('reverse sweep V matches V* to 5 decimals', () => {
    const { V } = asyncValueIteration(mdp, [8, 7, 6, 5, 4, 3, 2, 1, 0], 1e-10);
    const optimal = valueIteration(mdp, 1e-12).V;
    for (let s = 0; s < 9; s++) expect(V[s]).toBeCloseTo(optimal[s], 5);
  });

  test('forward sweep V matches V* to 5 decimals', () => {
    const { V } = asyncValueIteration(mdp, [0, 1, 2, 3, 4, 5, 6, 7, 8], 1e-10);
    const optimal = valueIteration(mdp, 1e-12).V;
    for (let s = 0; s < 9; s++) expect(V[s]).toBeCloseTo(optimal[s], 5);
  });
});

describe('supDist', () => {
  test('zero for identical vectors', () => {
    expect(supDist([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  test('max absolute difference', () => {
    expect(supDist([0, 1, 3], [0, 0, 0])).toBe(3);
  });
});
