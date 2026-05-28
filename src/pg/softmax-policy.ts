import type { Policy } from "../mdp/types";

/**
 * Softmax (Boltzmann) policy: π_θ(a|s) = exp(θ_{s,a}) / Σ_{a'} exp(θ_{s,a'}).
 * Parameters stored flat: theta[s * nA + a].
 * Score: ∂/∂θ_{s,a'} log π(a|s) = 1[a==a'] − π(a'|s).
 */
export class SoftmaxPolicy {
  readonly theta: Float64Array;

  constructor(
    public readonly nS: number,
    public readonly nA: number,
  ) {
    this.theta = new Float64Array(nS * nA);
  }

  /** Numerically stable softmax probabilities for state s. */
  probs(s: number): Float64Array {
    const base = s * this.nA;
    const p = new Float64Array(this.nA);
    let maxVal = -Infinity;
    for (let a = 0; a < this.nA; a++) {
      const v = this.theta[base + a];
      if (v > maxVal) maxVal = v;
    }
    let sum = 0;
    for (let a = 0; a < this.nA; a++) {
      p[a] = Math.exp(this.theta[base + a] - maxVal);
      sum += p[a];
    }
    for (let a = 0; a < this.nA; a++) p[a] /= sum;
    return p;
  }

  /** Sample action in state s using the provided RNG. */
  sample(s: number, rng: () => number = Math.random): number {
    const p = this.probs(s);
    const u = rng();
    let cum = 0;
    for (let a = 0; a < this.nA; a++) {
      cum += p[a];
      if (u < cum) return a;
    }
    return this.nA - 1;
  }

  /**
   * Score function for the taken action a at state s.
   * Returns Float64Array of length nA: score[a'] = 1[a==a'] − π(a'|s).
   * Sums to exactly zero (since Σ π(a'|s) = 1).
   */
  scoreFunction(s: number, a: number): Float64Array {
    const p = this.probs(s);
    const score = new Float64Array(this.nA);
    for (let ap = 0; ap < this.nA; ap++) {
      score[ap] = (ap === a ? 1 : 0) - p[ap];
    }
    return score;
  }

  /**
   * Materialize full pi matrix for policyEvaluationExact and rollout().
   * Only call at evaluation time — allocates a 2D array.
   */
  toPolicy(): Policy {
    const pi: number[][] = [];
    for (let s = 0; s < this.nS; s++) {
      pi.push(Array.from(this.probs(s)));
    }
    return { pi };
  }
}
