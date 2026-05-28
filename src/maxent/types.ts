export interface SoftValueIterationResult {
  readonly Q: Float64Array;     // nS * nA (flat, row-major)
  readonly V: Float64Array;     // nS — the soft Bellman fixed-point values
  readonly pi: Float64Array;    // nS * nA — Boltzmann policy
  readonly iterations: number;
  readonly converged: boolean;
}

export interface SoftEvaluationResult {
  readonly V_soft: Float64Array;  // J_alpha(pi) per state — includes entropy bonus
  readonly V_pi: Float64Array;    // E^pi[sum gamma^t r_t] — standard discounted return, no entropy
}

export interface MonteCarloDiagnostic {
  readonly goalReachProb: number;
  readonly pitReachProb: number;
  readonly timeoutProb: number;
  readonly meanStepsToTerminal: number;
  readonly lengthHistogram: Float64Array;  // 50 buckets, 0-500 steps
}
