/**
 * Importance sampling estimators for the IS lesson.
 * ~80 lines. All functions are pure; RNG is injected.
 */

/** Importance weight w(x) = p(x) / q(x). Returns Infinity if q(x)=0 (coverage violated). */
export function importanceWeight(
  pPdf: (x: number) => number,
  qPdf: (x: number) => number,
  x: number,
): number {
  const q = qPdf(x);
  if (q === 0) return Infinity;
  return pPdf(x) / q;
}

/** Ordinary IS estimator: (1/N) Σ w(x_i) f(x_i). */
export function ordinaryIS(
  samples: number[],
  f: (x: number) => number,
  w: (x: number) => number,
): number {
  if (samples.length === 0) return 0;
  return samples.reduce((s, x) => s + w(x) * f(x), 0) / samples.length;
}

/** Weighted (self-normalized) IS estimator: Σ w f / Σ w. */
export function weightedIS(
  samples: number[],
  f: (x: number) => number,
  w: (x: number) => number,
): number {
  let num = 0;
  let den = 0;
  for (const x of samples) {
    const wi = w(x);
    num += wi * f(x);
    den += wi;
  }
  return den === 0 ? 0 : num / den;
}

/** Effective sample size N_eff = (Σw)² / Σw². Returns 0 when all weights are zero. */
export function effectiveSampleSize(weights: number[]): number {
  let s = 0;
  let s2 = 0;
  for (const wi of weights) {
    s += wi;
    s2 += wi * wi;
  }
  return s2 === 0 ? 0 : (s * s) / s2;
}

/**
 * Trajectory-level IS weight ρ_{0:T-1} = ∏ π_t(a_t|s_t) / π_b(a_t|s_t).
 * Returns 0 if any per-step target probability is zero (trajectory rejected).
 * Returns 0 if pb=0 (defensive; should not happen when traj was sampled under piB).
 */
export function trajectoryISWeight(
  traj: { s: number; a: number }[],
  piTarget: (s: number, a: number) => number,
  piBehavior: (s: number, a: number) => number,
): number {
  let rho = 1.0;
  for (const { s, a } of traj) {
    const pb = piBehavior(s, a);
    if (pb === 0) return 0;
    rho *= piTarget(s, a) / pb;
    if (rho === 0) return 0;
  }
  return rho;
}

/**
 * Per-decision IS contribution for one trajectory.
 * Each reward γ^t * r_t is weighted by ρ_{0:t} (ratios 0 through t inclusive),
 * eliminating only the FUTURE ratios ρ_{t+1:T-1} that add variance without
 * signal. For sparse terminal rewards this equals trajectory IS exactly.
 */
export function perDecisionIS(
  traj: { s: number; a: number; r: number }[],
  piTarget: (s: number, a: number) => number,
  piBehavior: (s: number, a: number) => number,
  gamma: number,
): number {
  let total = 0;
  let rho = 1.0;
  for (let t = 0; t < traj.length; t++) {
    const { s, a, r } = traj[t];
    const pb = piBehavior(s, a);
    if (pb === 0) return total;
    rho *= piTarget(s, a) / pb;
    if (rho === 0) return total;
    total += Math.pow(gamma, t) * r * rho;
  }
  return total;
}
