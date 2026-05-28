import { sampleNormal } from "../importance-sampling/gaussian";

/**
 * Score function estimator for E_{X~N(theta,1)}[X²].
 *
 * True gradient: ∇_theta E[X²] = 2·theta.
 * Score fn:      ∇_theta log N(x; theta, 1) = x − theta.
 * Estimator:     (1/N) Σ_i X_i² (X_i − theta), X_i ~ N(theta, 1).
 */
export function gaussianScoreEstimator(
  theta: number,
  n: number,
  rng: () => number,
): number {
  const samples = sampleNormal(n, theta, 1, rng);
  let total = 0;
  for (const x of samples) total += x * x * (x - theta);
  return total / n;
}

/** True gradient of E_{X~N(theta,1)}[X²] = 2·theta. */
export function gaussianTrueGradient(theta: number): number {
  return 2 * theta;
}

/**
 * Draw N samples from N(theta, 1) for the interactive V2 demo.
 * Returns both the samples and the score estimate.
 */
export function gaussianScoreDemo(
  theta: number,
  n: number,
  rng: () => number,
): { samples: number[]; estimate: number; trueGrad: number } {
  const samples = sampleNormal(n, theta, 1, rng);
  let total = 0;
  for (const x of samples) total += x * x * (x - theta);
  return {
    samples,
    estimate: total / n,
    trueGrad: 2 * theta,
  };
}
