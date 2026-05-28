/**
 * Numerically stable log-sum-exp at temperature alpha.
 * Computes: alpha * log( sum_i exp(values[i] / alpha) )
 * Uses shift by max to keep all exp() arguments in (-inf, 0].
 */
export function logSumExp(values: Float64Array, alpha: number): number {
  let maxVal = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i] > maxVal) maxVal = values[i];
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += Math.exp((values[i] - maxVal) / alpha);
  return alpha * Math.log(sum) + maxVal;
}

/**
 * Boltzmann probabilities: softmax(values / alpha).
 * Numerically stable via the same max shift.
 */
export function boltzmannProbs(values: Float64Array, alpha: number): Float64Array {
  let maxVal = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i] > maxVal) maxVal = values[i];
  const p = new Float64Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    p[i] = Math.exp((values[i] - maxVal) / alpha);
    sum += p[i];
  }
  for (let i = 0; i < values.length; i++) p[i] /= sum;
  return p;
}

/** Shannon entropy in nats: -sum p log p. Safe for p=0 (returns 0 for that term). */
export function shannonEntropy(probs: Float64Array): number {
  let H = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > 0) H -= probs[i] * Math.log(probs[i]);
  return H;
}
