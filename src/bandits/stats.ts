/**
 * Statistical primitives for the bandit lesson.
 *
 * Everything here is pure and RNG-injectable so the same code runs
 * deterministically in tests and reproducibly in the Battle Arena.
 */

export type RNG = () => number;

/**
 * Mulberry32 — a tiny, fast, decent-quality seeded PRNG.
 * Used so URL-shareable seeds (`?seed=42`) reproduce exactly in-browser.
 */
export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Index of the maximum value. Ties broken toward the lowest index. */
export function argmax(xs: ArrayLike<number>): number {
  let best = 0;
  let bestVal = xs[0];
  for (let i = 1; i < xs.length; i++) {
    if (xs[i] > bestVal) {
      bestVal = xs[i];
      best = i;
    }
  }
  return best;
}

/**
 * Standard normal sample via Box–Muller. Consumes two uniforms.
 */
export function sampleNormal(rng: RNG = Math.random): number {
  let u1 = rng();
  // guard against log(0)
  if (u1 < 1e-12) u1 = 1e-12;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Gamma(shape, scale=1) sample via Marsaglia–Tsang (2000).
 * Correct for fractional shape, unlike naive methods.
 */
export function sampleGamma(shape: number, rng: RNG = Math.random): number {
  if (shape < 1) {
    // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
    const u = rng();
    return sampleGamma(shape + 1, rng) * Math.pow(u < 1e-12 ? 1e-12 : u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  // eslint-disable-next-line no-constant-condition
  while (true) {
    let x: number;
    let v: number;
    do {
      x = sampleNormal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    const x2 = x * x;
    if (u < 1 - 0.0331 * x2 * x2) return d * v;
    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * Beta(α, β) sample via two Gamma draws: Beta = Ga / (Ga + Gb).
 * The correct general approach for fractional parameters.
 */
export function sampleBeta(alpha: number, beta: number, rng: RNG = Math.random): number {
  const ga = sampleGamma(alpha, rng);
  const gb = sampleGamma(beta, rng);
  return ga / (ga + gb);
}

/**
 * Hoeffding two-sided bound: Pr(|μ̂_n − μ| > ε) ≤ 2·exp(−2 n ε²),
 * for rewards in [0, 1].
 */
export function hoeffdingBound(n: number, epsilon: number): number {
  return 2 * Math.exp(-2 * n * epsilon * epsilon);
}

/**
 * KL divergence between two Bernoulli distributions Bern(p) ‖ Bern(q).
 * KL = p·ln(p/q) + (1−p)·ln((1−p)/(1−q)), with the usual 0·ln0 = 0 limits.
 */
export function bernoulliKL(p: number, q: number): number {
  const eps = 1e-12;
  const pc = Math.min(Math.max(p, eps), 1 - eps);
  const qc = Math.min(Math.max(q, eps), 1 - eps);
  return pc * Math.log(pc / qc) + (1 - pc) * Math.log((1 - pc) / (1 - qc));
}

/**
 * Lai–Robbins asymptotic constant for a Bernoulli instance:
 *   C = Σ_{i: Δ_i > 0} Δ_i / KL(ν_i ‖ ν*)
 * The lower bound says liminf R_T / log T ≥ C.
 */
export function laiRobbinsConstant(means: number[]): number {
  const muStar = Math.max(...means);
  let c = 0;
  for (const mu of means) {
    const delta = muStar - mu;
    if (delta > 0) {
      c += delta / bernoulliKL(mu, muStar);
    }
  }
  return c;
}

/** Natural log of the Gamma function (Lanczos approximation). */
export function logGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Beta(α, β) probability density at x ∈ (0, 1). Used to draw the
 * posterior curves in V6. Returns 0 outside the open interval.
 */
export function betaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0;
  const logB = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  const logPdf = (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logB;
  return Math.exp(logPdf);
}

/** Mean and standard deviation of Beta(α, β). */
export function betaMoments(alpha: number, beta: number): { mean: number; std: number } {
  const s = alpha + beta;
  const mean = alpha / s;
  const variance = (alpha * beta) / (s * s * (s + 1));
  return { mean, std: Math.sqrt(variance) };
}
