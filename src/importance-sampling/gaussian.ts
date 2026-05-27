/**
 * Gaussian utilities: Box-Muller sampler and normal PDF.
 * mulberry32 seeded PRNG replaces Math.random for deterministic tests.
 */

/** mulberry32 — high-quality 32-bit seeded PRNG. Returns a factory. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function (): number {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller transform: two N(0,1) samples from two independent uniforms.
 * Returns a pair — consume both to avoid correlation.
 */
export function boxMuller(rng: () => number): [number, number] {
  const u1 = rng();
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1 === 0 ? 1e-10 : u1));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

/** Normal PDF: N(x; mu, sigma). */
export function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/** Draw N samples from N(mu, sigma) using the provided RNG. */
export function sampleNormal(
  n: number,
  mu: number,
  sigma: number,
  rng: () => number,
): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i += 2) {
    const [a, b] = boxMuller(rng);
    out[i] = mu + sigma * a;
    if (i + 1 < n) out[i + 1] = mu + sigma * b;
  }
  return out;
}
