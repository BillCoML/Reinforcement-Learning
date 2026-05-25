/**
 * Newton's method for sqrt(2) as rational fractions, used in the
 * Counterexample Gallery (V4 tab 2) to demonstrate non-completeness of Q.
 *
 * Each iterate is stored as a [numerator, denominator] bigint pair so the
 * fractions render exactly, making the "limit is irrational" point land hard.
 */

export interface Fraction {
  n: bigint;
  d: bigint;
}

/** One Newton step: x_{k+1} = (x + 2/x) / 2 in exact fractions. */
export function newtonStep(x: Fraction): Fraction {
  // x_{k+1} = (x + 2/x) / 2 = (x^2 + 2) / (2x)
  const n = x.n * x.n + 2n * x.d * x.d;
  const d = 2n * x.n * x.d;
  return simplify({ n, d });
}

function gcd(a: bigint, b: bigint): bigint {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) { const t = b; b = a % b; a = t; }
  return a;
}

export function simplify(f: Fraction): Fraction {
  const g = gcd(f.n < 0n ? -f.n : f.n, f.d < 0n ? -f.d : f.d);
  return { n: f.n / g, d: f.d / g };
}

/** Generate the first n iterates starting from x_0 = 1. */
export function newtonIterates(n: number): Fraction[] {
  const result: Fraction[] = [{ n: 1n, d: 1n }];
  for (let k = 1; k < n; k++) result.push(newtonStep(result[k - 1]));
  return result;
}

export function fractionToString(f: Fraction): string {
  return f.d === 1n ? `${f.n}` : `${f.n}/${f.d}`;
}

export function fractionToDecimal(f: Fraction): number {
  return Number(f.n) / Number(f.d);
}
