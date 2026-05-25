/**
 * MarkovChain — the math substrate for the Markov Chains lesson.
 *
 * Finite, time-homogeneous, discrete-time chains on K states. Everything is
 * pure and (where randomness is involved) RNG-injectable, so the same code
 * runs deterministically in Vitest and reproducibly in the browser.
 *
 * Linear algebra (eigendecomposition) goes through ml-matrix; classification
 * (communicating classes, period, recurrence) is graph-theoretic on the
 * support graph of P.
 */
import { Matrix, EigenvalueDecomposition } from "ml-matrix";

/** A complex number, used for the (possibly complex) spectrum of P. */
export interface Complex {
  re: number;
  im: number;
}

export function complexAbs(c: Complex): number {
  return Math.hypot(c.re, c.im);
}

/** Treat |x| ≤ EPS as zero for support-graph / balance checks. */
const EPS = 1e-12;

export type RNG = () => number;

/** Per-state structural classification, as surfaced by V3's inspector. */
export interface StateClass {
  /** Indices of the states in this communicating class. */
  members: number[];
  /** Recurrent (closed class) vs transient (leaks out). */
  recurrent: boolean;
  /** Common period of the class (1 = aperiodic). */
  period: number;
}

export class MarkovChain {
  readonly P: Matrix; // K x K row-stochastic
  readonly K: number;
  private readonly rows: number[][];

  constructor(P: number[][]) {
    this.rows = P.map((r) => r.slice());
    this.P = new Matrix(this.rows);
    this.K = P.length;
    this.validateRowStochastic();
  }

  private validateRowStochastic(): void {
    for (let i = 0; i < this.K; i++) {
      const row = this.rows[i];
      if (row.length !== this.K) {
        throw new Error(`Row ${i} has length ${row.length}, expected ${this.K} (P must be square).`);
      }
      let sum = 0;
      for (const v of row) {
        if (v < -1e-9) throw new Error(`Negative transition probability at row ${i}: ${v}.`);
        sum += v;
      }
      if (Math.abs(sum - 1) > 1e-6) {
        throw new Error(`Row ${i} sums to ${sum.toFixed(6)}, expected 1 (P must be row-stochastic).`);
      }
    }
  }

  /** P raised to the n-th power (n ≥ 0). P^0 = I. */
  pPower(n: number): Matrix {
    if (n < 0) throw new Error("pPower requires n ≥ 0.");
    let result = Matrix.eye(this.K);
    // Exponentiation by squaring keeps large powers cheap and stable.
    let base = this.P.clone();
    let e = n;
    while (e > 0) {
      if (e & 1) result = result.mmul(base);
      e >>= 1;
      if (e > 0) base = base.mmul(base);
    }
    return result;
  }

  /** Distribution at time n from initial row distribution μ₀: μ_n = μ₀ Pⁿ. */
  distributionAfter(mu0: number[], n: number): number[] {
    const Pn = this.pPower(n);
    const out = new Array<number>(this.K).fill(0);
    for (let j = 0; j < this.K; j++) {
      let s = 0;
      for (let i = 0; i < this.K; i++) s += mu0[i] * Pn.get(i, j);
      out[j] = s;
    }
    return out;
  }

  /**
   * Stationary distribution via the left eigenvector of P for eigenvalue 1,
   * i.e. the eigenvector of Pᵀ for eigenvalue 1, normalized to the simplex.
   * For reducible chains this returns one valid stationary distribution
   * (the one the dominant eigenvector picks out).
   */
  stationary(): number[] {
    const eig = new EigenvalueDecomposition(this.P.transpose());
    const re = eig.realEigenvalues;
    const im = eig.imaginaryEigenvalues;
    // Pick the eigenvalue closest to 1 + 0i (it is exactly 1 for stochastic P).
    let idx = 0;
    let best = Infinity;
    for (let k = 0; k < re.length; k++) {
      const d = Math.hypot(re[k] - 1, im[k]);
      if (d < best) {
        best = d;
        idx = k;
      }
    }
    const v = eig.eigenvectorMatrix.getColumn(idx);
    return normalizeToSimplex(v);
  }

  /** Full (possibly complex) spectrum of P. */
  eigenvalues(): Complex[] {
    const eig = new EigenvalueDecomposition(this.P);
    const re = eig.realEigenvalues;
    const im = eig.imaginaryEigenvalues;
    return re.map((r, k) => ({ re: r, im: im[k] }));
  }

  /** Second-largest eigenvalue magnitude |λ₂| — sets the convergence rate. */
  lambdaStar(): number {
    const mags = this.eigenvalues()
      .map(complexAbs)
      .sort((a, b) => b - a);
    return mags.length > 1 ? mags[1] : 0;
  }

  /** Spectral gap γ = 1 − |λ₂|. */
  spectralGap(): number {
    return 1 - this.lambdaStar();
  }

  /** Mixing-time bound t_mix(ε) ≤ log(1/ε) / (1 − λ⋆). */
  mixingTimeBound(epsilon: number): number {
    const gap = this.spectralGap();
    if (gap <= EPS) return Infinity;
    return Math.log(1 / epsilon) / gap;
  }

  /** Support-graph adjacency: i → j present iff P_{ij} > 0. */
  private adjacency(): number[][] {
    const adj: number[][] = [];
    for (let i = 0; i < this.K; i++) {
      const out: number[] = [];
      for (let j = 0; j < this.K; j++) if (this.rows[i][j] > EPS) out.push(j);
      adj.push(out);
    }
    return adj;
  }

  /**
   * Communicating classes = strongly-connected components of the support
   * graph, via Tarjan's algorithm. Each returned array is a class; the order
   * within and between classes is deterministic (ascending by min member).
   */
  communicatingClasses(): number[][] {
    const sccs = tarjanSCC(this.adjacency(), this.K);
    sccs.forEach((c) => c.sort((a, b) => a - b));
    sccs.sort((a, b) => a[0] - b[0]);
    return sccs;
  }

  /**
   * Period of state i: gcd of all return times n with (Pⁿ)_{ii} > 0. We scan
   * n = 1 … K² (a safe finite-chain horizon). Returns 0 if i never returns.
   */
  period(i: number): number {
    let g = 0;
    let Pn = this.P.clone();
    const horizon = this.K * this.K;
    for (let n = 1; n <= horizon; n++) {
      if (Pn.get(i, i) > EPS) g = gcd(g, n);
      if (g === 1) break; // can't get smaller
      if (n < horizon) Pn = Pn.mmul(this.P);
    }
    return g;
  }

  isIrreducible(): boolean {
    return this.communicatingClasses().length === 1;
  }

  /** Aperiodic iff the (class) period is 1. Assumes irreducibility for the shorthand. */
  isAperiodic(): boolean {
    return this.period(0) === 1;
  }

  /**
   * Structural classification of every state: which communicating class it
   * belongs to, whether that class is recurrent (closed) or transient, and the
   * class period. Drives V3's badges and node tinting.
   */
  classify(): { classes: StateClass[]; stateToClass: number[] } {
    const classes = this.communicatingClasses();
    const stateToClass = new Array<number>(this.K).fill(-1);
    classes.forEach((members, ci) => members.forEach((s) => (stateToClass[s] = ci)));

    const adj = this.adjacency();
    const info: StateClass[] = classes.map((members) => {
      const set = new Set(members);
      // A finite-chain class is recurrent iff it is closed (no edge leaves it).
      let closed = true;
      for (const s of members) {
        for (const t of adj[s]) {
          if (!set.has(t)) {
            closed = false;
            break;
          }
        }
        if (!closed) break;
      }
      return { members, recurrent: closed, period: this.period(members[0]) };
    });
    return { classes: info, stateToClass };
  }

  /** Total-variation distance ‖μ_n − π‖_TV from initial distribution μ₀. */
  tvToStationary(mu0: number[], n: number): number {
    const mu = this.distributionAfter(mu0, n);
    const pi = this.stationary();
    return totalVariation(mu, pi);
  }

  /**
   * Detailed-balance residual against a candidate π: max over pairs of
   * |π_i P_{ij} − π_j P_{ji}|. Zero (within tolerance) ⇔ reversible w.r.t. π.
   */
  detailedBalanceResidual(pi: number[]): number {
    let maxRes = 0;
    for (let i = 0; i < this.K; i++) {
      for (let j = i + 1; j < this.K; j++) {
        const res = Math.abs(pi[i] * this.rows[i][j] - pi[j] * this.rows[j][i]);
        if (res > maxRes) maxRes = res;
      }
    }
    return maxRes;
  }

  /** Sample one trajectory of length T (T states: X₀ … X_{T−1}) starting at s0. */
  sampleTrajectory(s0: number, T: number, rng: RNG = Math.random): number[] {
    const path = new Array<number>(T);
    let s = s0;
    for (let t = 0; t < T; t++) {
      path[t] = s;
      s = this.sampleNext(s, rng);
    }
    return path;
  }

  /** Sample the next state from row s. */
  sampleNext(s: number, rng: RNG = Math.random): number {
    const u = rng();
    let cum = 0;
    const row = this.rows[s];
    for (let j = 0; j < this.K; j++) {
      cum += row[j];
      if (u < cum) return j;
    }
    return this.K - 1; // numerical fallback
  }
}

// ---- free helpers ---------------------------------------------------------

export function totalVariation(p: number[], q: number[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += Math.abs(p[i] - q[i]);
  return 0.5 * s;
}

export function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    [a, b] = [b, a % b];
  }
  return a;
}

/**
 * Normalize a (real) eigenvector to a probability distribution: flip sign so
 * the mass is positive, clamp tiny negatives from numerical noise, sum to 1.
 */
export function normalizeToSimplex(v: number[]): number[] {
  let sum = 0;
  for (const x of v) sum += x;
  if (Math.abs(sum) < EPS) {
    // Degenerate; fall back to magnitudes.
    const mags = v.map(Math.abs);
    const m = mags.reduce((a, b) => a + b, 0) || 1;
    return mags.map((x) => x / m);
  }
  const signed = v.map((x) => x / sum);
  const clamped = signed.map((x) => (x < 0 ? 0 : x));
  const total = clamped.reduce((a, b) => a + b, 0) || 1;
  return clamped.map((x) => x / total);
}

/**
 * Tarjan's strongly-connected-components. Iterative-friendly recursive form;
 * K ≤ 8 here so recursion depth is a non-issue. Returns the SCCs as arrays of
 * node indices.
 */
export function tarjanSCC(adj: number[][], n: number): number[][] {
  let index = 0;
  const idx = new Array<number>(n).fill(-1);
  const low = new Array<number>(n).fill(0);
  const onStack = new Array<boolean>(n).fill(false);
  const stack: number[] = [];
  const out: number[][] = [];

  const strongConnect = (v: number): void => {
    idx[v] = index;
    low[v] = index;
    index++;
    stack.push(v);
    onStack[v] = true;

    for (const w of adj[v]) {
      if (idx[w] === -1) {
        strongConnect(w);
        low[v] = Math.min(low[v], low[w]);
      } else if (onStack[w]) {
        low[v] = Math.min(low[v], idx[w]);
      }
    }

    if (low[v] === idx[v]) {
      const comp: number[] = [];
      let w: number;
      do {
        w = stack.pop() as number;
        onStack[w] = false;
        comp.push(w);
      } while (w !== v);
      out.push(comp);
    }
  };

  for (let v = 0; v < n; v++) if (idx[v] === -1) strongConnect(v);
  return out;
}
