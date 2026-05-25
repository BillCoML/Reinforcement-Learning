/** Core contraction-mapping utilities. All operate on plain number[] or number[][]. */

export function apply1D(a: number, b: number, x: number): number {
  return a * x + b;
}

export function iterate1D(T: (x: number) => number, x0: number, n: number): number[] {
  const traj = [x0];
  for (let k = 0; k < n; k++) traj.push(T(traj[k]));
  return traj;
}

export function apply2D(A: number[][], b: number[], x: number[]): number[] {
  return [
    A[0][0] * x[0] + A[0][1] * x[1] + b[0],
    A[1][0] * x[0] + A[1][1] * x[1] + b[1],
  ];
}

/** Maximum absolute row sum — the infinity operator norm. */
export function opNormInfinity(A: number[][]): number {
  return Math.max(...A.map(row => row.reduce((s, x) => s + Math.abs(x), 0)));
}

/** Fixed point of an affine 2D map T(x)=Ax+b via (I-A)^{-1}b. */
export function fixedPoint2D(A: number[][], b: number[]): number[] {
  const M = [
    [1 - A[0][0], -A[0][1]],
    [-A[1][0], 1 - A[1][1]],
  ];
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return [
    (M[1][1] * b[0] - M[0][1] * b[1]) / det,
    (-M[1][0] * b[0] + M[0][0] * b[1]) / det,
  ];
}

/** Sup-norm distance between two value functions. */
export function supDist(V: number[], Vp: number[]): number {
  let m = 0;
  for (let i = 0; i < V.length; i++) m = Math.max(m, Math.abs(V[i] - Vp[i]));
  return m;
}

/** Banach error bound: c^k / (1 - c) * d0, where d0 = d(x_1, x_0). */
export function banachBound(c: number, k: number, d0: number): number {
  return (Math.pow(c, k) / (1 - c)) * d0;
}

/** Spectral radius (max |eigenvalue|) of a 2×2 real matrix. */
export function spectralRadius2x2(A: number[][]): number {
  const tr = A[0][0] + A[1][1];
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const disc = tr * tr - 4 * det;
  if (disc >= 0) {
    const r1 = (tr + Math.sqrt(disc)) / 2;
    const r2 = (tr - Math.sqrt(disc)) / 2;
    return Math.max(Math.abs(r1), Math.abs(r2));
  }
  return Math.sqrt(Math.max(0, det));
}

/** Iterate a 2D affine map n steps from x0, returning the trajectory. */
export function iterate2D(A: number[][], b: number[], x0: number[], n: number): number[][] {
  const traj: number[][] = [x0];
  for (let k = 0; k < n; k++) traj.push(apply2D(A, b, traj[k]));
  return traj;
}
