import type { Policy } from '../mdp/types';

/** Sup-norm distance between two value functions. */
export function supDist(V: number[], Vp: number[]): number {
  let m = 0;
  for (let i = 0; i < V.length; i++) m = Math.max(m, Math.abs(V[i] - Vp[i]));
  return m;
}

/** Deep-clone a policy. */
export function clonePolicy(policy: Policy): Policy {
  return { pi: policy.pi.map(row => row.slice()) };
}

/** Element-wise equality check for two policies. */
export function policiesEqual(a: Policy, b: Policy): boolean {
  for (let s = 0; s < a.pi.length; s++) {
    for (let act = 0; act < a.pi[s].length; act++) {
      if (Math.abs(a.pi[s][act] - b.pi[s][act]) > 1e-12) return false;
    }
  }
  return true;
}
