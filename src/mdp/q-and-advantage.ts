/**
 * Action-value Q^π and advantage A^π, both derived from a state-value V^π via
 * the one-step Bellman relation Q^π(s,a) = r(s,a) + γ Σ_s' P(s'|s,a) V^π(s').
 */
import { qOfAction } from "./value-iteration";
import type { MDP } from "./types";

/** Q^π(s,a) for all (s,a) from V^π. Terminal states get all-zero rows. */
export function qFromV(mdp: MDP, V: number[]): number[][] {
  return Array.from({ length: mdp.nS }, (_, s) =>
    Array.from({ length: mdp.nA }, (_, a) =>
      mdp.terminals[s] ? 0 : qOfAction(mdp, V, s, a),
    ),
  );
}

/** Advantage A^π(s,a) = Q^π(s,a) − V^π(s). */
export function advantage(Q: number[][], V: number[]): number[][] {
  return Q.map((row, s) => row.map((q) => q - V[s]));
}
