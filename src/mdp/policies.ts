/**
 * Named policies on the gridworld, shared by V2–V7 so the dropdowns agree.
 * The "optimal" policy breaks ties deterministically (prefer Right, then Down)
 * so its arrows match the spec's §7 figure; ties themselves are surfaced
 * separately via greedyActions.
 */
import { deterministicPolicy, uniformPolicy } from "./gridworld";
import { greedyActions, optimalValue } from "./value-iteration";
import { RIGHT, DOWN, UP, LEFT, rc, type MDP, type Policy } from "./types";

const TIE_PRIORITY = [RIGHT, DOWN, UP, LEFT];

/** Per-state action of a deterministic optimal policy (spec tie-break). */
export function optimalActions(mdp: MDP): number[] {
  const Vstar = optimalValue(mdp, 200);
  return Array.from({ length: mdp.nS }, (_, s) => {
    if (mdp.terminals[s]) return RIGHT;
    const tied = greedyActions(mdp, Vstar, s);
    return TIE_PRIORITY.find((a) => tied.includes(a)) ?? tied[0];
  });
}

export function optimalPolicy(mdp: MDP): Policy {
  return deterministicPolicy(mdp, optimalActions(mdp));
}

/**
 * A hand-coded policy that reaches the goal but takes a long way around:
 * head Down until the bottom row, then Right. Used as a V4 contrast policy.
 */
export function goDownThenRightPolicy(mdp: MDP): Policy {
  const actions = Array.from({ length: mdp.nS }, (_, s) => {
    const { r } = rc(s);
    return r < 2 ? DOWN : RIGHT;
  });
  return deterministicPolicy(mdp, actions);
}

/** ε-soft version of the optimal policy: best action 1−ε+ε/m, others ε/m. */
export function epsilonSoftOptimal(mdp: MDP, eps: number): Policy {
  const best = optimalActions(mdp);
  const m = mdp.nA;
  const pi = Array.from({ length: mdp.nS }, (_, s) => {
    if (mdp.terminals[s]) return new Array<number>(m).fill(1 / m);
    const row = new Array<number>(m).fill(eps / m);
    row[best[s]] += 1 - eps;
    return row;
  });
  return { pi };
}

export { uniformPolicy };
