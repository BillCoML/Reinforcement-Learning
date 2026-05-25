/**
 * Cross-component MDP state. V2 (Policy Explorer) writes the user's custom
 * policy here via "Save policy"; V4/V5/V6 read it for their "custom (from V2)"
 * option. A single shared Store keeps them in sync without prop-drilling.
 */
import { Store } from "./store";
import { uniformPolicy } from "../mdp/gridworld";
import { optimalPolicy, goDownThenRightPolicy } from "../mdp/policies";
import type { MDP, Policy } from "../mdp/types";

export interface SavedPolicyState {
  /** pi[s][a] of the last policy the user saved in V2, or null if none yet. */
  pi: number[][] | null;
  /** Human label for the saved policy (shown in dropdowns). */
  label: string;
}

export const savedPolicy = new Store<SavedPolicyState>({ pi: null, label: "custom (from V2)" });

export type PolicyKey = "uniform" | "optimal" | "downRight" | "saved";

/** The shared policy-dropdown options used by V4, V5, V7. */
export const POLICY_OPTIONS: [PolicyKey, string][] = [
  ["uniform", "uniform random"],
  ["optimal", "deterministic optimal"],
  ["downRight", "go-down-then-right"],
  ["saved", "custom (from V2)"],
];

/** Resolve a dropdown key to a concrete Policy on the given MDP. */
export function resolvePolicy(mdp: MDP, key: PolicyKey): Policy {
  switch (key) {
    case "uniform":
      return uniformPolicy(mdp);
    case "optimal":
      return optimalPolicy(mdp);
    case "downRight":
      return goDownThenRightPolicy(mdp);
    case "saved": {
      const s = savedPolicy.get();
      return s.pi ? { pi: s.pi } : uniformPolicy(mdp);
    }
  }
}

/** Build a <select> wired to the shared options; calls onChange with the key. */
export function buildPolicySelect(onChange: (k: PolicyKey) => void): HTMLSelectElement {
  const sel = document.createElement("select");
  sel.setAttribute("aria-label", "policy");
  for (const [v, t] of POLICY_OPTIONS) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => onChange(sel.value as PolicyKey));
  return sel;
}
