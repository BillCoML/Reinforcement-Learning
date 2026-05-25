/**
 * Cross-component MDP state. V2 (Policy Explorer) writes the user's custom
 * policy here via "Save policy"; V4/V5/V6 read it for their "custom (from V2)"
 * option. A single shared Store keeps them in sync without prop-drilling.
 */
import { Store } from "./store";

export interface SavedPolicyState {
  /** pi[s][a] of the last policy the user saved in V2, or null if none yet. */
  pi: number[][] | null;
  /** Human label for the saved policy (shown in dropdowns). */
  label: string;
}

export const savedPolicy = new Store<SavedPolicyState>({ pi: null, label: "custom (from V2)" });
