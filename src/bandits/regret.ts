/**
 * Pseudo-regret tracking. On each step we add the gap Δ_arm of the chosen arm
 * (the expected per-step shortfall vs. the oracle). The optimal arm has Δ = 0
 * and contributes nothing — exactly the gap decomposition from §2:
 *   R_T = Σ_i Δ_i · E[N_i(T)].
 */

import type { BanditEnvironment } from "./env";

export class RegretTracker {
  private regret = 0;
  private history: number[] = [];

  /** Record a pull of `arm`: pseudo-regret grows by Δ_arm. */
  observe(env: BanditEnvironment, arm: number): void {
    this.regret += env.gaps()[arm];
    this.history.push(this.regret);
  }

  current(): number {
    return this.regret;
  }

  curve(): number[] {
    return [...this.history];
  }

  reset(): void {
    this.regret = 0;
    this.history = [];
  }
}
