/**
 * Panel F — Auto-caption. Snap-to-regime text that updates with alpha.
 * Four regimes: Tie-breaking, Useful, Trade-off, Failure.
 */

interface Regime {
  name: string;
  color: string;
  bg: string;
  description: string;
}

const REGIMES: Regime[] = [
  {
    name: "Tie-breaking",
    color: "var(--maxent-hard)",
    bg: "var(--maxent-hard-tint)",
    description:
      "The policy is essentially greedy with slight stochasticity at states where two actions are equally good. " +
      "VᵰB ≈ V* = 0.729. The entropy bonus is negligible.",
  },
  {
    name: "Useful",
    color: "var(--maxent-soft)",
    bg: "var(--maxent-soft-tint)",
    description:
      "The policy is goal-directed but explores. The agent takes slightly longer paths and samples " +
      "sub-optimal actions occasionally. This is where SAC would operate in continuous control. " +
      "VᵰB ≈ 0.59–0.72. The L10 softmax cap (0.722) sits at the lower end of this regime.",
  },
  {
    name: "Trade-off",
    color: "var(--maxent-alpha-high)",
    bg: "#fff7ed",
    description:
      "The agent takes longer paths to the goal. Entropy is paying a real cost in efficiency: " +
      "mean steps ≈ 79 (vs 4 for greedy). The policy still reaches the goal almost always, " +
      "but the discounted return VᵰB ≈ 0.07 is far below V*.",
  },
  {
    name: "Failure",
    color: "var(--maxent-failure)",
    bg: "var(--maxent-failure-tint)",
    description:
      "The agent has learned to avoid the goal. The entropy bonus from continued exploration " +
      "exceeds the +1 terminal reward. Wall-bumping actions are preferred at the start state. " +
      "Goal-reach probability collapses to ≤5%. The policy is objectively correct for the " +
      "entropy-regularized objective — but it fails the original task.",
  },
];

function getRegimeIdx(alpha: number): number {
  if (alpha <= 0.01) return 0;
  if (alpha <= 0.05) return 1;
  if (alpha <= 0.1) return 2;
  return 3;
}

export class PanelAutoCaption extends HTMLElement {
  private nameEl!: HTMLElement;
  private descEl!: HTMLElement;
  private containerEl!: HTMLElement;

  connectedCallback() {
    const container = document.createElement("div");
    container.style.cssText = `
      padding: 16px;
      border-radius: 8px;
      border: 2px solid;
      min-height: 120px;
      transition: background 0.3s, border-color 0.3s;
    `;
    this.containerEl = container;

    const name = document.createElement("div");
    name.style.cssText = "font-weight:700;font-size:1.1em;margin-bottom:8px;font-family:var(--rl-font-ui);transition:color 0.3s;";
    this.nameEl = name;

    const desc = document.createElement("p");
    desc.style.cssText = "margin:0;font-size:0.88em;line-height:1.6;color:var(--rl-ink);";
    this.descEl = desc;

    container.appendChild(name);
    container.appendChild(desc);
    this.appendChild(container);

    this.update(0.05); // initial
  }

  update(alpha: number) {
    const idx = getRegimeIdx(alpha);
    const regime = REGIMES[idx];
    this.containerEl.style.background = regime.bg;
    this.containerEl.style.borderColor = regime.color;
    this.nameEl.style.color = regime.color;
    this.nameEl.textContent = `Regime: ${regime.name}  (α = ${alpha.toFixed(4)})`;
    this.descEl.textContent = regime.description;
  }
}

customElements.define("esl-panel-auto-caption", PanelAutoCaption);
