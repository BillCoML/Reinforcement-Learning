/**
 * V7 — Optimality Explorer. The gridworld shaded by V^*(s) with the greedy
 * policy arrows auto-drawn, the Bellman optimality equation filled in for the
 * hovered state, and a "compare to V^π" mode that shows the regret heatmap
 * V^* − V^π. Cells with more than one optimal action carry a count badge —
 * exposing policy non-uniqueness ((0,0) has two: Right and Down, both 0.729).
 */
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { buildGridworld } from "../mdp/gridworld";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { optimalValue, greedyPolicy, greedyActions, qOfAction } from "../mdp/value-iteration";
import { resolvePolicy, savedPolicy } from "./mdp-shared";
import { makeValueColorScale, fmtV } from "./value-scale";
import { ACTION_NAMES, idx, rc, type MDP } from "../mdp/types";

export class OptimalityExplorer extends HTMLElement {
  private mdp: MDP = buildGridworld({ slippery: false, gamma: 0.9 });
  private Vstar: number[] = [];
  private gr!: GridworldRenderer;
  private diffMode = false;
  private comparePolicy: "uniform" | "saved" = "uniform";
  private eqn!: HTMLElement;
  private summary!: HTMLElement;

  connectedCallback(): void {
    this.innerHTML = "";
    this.Vstar = optimalValue(this.mdp, 300);
    const { panel, body } = createPanel({ id: "optimality-explorer" });

    const controls = document.createElement("div");
    controls.className = "mdp-controls";

    // view toggle
    const seg = document.createElement("div");
    seg.className = "mdp-seg";
    const mk = (text: string, diff: boolean) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.setAttribute("aria-label", text);
      if (this.diffMode === diff) b.classList.add("is-active");
      b.addEventListener("click", () => {
        this.diffMode = diff;
        seg.querySelectorAll("button").forEach((x) => x.classList.remove("is-active"));
        b.classList.add("is-active");
        cmpLabel.style.display = diff ? "" : "none";
        this.draw();
      });
      return b;
    };
    seg.append(mk("V*", false), mk("V* − V^π (regret)", true));
    const segLabel = document.createElement("label");
    segLabel.append("view ");
    segLabel.appendChild(seg);
    controls.appendChild(segLabel);

    // comparison policy (only in diff mode)
    const cmpLabel = document.createElement("label");
    cmpLabel.style.display = "none";
    cmpLabel.append("π ");
    const cmpSel = document.createElement("select");
    cmpSel.setAttribute("aria-label", "comparison policy");
    for (const [v, t] of [["uniform", "uniform random"], ["saved", "custom (from V2)"]]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t;
      cmpSel.appendChild(o);
    }
    cmpSel.addEventListener("change", () => {
      this.comparePolicy = cmpSel.value as "uniform" | "saved";
      this.draw();
    });
    cmpLabel.appendChild(cmpSel);
    controls.appendChild(cmpLabel);
    body.appendChild(controls);

    const row = document.createElement("div");
    row.className = "mdp-row";
    const gridCol = document.createElement("div");
    const sideCol = document.createElement("div");
    sideCol.className = "mdp-col";
    row.append(gridCol, sideCol);
    body.appendChild(row);

    this.gr = new GridworldRenderer(gridCol, {
      mdp: this.mdp,
      valueFn: this.Vstar,
      policy: greedyPolicy(this.mdp, this.Vstar),
      showArrows: true,
      badges: this.badges(),
      onCellHover: (s) => this.showEqn(s),
    });

    this.eqn = document.createElement("div");
    this.eqn.className = "mdp-readout";
    this.summary = document.createElement("div");
    this.summary.className = "mdp-readout";
    sideCol.append(this.eqn, this.summary);

    savedPolicy.subscribe(() => {
      if (this.diffMode && this.comparePolicy === "saved") this.draw();
    });

    this.appendChild(panel);
    this.draw();
    this.showEqn(idx(0, 0));
  }

  private optCounts(): number[] {
    return this.mdp.terminals.map((term, s) => (term ? 0 : greedyActions(this.mdp, this.Vstar, s).length));
  }

  private badges(): (string | null)[] {
    return this.optCounts().map((n) => (n >= 2 ? String(n) : null));
  }

  private draw(): void {
    if (this.diffMode) {
      const Vpi = policyEvaluationExact(this.mdp, resolvePolicy(this.mdp, this.comparePolicy));
      const diff = this.Vstar.map((v, s) => v - Vpi[s]);
      // regret is ≥ 0; symmetric domain so 0 → neutral and the largest gap → deep green
      const mx = Math.max(...diff, 1e-6);
      const scale = makeValueColorScale([-mx, mx]);
      this.gr.update({ valueFn: diff, colorScale: scale, policy: undefined, showArrows: false, badges: [] }, { animate: true });
      const total = diff.reduce((a, b) => a + b, 0);
      this.summary.innerHTML =
        `Regret heatmap <span class="mdp-stat">V*(s) − V^π(s) ≥ 0</span> for ${this.comparePolicy === "saved" ? "your saved policy" : "uniform random"}. ` +
        `Greener = more value left on the table. Σ regret = <span class="mdp-stat">${total.toFixed(3)}</span>.`;
    } else {
      this.gr.update(
        { valueFn: this.Vstar, colorScale: makeValueColorScale(), policy: greedyPolicy(this.mdp, this.Vstar), showArrows: true, badges: this.badges() },
        { animate: true },
      );
      const multi = this.optCounts().filter((n) => n >= 2).length;
      this.summary.innerHTML =
        `Greedy arrows are optimal; the amber badge counts <span class="mdp-stat">|argmax|</span> where it exceeds 1. ` +
        `${multi} cell${multi === 1 ? "" : "s"} ${multi === 1 ? "has" : "have"} multiple optimal actions — e.g. (0,0): Right and Down, both reaching V* = 0.729.`;
    }
  }

  private showEqn(s: number | null): void {
    if (s == null) return;
    const { r, c } = rc(s);
    if (this.mdp.terminals[s]) {
      this.eqn.innerHTML = `<p style="margin:0">(${r},${c}) is terminal: <span class="mdp-stat">V*(${r},${c}) = 0</span>.</p>`;
      return;
    }
    const q = Array.from({ length: this.mdp.nA }, (_, a) => qOfAction(this.mdp, this.Vstar, s, a));
    const best = Math.max(...q);
    let html = `<p style="margin:0 0 4px">$V^*(s) = \\max_a\\,[\\,r(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^*(s')\\,]$</p>`;
    html += `<p style="margin:0 0 2px">at <span class="mdp-stat">(${r},${c})</span>:</p><div class="mdp-backup-row">`;
    for (let a = 0; a < this.mdp.nA; a++) {
      const isBest = Math.abs(q[a] - best) < 1e-9;
      const style = isBest ? ' style="color:var(--mdp-backup-source);font-weight:600"' : "";
      html += `<span${style}>${ACTION_NAMES[a]}</span><span${style}>${q[a].toFixed(3)}${isBest ? " ◀ max" : ""}</span>`;
    }
    html += `</div><p style="margin:4px 0 0"><span class="mdp-stat">V*(${r},${c}) = ${fmtV(best)}</span></p>`;
    this.eqn.innerHTML = html;
    // KaTeX render the equation line if available
    void import("../lesson/render-math").then((m) => m.typesetMath(this.eqn)).catch(() => {});
  }
}

customElements.define("optimality-explorer", OptimalityExplorer);
