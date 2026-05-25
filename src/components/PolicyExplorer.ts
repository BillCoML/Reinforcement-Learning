/**
 * V2 — Policy Explorer. The gridworld with policy arrows (opacity ∝ π(a|s)).
 * Switch between uniform random, deterministic optimal, ε-soft optimal (slider),
 * and a custom policy you build by bumping action weights on a selected cell.
 * "Save policy" stashes the current π in the shared store so V4–V6 can evaluate
 * it under "custom (from V2)".
 */
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { optimalPolicy, epsilonSoftOptimal } from "../mdp/policies";
import { savedPolicy } from "./mdp-shared";
import { cssVar } from "./base";
import { ACTION_NAMES, rc, type MDP, type Policy } from "../mdp/types";

type Mode = "uniform" | "optimal" | "epsilon" | "custom";
const ACTION_VARS = ["--mdp-action-up", "--mdp-action-right", "--mdp-action-down", "--mdp-action-left"];

export class PolicyExplorer extends HTMLElement {
  private mdp: MDP = buildGridworld({ slippery: false, gamma: 0.9 });
  private gr!: GridworldRenderer;
  private mode: Mode = "uniform";
  private eps = 0.3;
  private custom: number[][] = [];
  private sel: number | null = null;
  private bars!: HTMLElement;
  private epsLabel!: HTMLElement;
  private epsWrap!: HTMLElement;

  connectedCallback(): void {
    this.innerHTML = "";
    this.custom = uniformPolicy(this.mdp).pi.map((r) => [...r]);
    const { panel, body } = createPanel({ id: "policy-explorer" });

    // controls
    const controls = document.createElement("div");
    controls.className = "mdp-controls";
    const modeLabel = document.createElement("label");
    modeLabel.append("policy ");
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", "policy mode");
    for (const [v, t] of [
      ["uniform", "uniform random"],
      ["optimal", "deterministic optimal"],
      ["epsilon", "ε-soft optimal"],
      ["custom", "custom"],
    ] as [Mode, string][]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      this.mode = sel.value as Mode;
      this.epsWrap.style.display = this.mode === "epsilon" ? "" : "none";
      this.refresh();
    });
    modeLabel.appendChild(sel);
    controls.appendChild(modeLabel);

    // epsilon slider
    this.epsWrap = document.createElement("label");
    this.epsWrap.style.display = "none";
    this.epsWrap.append("ε ");
    const eslider = document.createElement("input");
    eslider.type = "range";
    eslider.min = "0";
    eslider.max = "1";
    eslider.step = "0.05";
    eslider.value = String(this.eps);
    eslider.setAttribute("aria-label", "epsilon (softness)");
    this.epsLabel = document.createElement("span");
    this.epsLabel.className = "mdp-stat";
    this.epsLabel.textContent = this.eps.toFixed(2);
    eslider.addEventListener("input", () => {
      this.eps = Number(eslider.value);
      this.epsLabel.textContent = this.eps.toFixed(2);
      this.refresh();
    });
    this.epsWrap.append(eslider, " ", this.epsLabel);
    controls.appendChild(this.epsWrap);

    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "Save policy → V4/V6";
    save.addEventListener("click", () => {
      const p = this.currentPolicy();
      savedPolicy.set({ pi: p.pi.map((r) => [...r]), label: `custom (${this.mode})` });
      save.textContent = "Saved ✓";
      setTimeout(() => (save.textContent = "Save policy → V4/V6"), 1200);
    });
    controls.appendChild(save);
    body.appendChild(controls);

    // grid + side panel
    const row = document.createElement("div");
    row.className = "mdp-row";
    const gridCol = document.createElement("div");
    const sideCol = document.createElement("div");
    sideCol.className = "mdp-col";
    row.append(gridCol, sideCol);
    body.appendChild(row);

    this.gr = new GridworldRenderer(gridCol, {
      mdp: this.mdp,
      policy: this.currentPolicy(),
      onCellClick: (s) => {
        if (!this.mdp.terminals[s]) {
          this.sel = s;
          this.renderBars();
        }
      },
    });

    this.bars = document.createElement("div");
    sideCol.appendChild(this.bars);

    // action legend
    const legend = document.createElement("div");
    legend.className = "mdp-legend";
    ACTION_NAMES.forEach((n, a) => {
      const s = document.createElement("span");
      s.innerHTML = `<span class="mdp-swatch" style="background:${cssVar(ACTION_VARS[a])}"></span>${n}`;
      legend.appendChild(s);
    });
    body.appendChild(legend);

    this.appendChild(panel);
    this.refresh();
  }

  private currentPolicy(): Policy {
    switch (this.mode) {
      case "uniform":
        return uniformPolicy(this.mdp);
      case "optimal":
        return optimalPolicy(this.mdp);
      case "epsilon":
        return epsilonSoftOptimal(this.mdp, this.eps);
      case "custom":
        return { pi: this.custom };
    }
  }

  private bump(a: number): void {
    if (this.sel == null) return;
    const row = this.custom[this.sel];
    row[a] = Math.min(1, row[a] + 0.1);
    const sum = row.reduce((x, y) => x + y, 0);
    for (let i = 0; i < row.length; i++) row[i] /= sum;
    this.refresh();
  }

  private refresh(): void {
    this.gr.update({ policy: this.currentPolicy() });
    this.renderBars();
  }

  private renderBars(): void {
    this.bars.innerHTML = "";
    if (this.sel == null) {
      this.bars.innerHTML = `<p class="mdp-readout">Click a cell to inspect π(·|s).</p>`;
      return;
    }
    const { r, c } = rc(this.sel);
    const pol = this.currentPolicy();
    const head = document.createElement("p");
    head.className = "mdp-readout";
    head.innerHTML = `π(·|s) at <span class="mdp-stat">(${r},${c})</span>`;
    this.bars.appendChild(head);

    for (let a = 0; a < this.mdp.nA; a++) {
      const w = pol.pi[this.sel][a];
      const line = document.createElement("div");
      line.style.cssText = "display:flex;align-items:center;gap:8px;margin:3px 0;font-family:var(--rl-font-ui);font-size:12px";
      const track = document.createElement("div");
      track.style.cssText = "flex:1;height:14px;background:var(--rl-surface-2);border-radius:3px;overflow:hidden";
      const fill = document.createElement("div");
      fill.style.cssText = `height:100%;width:${(w * 100).toFixed(1)}%;background:${cssVar(ACTION_VARS[a])}`;
      track.appendChild(fill);
      const lab = document.createElement("span");
      lab.style.cssText = "width:42px;color:var(--rl-ink-muted)";
      lab.textContent = ACTION_NAMES[a];
      const val = document.createElement("span");
      val.className = "mdp-stat";
      val.style.width = "38px";
      val.textContent = w.toFixed(2);
      line.append(lab, track, val);
      if (this.mode === "custom") {
        const plus = document.createElement("button");
        plus.type = "button";
        plus.textContent = "+0.1";
        plus.style.cssText = "font-size:11px;padding:1px 6px";
        plus.setAttribute("aria-label", `increase ${ACTION_NAMES[a]} weight`);
        plus.addEventListener("click", () => this.bump(a));
        line.appendChild(plus);
      }
      this.bars.appendChild(line);
    }
    if (this.mode !== "custom") {
      const hint = document.createElement("p");
      hint.className = "mdp-readout";
      hint.style.fontStyle = "italic";
      hint.textContent = "Switch to “custom” to edit weights.";
      this.bars.appendChild(hint);
    }
  }
}

customElements.define("policy-explorer", PolicyExplorer);
