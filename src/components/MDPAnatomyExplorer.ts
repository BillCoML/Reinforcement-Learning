/**
 * V1 — MDP Anatomy Explorer. Click a non-terminal cell, pick an action, and
 * watch the dynamics: the target cell(s) light up with P(s'|s,a) and the reward,
 * and the MDP tuple fills in with the chosen (s, a, s', r). A deterministic /
 * slippery toggle fans the single arrow into the 80-10-10 slip distribution.
 * The point is tactile mastery of the gridworld before any equations appear.
 */
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { MDPEditor } from "./MDPEditor";
import { buildGridworld } from "../mdp/gridworld";
import { ACTION_NAMES, idx, rc, type MDP } from "../mdp/types";

export class MDPAnatomyExplorer extends HTMLElement {
  private mdp: MDP = buildGridworld({ slippery: false });
  private gr!: GridworldRenderer;
  private sel: number | null = null;
  private act: number | null = null;
  private info!: HTMLElement;
  private actionBtns: HTMLButtonElement[] = [];

  connectedCallback(): void {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "mdp-anatomy-explorer" });

    const row = document.createElement("div");
    row.className = "mdp-row";
    const gridCol = document.createElement("div");
    const infoCol = document.createElement("div");
    infoCol.className = "mdp-col";
    row.append(gridCol, infoCol);
    body.appendChild(row);

    this.gr = new GridworldRenderer(gridCol, {
      mdp: this.mdp,
      startState: idx(0, 0),
      onCellClick: (s) => this.selectCell(s),
    });

    // action buttons
    const btnRow = document.createElement("div");
    btnRow.className = "mdp-controls";
    this.actionBtns = ACTION_NAMES.map((name, a) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = name;
      b.disabled = true;
      b.setAttribute("aria-label", `take action ${name}`);
      b.addEventListener("click", () => {
        this.act = a;
        this.refresh();
      });
      btnRow.appendChild(b);
      return b;
    });
    infoCol.appendChild(btnRow);

    this.info = document.createElement("div");
    this.info.className = "mdp-readout";
    infoCol.appendChild(this.info);

    // deterministic / slippery toggle
    const editorWrap = document.createElement("div");
    new MDPEditor(editorWrap, {
      opts: { slippery: false, gamma: 0.9 },
      showGamma: false,
      onChange: (opts) => {
        this.mdp = buildGridworld(opts);
        this.act = null;
        this.gr.update({ mdp: this.mdp });
        this.refresh();
      },
    });
    infoCol.appendChild(editorWrap);

    this.appendChild(panel);
    this.refresh();
  }

  private selectCell(s: number): void {
    if (this.mdp.terminals[s]) return;
    this.sel = s;
    this.act = null;
    this.refresh();
  }

  private refresh(): void {
    const sel = this.sel;
    const act = this.act;
    const inputs = sel != null && act != null
      ? this.mdp.P[sel][act].map((p, sp) => (p > 0 ? sp : -1)).filter((x) => x >= 0)
      : [];
    this.gr.update({ selectedState: sel, highlightInputs: inputs, highlightState: null });

    this.actionBtns.forEach((b, a) => {
      b.disabled = sel == null;
      b.classList.toggle("is-active", a === act);
    });

    this.info.innerHTML = this.tupleHTML(sel, act);
  }

  private tupleHTML(sel: number | null, act: number | null): string {
    let h = `<p style="margin:0 0 8px"><span class="mdp-stat">𝓜 = (𝒮, 𝒜, P, r, γ)</span> · 9 states, 4 actions, γ = ${this.mdp.gamma.toFixed(2)}</p>`;
    if (sel == null) {
      h += `<p style="color:var(--rl-ink-faint)">Click a non-terminal cell to select a state, then pick an action.</p>`;
      return h;
    }
    const { r, c } = rc(sel);
    h += `<p>state <span class="mdp-stat">s = (${r},${c})</span>`;
    if (act == null) {
      h += ` — choose an action above.</p>`;
      return h;
    }
    h += ` · action <span class="mdp-stat">a = ${ACTION_NAMES[act]}</span></p>`;
    h += `<p style="margin:6px 0 2px;color:var(--rl-ink-faint)">resulting transitions:</p><ul style="margin:0;padding-left:18px">`;
    const row = this.mdp.P[sel][act];
    for (let sp = 0; sp < this.mdp.nS; sp++) {
      if (row[sp] <= 0) continue;
      const t = rc(sp);
      const enter = this.mdp.terminals[sp]
        ? sp === idx(2, 2) ? " (goal)" : sp === idx(1, 1) ? " (pit)" : ""
        : "";
      h += `<li><span class="mdp-stat">P((${t.r},${t.c}) | s,a) = ${row[sp].toFixed(2)}</span>${enter}</li>`;
    }
    h += `</ul>`;
    h += `<p style="margin:6px 0 0"><span class="mdp-stat">r(s,a) = ${this.mdp.r[sel][act].toFixed(2)}</span> &nbsp;<span style="color:var(--rl-ink-faint)">(expected reward on entry)</span></p>`;
    return h;
  }
}

customElements.define("mdp-anatomy-explorer", MDPAnatomyExplorer);
