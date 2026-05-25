/**
 * V3 — State Classification Inspector. Edit a chain (≤ 6 states) via the shared
 * ChainEditor; the TransitionGraph recolors nodes by communicating class in
 * real time, and a textual summary reports reducibility, recurrence/transience,
 * and period per class. Three presets: weather, periodic, reducible.
 */
import { createPanel, type PanelHandle } from "./PanelChrome";
import { ChainEditor } from "./ChainEditor";
import { TransitionGraph } from "./TransitionGraph";
import { MarkovChain } from "../markov/chain";
import { weather, periodic2, reducible } from "../markov/presets";

const CLASS_VARS = [
  "--mc-state-1", "--mc-state-2", "--mc-state-3", "--mc-state-4",
  "--mc-state-5", "--mc-state-6", "--mc-state-7", "--mc-state-8",
];
const CLASS_TINTS = ["--mc-tint-1", "--mc-tint-2", "--mc-tint-3", "--mc-tint-4"];

export class StateClassificationInspector extends HTMLElement {
  private P: number[][] = weather.P.map((r) => r.slice());

  private panel!: PanelHandle;
  private editor!: ChainEditor;
  private graph!: TransitionGraph;
  private summaryEl!: HTMLElement;

  connectedCallback(): void {
    this.render();
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({ id: "state-classification-inspector", heavy: true,
      mobileNotice: "The classification inspector is interactive — view on a wider screen." });
    this.panel.body.append(this.buildPresetBar());

    const row = document.createElement("div");
    row.className = "mc-row";

    const editorCol = document.createElement("div");
    editorCol.className = "mc-col";
    this.editor = new ChainEditor(editorCol, {
      P: this.P,
      maxStates: 6,
      minStates: 2,
      onChange: (P) => {
        this.P = P;
        this.refresh();
      },
    });

    const graphCol = document.createElement("div");
    graphCol.className = "mc-col";
    this.graph = new TransitionGraph(graphCol, { P: this.P, width: 420, height: 360 });

    row.append(editorCol, graphCol);

    this.summaryEl = document.createElement("div");
    this.summaryEl.className = "mc-readout";

    this.panel.body.append(row, this.summaryEl);
    this.appendChild(this.panel.panel);

    this.refresh();
  }

  private buildPresetBar(): HTMLElement {
    const bar = document.createElement("div");
    bar.className = "rl-controls";
    bar.style.marginBottom = "12px";
    bar.append("Load preset ");
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", "load preset chain");
    const presets = [
      { key: "weather", label: "weather (irreducible, aperiodic)", P: weather.P },
      { key: "periodic", label: "periodic (period 2)", P: periodic2.P },
      { key: "reducible", label: "reducible (2 classes)", P: reducible.P },
    ];
    for (const p of presets) {
      const o = document.createElement("option");
      o.value = p.key;
      o.textContent = p.label;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      const p = presets.find((x) => x.key === sel.value)!;
      this.P = p.P.map((r) => r.slice());
      this.editor.setMatrix(this.P, { silent: true });
      this.refresh();
    });
    bar.appendChild(sel);
    return bar;
  }

  private refresh(): void {
    let chain: MarkovChain;
    try {
      chain = new MarkovChain(this.P);
    } catch {
      this.summaryEl.textContent = "Invalid matrix (rows must sum to 1).";
      return;
    }
    const { classes, stateToClass } = chain.classify();
    // Nodes keep their per-state identity color (matching the editor's dots);
    // communicating class is shown as a background tint halo, per the spec.
    const tint = (i: number) => `var(${CLASS_TINTS[stateToClass[i] % CLASS_TINTS.length]})`;
    this.graph.update({ P: this.P, classTint: tint, width: 420, height: 360 });

    const irreducible = classes.length === 1;
    const letter = (ci: number) => String.fromCharCode(65 + ci);
    const classLine = classes
      .map((c, ci) => {
        const swatch = `<span class="mc-badge" style="background:var(${CLASS_TINTS[ci % CLASS_TINTS.length]});color:var(${CLASS_VARS[ci % CLASS_VARS.length]})">Class ${letter(ci)}</span>`;
        const rec = c.recurrent
          ? `<span class="mc-badge" style="background:var(--mc-tint-1);color:var(--mc-recurrent)">recurrent</span>`
          : `<span class="mc-badge" style="background:var(--mc-tint-2);color:var(--mc-transient)">transient</span>`;
        const per =
          c.period === 1
            ? `<span class="mc-badge" style="background:var(--mc-tint-3);color:var(--mc-stationary)">aperiodic</span>`
            : `<span class="mc-badge" style="background:rgba(185,28,28,0.12);color:var(--mc-periodic)">period ${c.period}</span>`;
        return `${swatch} {${c.members.join(", ")}} ${rec} ${per}`;
      })
      .join("<br>");

    const headline = irreducible
      ? `Chain is <strong>irreducible</strong> — a single communicating class.`
      : `Chain is <strong>reducible</strong>: ${classes.length} communicating classes.`;
    this.summaryEl.innerHTML = `${headline}<br>${classLine}`;
    this.panel.setStatus(`${chain.K} states · ${classes.length} ${classes.length === 1 ? "class" : "classes"}`);
  }
}

customElements.define("state-classification-inspector", StateClassificationInspector);
