/**
 * V7 — Policy-Induced Chain Preview. A 4×4 gridworld with a policy π(a|s) drawn
 * as per-cell action arrows. Marginalizing actions gives the induced chain
 * Pᵖ_{ss'} = Σ_a π(a|s)·[move(s,a)=s'] — an ordinary 16×16 stochastic matrix,
 * shown as a heatmap. Its stationary distribution dᵖ shades the grid (darker =
 * visited more), and a sampled trajectory walks an agent token cell to cell.
 *
 * This is deliberately a *preview*, not an MDP: there are no rewards. It only
 * makes "a policy turns an MDP into a Markov chain on states" visceral.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { MarkovChain } from "../markov/chain";
import { mulberry32 } from "../bandits/stats";
import { prefersReducedMotion } from "./base";

const NS = "http://www.w3.org/2000/svg";
const N = 4; // grid side
const K = N * N;
// action deltas: up, down, left, right (row, col)
const ACTIONS: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]];

type PolicyKey = "uniform" | "right" | "diagonal";
const POLICIES: Record<PolicyKey, number[]> = {
  uniform: [0.25, 0.25, 0.25, 0.25],
  right: [0.1, 0.1, 0.1, 0.7],
  diagonal: [0.15, 0.35, 0.15, 0.35],
};

export class PolicyInducedChainPreview extends HTMLElement {
  private policy: PolicyKey = "uniform";
  private showD = true;
  private t = 0;
  private timer = 0;

  private P: number[][] = [];
  private d: number[] = [];
  private path: number[] = [];

  private panel!: PanelHandle;
  private gridEl!: SVGSVGElement;
  private heatEl!: SVGSVGElement;
  private token!: d3.Selection<SVGCircleElement, unknown, null, undefined>;
  private readoutEl!: HTMLElement;
  private playBtn!: HTMLButtonElement;

  connectedCallback(): void {
    this.rebuild();
    this.render();
  }

  disconnectedCallback(): void {
    this.stop();
  }

  private move(s: number, a: number): number {
    const r = Math.floor(s / N);
    const c = s % N;
    const nr = r + ACTIONS[a][0];
    const nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= N || nc < 0 || nc >= N) return s; // wall → stay
    return nr * N + nc;
  }

  private rebuild(): void {
    const pi = POLICIES[this.policy];
    this.P = Array.from({ length: K }, () => new Array<number>(K).fill(0));
    for (let s = 0; s < K; s++) {
      for (let a = 0; a < 4; a++) {
        if (pi[a] <= 0) continue;
        this.P[s][this.move(s, a)] += pi[a];
      }
    }
    const chain = new MarkovChain(this.P);
    this.d = chain.stationary();
    this.path = chain.sampleTrajectory(0, 400, mulberry32(2024));
    this.t = 0;
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "policy-induced-chain-preview",
      heavy: true,
      mobileNotice: "The policy-induced-chain preview is interactive — view on a wider screen.",
    });
    this.panel.body.append(this.buildControls());

    const row = document.createElement("div");
    row.className = "mc-row";

    const gridCol = document.createElement("div");
    gridCol.className = "mc-col";
    gridCol.append(this.smallLabel("gridworld: policy arrows + dᵖ shading + agent"));
    this.gridEl = document.createElementNS(NS, "svg");
    this.gridEl.setAttribute("viewBox", "0 0 300 300");
    this.gridEl.setAttribute("width", "100%");
    this.gridEl.style.maxWidth = "320px";
    this.gridEl.classList.add("rl-svg");
    gridCol.append(this.gridEl);

    const heatCol = document.createElement("div");
    heatCol.className = "mc-col";
    heatCol.append(this.smallLabel("induced chain Pᵖ (16×16)"));
    this.heatEl = document.createElementNS(NS, "svg");
    this.heatEl.setAttribute("viewBox", "0 0 240 240");
    this.heatEl.setAttribute("width", "100%");
    this.heatEl.style.maxWidth = "260px";
    this.heatEl.classList.add("rl-svg");
    heatCol.append(this.heatEl);

    row.append(gridCol, heatCol);

    this.readoutEl = document.createElement("div");
    this.readoutEl.className = "mc-readout";

    this.panel.body.append(row, this.readoutEl);
    this.appendChild(this.panel.panel);

    this.drawGrid();
    this.drawHeat();
    this.drawReadout();
  }

  private smallLabel(text: string): HTMLElement {
    const d = document.createElement("div");
    d.className = "axis-label";
    d.style.cssText =
      "font-family:var(--rl-font-ui);font-size:11px;color:var(--rl-ink-muted);margin-bottom:4px";
    d.textContent = text;
    return d;
  }

  private buildControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";
    c.style.marginBottom = "10px";

    c.append("policy ");
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", "policy");
    for (const [v, label] of [
      ["uniform", "uniform random"],
      ["right", "go-right-biased"],
      ["diagonal", "diagonal-biased"],
    ] as [PolicyKey, string][]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      this.stop();
      this.policy = sel.value as PolicyKey;
      this.rebuild();
      this.drawGrid();
      this.drawHeat();
      this.drawReadout();
    });
    c.appendChild(sel);

    this.playBtn = document.createElement("button");
    this.playBtn.className = "primary";
    this.playBtn.textContent = "Play path";
    this.playBtn.addEventListener("click", () => this.togglePlay());

    const stepBtn = document.createElement("button");
    stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => {
      this.stop();
      this.advance();
    });

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.stop();
      this.t = 0;
      this.moveToken(false);
      this.drawReadout();
    });

    const dWrap = document.createElement("label");
    const dchk = document.createElement("input");
    dchk.type = "checkbox";
    dchk.checked = this.showD;
    dchk.addEventListener("change", () => {
      this.showD = dchk.checked;
      this.drawGrid();
    });
    dWrap.append(dchk, " show dᵖ shading");

    c.append(this.playBtn, stepBtn, resetBtn, dWrap);
    return c;
  }

  private cellCenter(s: number): { x: number; y: number } {
    const cell = 70;
    const pad = 10;
    const r = Math.floor(s / N);
    const c = s % N;
    return { x: pad + c * cell + cell / 2, y: pad + r * cell + cell / 2 };
  }

  private drawGrid(): void {
    const svg = d3.select(this.gridEl);
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "pi-arrow").attr("viewBox", "0 0 10 10")
      .attr("refX", 8).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6)
      .attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "var(--mc-edge-strong)");

    const cell = 70;
    const pad = 10;
    const dMax = Math.max(...this.d);
    const shade = d3.interpolateRgb("#ffffff", "#15803d");
    const pi = POLICIES[this.policy];

    for (let s = 0; s < K; s++) {
      const r = Math.floor(s / N);
      const c = s % N;
      const x = pad + c * cell;
      const y = pad + r * cell;
      svg.append("rect").attr("x", x).attr("y", y).attr("width", cell).attr("height", cell)
        .attr("fill", this.showD ? shade(this.d[s] / dMax) : "var(--rl-surface)")
        .attr("stroke", "var(--rl-border)");
      if (this.showD)
        svg.append("text").attr("x", x + 5).attr("y", y + 13)
          .style("font-family", "var(--rl-font-mono)").style("font-size", "8px")
          .style("fill", this.d[s] / dMax > 0.6 ? "#fff" : "var(--rl-ink-faint)")
          .text(this.d[s].toFixed(3));

      // policy arrows
      const ctr = this.cellCenter(s);
      for (let a = 0; a < 4; a++) {
        if (pi[a] <= 0) continue;
        const len = 7 + 20 * pi[a];
        const dx = ACTIONS[a][1] * len;
        const dy = ACTIONS[a][0] * len;
        svg.append("line").attr("x1", ctr.x).attr("y1", ctr.y)
          .attr("x2", ctr.x + dx).attr("y2", ctr.y + dy)
          .attr("stroke", "var(--mc-edge-strong)").attr("stroke-width", 1 + 3 * pi[a])
          .attr("opacity", 0.7).attr("marker-end", "url(#pi-arrow)");
      }
    }

    // agent token
    const c0 = this.cellCenter(this.path[this.t]);
    this.token = svg.append("circle").attr("cx", c0.x).attr("cy", c0.y).attr("r", 11)
      .attr("fill", "var(--mc-current)").attr("stroke", "#fff").attr("stroke-width", 2);
  }

  private drawHeat(): void {
    const svg = d3.select(this.heatEl);
    svg.selectAll("*").remove();
    const cell = 224 / K;
    const interp = d3.interpolateRgb("#ffffff", "#1c1e22");
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++) {
        if (this.P[i][j] <= 1e-9) continue;
        svg.append("rect").attr("x", 8 + j * cell).attr("y", 8 + i * cell)
          .attr("width", cell).attr("height", cell).attr("fill", interp(this.P[i][j]))
          .append("title").text(`P(${i}→${j}) = ${this.P[i][j].toFixed(2)}`);
      }
    svg.append("rect").attr("x", 8).attr("y", 8).attr("width", K * cell).attr("height", K * cell)
      .attr("fill", "none").attr("stroke", "var(--rl-border)");
  }

  private moveToken(animate: boolean): void {
    const ctr = this.cellCenter(this.path[this.t]);
    if (animate && !prefersReducedMotion()) {
      this.token.transition().duration(180).attr("cx", ctr.x).attr("cy", ctr.y);
    } else {
      this.token.attr("cx", ctr.x).attr("cy", ctr.y);
    }
  }

  private advance(): void {
    this.t = (this.t + 1) % this.path.length;
    this.moveToken(true);
    this.drawReadout();
  }

  private togglePlay(): void {
    if (this.timer) {
      this.stop();
      return;
    }
    this.playBtn.textContent = "Pause";
    const interval = prefersReducedMotion() ? 400 : 220;
    this.timer = window.setInterval(() => this.advance(), interval);
  }

  private stop(): void {
    if (this.timer) {
      window.clearInterval(this.timer);
      this.timer = 0;
    }
    this.playBtn.textContent = "Play path";
  }

  private drawReadout(): void {
    const s = this.path[this.t];
    this.readoutEl.innerHTML =
      `Pᵖ is an ordinary ${K}×${K} stochastic matrix — the chain on states induced by the policy. ` +
      `Its stationary distribution dᵖ is the long-run state-visitation frequency.<br>` +
      `agent at cell ${s} (row ${Math.floor(s / N)}, col ${s % N}) · t=${this.t}`;
    this.panel.setStatus(`${this.policy} · t=${this.t}`);
  }
}

customElements.define("policy-induced-chain-preview", PolicyInducedChainPreview);
