/**
 * V5 — Convergence Lab (centerpiece). Four panels, all synchronized to one
 * shared `currentStep` held in a Store:
 *
 *   A  chain selector + matrix heatmap + eigenvalues on the complex unit circle
 *   B  distribution evolution — stacked area of μₜ over t, with a step cursor
 *   C  TV distance ‖μₜ − π‖_TV vs t on a log axis, with the λ⋆ⁿ reference bound
 *   D  a single sampled path tape + empirical visit frequencies converging to π
 *
 * The play loop and every control mutate the store; all four panels re-render
 * from the single shared state. Selecting "custom" reveals the shared
 * ChainEditor and drives P from it.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { ChainEditor } from "./ChainEditor";
import { DistributionBars } from "./markov-bars";
import { Store } from "./store";
import { MarkovChain, totalVariation, complexAbs, type Complex } from "../markov/chain";
import { CONVERGENCE_LAB_PRESETS } from "../markov/presets";
import { mulberry32 } from "../bandits/stats";
import { prefersReducedMotion } from "./base";

const MAXT = 100;
const NS = "http://www.w3.org/2000/svg";
const STATE_VARS = [
  "--mc-state-1", "--mc-state-2", "--mc-state-3", "--mc-state-4",
  "--mc-state-5", "--mc-state-6", "--mc-state-7", "--mc-state-8",
];

type InitKind = "uniform" | "delta0" | "deltaLast";

interface LabState {
  chainKey: string;
  P: number[][];
  currentStep: number;
  playing: boolean;
  speed: number;
  initKind: InitKind;
}

export class ConvergenceLab extends HTMLElement {
  private store!: Store<LabState>;
  private timer = 0;

  // derived model (rebuilt when P / init changes)
  private chain!: MarkovChain;
  private pi: number[] = [];
  private eig: Complex[] = [];
  private lambdaStar = 0;
  private muSeries: number[][] = [];
  private tvSeries: number[] = [];
  private path: number[] = [];
  private lastSig = "";

  private panel!: PanelHandle;
  private editorWrap!: HTMLElement;
  private editor?: ChainEditor;
  private freqBars!: DistributionBars;

  // panel SVGs / nodes
  private heatEl!: SVGSVGElement;
  private eigEl!: SVGSVGElement;
  private evoEl!: SVGSVGElement;
  private tvEl!: SVGSVGElement;
  private tapeEl!: HTMLElement;
  private stepLabel!: HTMLElement;
  private slider!: HTMLInputElement;
  private playBtn!: HTMLButtonElement;
  private metaEl!: HTMLElement;

  connectedCallback(): void {
    const first = CONVERGENCE_LAB_PRESETS[0];
    this.store = new Store<LabState>({
      chainKey: first.key,
      P: first.P.map((r) => r.slice()),
      currentStep: 0,
      playing: false,
      speed: 4,
      initKind: "uniform",
    });
    this.render();
    // All state changes funnel through sync(): it rebuilds the derived model
    // only when P / init actually change, then redraws every panel. This keeps
    // the four panels consistent and avoids drawing against a stale model.
    this.store.subscribe(() => this.sync());
    this.sync();
  }

  /** Rebuild the model iff the chain/init changed, then redraw all panels. */
  private sync(): void {
    const s = this.store.get();
    const sig = `${s.chainKey}|${s.initKind}|${JSON.stringify(s.P)}`;
    if (sig !== this.lastSig) {
      this.lastSig = sig;
      this.rebuildModel();
    }
    this.drawAll();
  }

  disconnectedCallback(): void {
    this.stop();
  }

  // ---- model ----

  private initialDist(): number[] {
    const K = this.store.get().P.length;
    const k = this.store.get().initKind;
    if (k === "delta0") return Array.from({ length: K }, (_, i) => (i === 0 ? 1 : 0));
    if (k === "deltaLast") return Array.from({ length: K }, (_, i) => (i === K - 1 ? 1 : 0));
    return new Array<number>(K).fill(1 / K);
  }

  private rebuildModel(): void {
    const { P, initKind } = this.store.get();
    this.chain = new MarkovChain(P);
    this.pi = this.chain.stationary();
    this.eig = this.chain.eigenvalues();
    this.lambdaStar = this.chain.lambdaStar();

    const mu0 = this.initialDist();
    this.muSeries = [mu0.slice()];
    let mu = mu0.slice();
    for (let t = 1; t <= MAXT; t++) {
      const next = new Array<number>(P.length).fill(0);
      for (let j = 0; j < P.length; j++)
        for (let i = 0; i < P.length; i++) next[j] += mu[i] * P[i][j];
      mu = next;
      this.muSeries.push(mu.slice());
    }
    this.tvSeries = this.muSeries.map((m) => totalVariation(m, this.pi));

    const start = initKind === "deltaLast" ? P.length - 1 : 0;
    this.path = this.chain.sampleTrajectory(start, MAXT + 1, mulberry32(12345));
  }

  // ---- rendering scaffold ----

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "convergence-lab",
      arena: true,
      heavy: true,
      mobileNotice: "The Convergence Lab is a four-panel interactive — view on a wider screen.",
    });

    this.panel.body.append(this.buildControls());

    this.editorWrap = document.createElement("div");
    this.editorWrap.style.margin = "8px 0";
    this.panel.body.append(this.editorWrap);

    const grid = document.createElement("div");
    grid.className = "mc-lab-grid";
    grid.append(
      this.buildPanelA(),
      this.buildPanelB(),
      this.buildPanelC(),
      this.buildPanelD(),
    );
    this.panel.body.append(grid);

    this.metaEl = document.createElement("div");
    this.metaEl.className = "mc-readout";
    this.panel.body.append(this.metaEl);

    this.appendChild(this.panel.panel);
  }

  private mkSvg(parent: HTMLElement, vbW: number, vbH: number): SVGSVGElement {
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${vbW} ${vbH}`);
    s.setAttribute("width", "100%");
    s.classList.add("rl-svg");
    parent.appendChild(s);
    return s;
  }

  private labPanel(title: string, sub: string): HTMLElement {
    const p = document.createElement("div");
    p.className = "mc-lab-panel";
    const t = document.createElement("p");
    t.className = "mc-lab-panel__title";
    t.innerHTML = `${title} <small>${sub}</small>`;
    p.appendChild(t);
    return p;
  }

  private buildPanelA(): HTMLElement {
    const p = this.labPanel("A · Chain & spectrum", "matrix + eigenvalues on the unit circle");
    const selBar = document.createElement("div");
    selBar.className = "rl-controls";
    selBar.style.marginBottom = "8px";
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", "select chain");
    for (const preset of CONVERGENCE_LAB_PRESETS) {
      const o = document.createElement("option");
      o.value = preset.key;
      o.textContent = preset.label;
      sel.appendChild(o);
    }
    const custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "custom (edit matrix)";
    sel.appendChild(custom);
    sel.value = this.store.get().chainKey;
    sel.addEventListener("change", () => this.selectChain(sel.value));
    selBar.appendChild(sel);
    p.appendChild(selBar);

    const row = document.createElement("div");
    row.className = "mc-row";
    row.style.gap = "10px";
    const heatCol = document.createElement("div");
    this.heatEl = this.mkSvg(heatCol, 200, 200);
    const eigCol = document.createElement("div");
    this.eigEl = this.mkSvg(eigCol, 200, 200);
    row.append(heatCol, eigCol);
    p.appendChild(row);
    return p;
  }

  private buildPanelB(): HTMLElement {
    const p = this.labPanel("B · Distribution evolution", "stacked μₜ; cursor at t");
    this.evoEl = this.mkSvg(p, 420, 240);
    return p;
  }

  private buildPanelC(): HTMLElement {
    const p = this.labPanel("C · Distance to stationarity", "‖μₜ − π‖_TV (log), with λ⋆ⁿ bound");
    this.tvEl = this.mkSvg(p, 420, 240);
    return p;
  }

  private buildPanelD(): HTMLElement {
    const p = this.labPanel("D · A sample path", "states visited + empirical frequencies");
    this.tapeEl = document.createElement("div");
    this.tapeEl.className = "mc-tape";
    p.appendChild(this.tapeEl);
    this.freqBars = new DistributionBars(p, { width: 380, rowHeight: 26 });
    return p;
  }

  private buildControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";

    this.playBtn = document.createElement("button");
    this.playBtn.className = "primary";
    this.playBtn.textContent = "Play";
    this.playBtn.addEventListener("click", () => this.togglePlay());

    const stepBtn = document.createElement("button");
    stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => {
      this.stop();
      this.store.set({ currentStep: Math.min(MAXT, this.store.get().currentStep + 1), playing: false });
    });

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.stop();
      this.store.set({ currentStep: 0, playing: false });
    });

    const initWrap = document.createElement("label");
    initWrap.append("start ");
    const initSel = document.createElement("select");
    for (const [v, label] of [
      ["uniform", "uniform"],
      ["delta0", "δ at state 0"],
      ["deltaLast", "δ at last state"],
    ] as [InitKind, string][]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      initSel.appendChild(o);
    }
    initSel.value = this.store.get().initKind;
    initSel.addEventListener("change", () => {
      this.stop();
      this.store.set({ initKind: initSel.value as InitKind, currentStep: 0, playing: false });
    });
    initWrap.appendChild(initSel);

    const spWrap = document.createElement("label");
    spWrap.append("speed ");
    const sp = document.createElement("select");
    for (const s of [1, 2, 4, 8, 16, 32]) {
      const o = document.createElement("option");
      o.value = String(s);
      o.textContent = `${s}×`;
      if (s === 4) o.selected = true;
      sp.appendChild(o);
    }
    sp.addEventListener("change", () => this.store.set({ speed: +sp.value }));
    spWrap.appendChild(sp);

    const tWrap = document.createElement("label");
    this.slider = document.createElement("input");
    this.slider.type = "range";
    this.slider.min = "0";
    this.slider.max = String(MAXT);
    this.slider.step = "1";
    this.slider.value = "0";
    this.slider.setAttribute("aria-label", "step t");
    this.slider.addEventListener("input", () => {
      this.stop();
      this.store.set({ currentStep: +this.slider.value, playing: false });
    });
    this.stepLabel = document.createElement("span");
    this.stepLabel.className = "rl-mono";
    this.stepLabel.textContent = "t=0";
    tWrap.append("t ", this.slider, this.stepLabel);

    c.append(this.playBtn, stepBtn, resetBtn, initWrap, spWrap, tWrap);
    return c;
  }

  // ---- control logic ----

  private selectChain(key: string): void {
    this.stop();
    if (key === "custom") {
      const cur = this.store.get().P;
      this.store.set({ chainKey: "custom", currentStep: 0, playing: false });
      this.showEditor(cur);
    } else {
      const preset = CONVERGENCE_LAB_PRESETS.find((p) => p.key === key)!;
      this.hideEditor();
      this.store.set({ chainKey: key, P: preset.P.map((r) => r.slice()), currentStep: 0, playing: false });
    }
  }

  private showEditor(P: number[][]): void {
    this.hideEditor();
    this.editor = new ChainEditor(this.editorWrap, {
      P,
      maxStates: 6,
      minStates: 2,
      onChange: (next) => {
        this.store.set({ P: next, currentStep: 0, playing: false });
      },
    });
  }

  private hideEditor(): void {
    this.editor?.destroy();
    this.editor = undefined;
    this.editorWrap.innerHTML = "";
  }

  private togglePlay(): void {
    if (this.timer) {
      this.stop();
      return;
    }
    if (this.store.get().currentStep >= MAXT) this.store.set({ currentStep: 0 });
    this.store.set({ playing: true });
    this.playBtn.textContent = "Pause";
    const tick = () => {
      const { speed, currentStep } = this.store.get();
      const targetMs = 250 / speed;
      const stepsPerTick = Math.max(1, Math.round(40 / targetMs));
      const next = Math.min(MAXT, currentStep + stepsPerTick);
      this.store.set({ currentStep: next });
      if (next >= MAXT) this.stop();
    };
    const interval = prefersReducedMotion() ? 250 : Math.max(40, 250 / this.store.get().speed);
    this.timer = window.setInterval(tick, interval);
  }

  private stop(): void {
    if (this.timer) {
      window.clearInterval(this.timer);
      this.timer = 0;
    }
    if (this.store.get().playing) this.store.set({ playing: false });
    this.playBtn.textContent = "Play";
  }

  // ---- drawing ----

  private color(i: number): string {
    return `var(${STATE_VARS[i % STATE_VARS.length]})`;
  }

  private drawAll(): void {
    const { currentStep } = this.store.get();
    this.slider.value = String(currentStep);
    this.stepLabel.textContent = `t=${currentStep}`;
    this.drawHeat();
    this.drawEig();
    this.drawEvolution();
    this.drawTV();
    this.drawTape();
    this.drawMeta();
    this.panel.setStatus(`${this.store.get().chainKey} · t=${currentStep}/${MAXT}`);
  }

  private drawHeat(): void {
    const P = this.store.get().P;
    const K = P.length;
    const svg = d3.select(this.heatEl);
    const size = 190;
    const cell = size / K;
    svg.attr("viewBox", `0 0 ${size} ${size}`).selectAll("*").remove();
    const interp = d3.interpolateRgb("#ffffff", "#1c1e22");
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const v = P[i][j];
        svg.append("rect")
          .attr("x", j * cell)
          .attr("y", i * cell)
          .attr("width", cell - 1.5)
          .attr("height", cell - 1.5)
          .attr("fill", interp(v))
          .attr("stroke", "var(--rl-border)")
          .attr("stroke-width", 0.5);
        if (K <= 4)
          svg.append("text")
            .attr("x", j * cell + cell / 2)
            .attr("y", i * cell + cell / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("font-family", "var(--rl-font-mono)")
            .style("font-size", "9px")
            .style("fill", v > 0.55 ? "#fff" : "var(--rl-ink-muted)")
            .text(v.toFixed(2));
      }
    }
  }

  private drawEig(): void {
    const svg = d3.select(this.eigEl);
    const S = 200;
    const cx = S / 2;
    const cy = S / 2;
    const R = 78;
    svg.attr("viewBox", `0 0 ${S} ${S}`).selectAll("*").remove();
    // axes + unit circle
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R)
      .attr("fill", "none").attr("stroke", "var(--rl-ink-faint)").attr("stroke-dasharray", "3 3");
    svg.append("line").attr("x1", cx - R - 10).attr("x2", cx + R + 10).attr("y1", cy).attr("y2", cy)
      .attr("stroke", "var(--rl-border)");
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", cy - R - 10).attr("y2", cy + R + 10)
      .attr("stroke", "var(--rl-border)");
    for (const e of this.eig) {
      const mag = complexAbs(e);
      const onCircle = Math.abs(mag - 1) < 1e-6;
      const isUnit = Math.abs(e.re - 1) < 1e-6 && Math.abs(e.im) < 1e-6;
      const fill = isUnit
        ? "var(--mc-stationary)"
        : onCircle
          ? "var(--mc-periodic)"
          : "var(--rl-ink-muted)";
      svg.append("circle")
        .attr("cx", cx + e.re * R)
        .attr("cy", cy - e.im * R)
        .attr("r", isUnit ? 5 : 4)
        .attr("fill", fill)
        .attr("opacity", 0.9)
        .append("title")
        .text(`λ = ${e.re.toFixed(3)}${e.im >= 0 ? "+" : "−"}${Math.abs(e.im).toFixed(3)}i  |λ|=${mag.toFixed(3)}`);
    }
    svg.append("text").attr("x", cx + R + 2).attr("y", cy - 4).attr("class", "annot")
      .style("font-family", "var(--rl-font-mono)").style("font-size", "9px")
      .style("fill", "var(--rl-ink-faint)").text("1");
  }

  private drawEvolution(): void {
    const W = 420;
    const H = 240;
    const m = { top: 10, right: 12, bottom: 24, left: 30 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const K = this.store.get().P.length;
    const svg = d3.select(this.evoEl);
    svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
    const x = d3.scaleLinear().domain([0, MAXT]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    // stacked areas
    for (let s = 0; s < K; s++) {
      const pts = this.muSeries.map((mu, t) => {
        let lower = 0;
        for (let k = 0; k < s; k++) lower += mu[k];
        const upper = lower + mu[s];
        return { t, lower, upper };
      });
      const area = d3
        .area<{ t: number; lower: number; upper: number }>()
        .x((d) => x(d.t))
        .y0((d) => y(d.lower))
        .y1((d) => y(d.upper));
      g.append("path").datum(pts).attr("fill", this.color(s)).attr("opacity", 0.85).attr("d", area);
    }
    // axes
    g.append("g").attr("class", "rl-axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "rl-axis").call(d3.axisLeft(y).ticks(3).tickFormat(d3.format(".0%")));
    // cursor
    const cstep = this.store.get().currentStep;
    g.append("line").attr("x1", x(cstep)).attr("x2", x(cstep)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", "var(--mc-current)").attr("stroke-width", 1.5);
  }

  private drawTV(): void {
    const W = 420;
    const H = 240;
    const m = { top: 10, right: 12, bottom: 24, left: 46 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const svg = d3.select(this.tvEl);
    svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const floor = 1e-6;
    const clamp = (v: number) => Math.max(floor, v);
    const x = d3.scaleLinear().domain([0, MAXT]).range([0, iw]);
    const yMax = Math.max(...this.tvSeries, 0.01);
    const y = d3.scaleLog().domain([floor, Math.max(yMax, floor * 10)]).range([ih, 0]).clamp(true);

    g.append("g").attr("class", "rl-axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "rl-axis")
      .call(d3.axisLeft(y).ticks(4, d3.format("~e")));

    // theoretical bound C·λ⋆ⁿ (C = tv[0] or 1)
    if (this.lambdaStar > 0 && this.lambdaStar < 1) {
      const C = Math.max(this.tvSeries[0], 1e-3);
      const bound = d3.range(0, MAXT + 1).map((t) => ({ t, v: clamp(C * Math.pow(this.lambdaStar, t)) }));
      g.append("path").datum(bound).attr("fill", "none").attr("stroke", "var(--rl-ink-faint)")
        .attr("stroke-width", 1.2).attr("stroke-dasharray", "5 4")
        .attr("d", d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v)));
    }
    // tv curve
    const pts = this.tvSeries.map((v, t) => ({ t, v: clamp(v) }));
    g.append("path").datum(pts).attr("fill", "none").attr("stroke", "var(--mc-stationary)")
      .attr("stroke-width", 2)
      .attr("d", d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v)));
    // cursor + current dot
    const cstep = this.store.get().currentStep;
    g.append("line").attr("x1", x(cstep)).attr("x2", x(cstep)).attr("y1", 0).attr("y2", ih)
      .attr("stroke", "var(--mc-current)").attr("stroke-width", 1.5);
    g.append("circle").attr("cx", x(cstep)).attr("cy", y(clamp(this.tvSeries[cstep]))).attr("r", 3.5)
      .attr("fill", "var(--mc-current)");
  }

  private drawTape(): void {
    const cstep = this.store.get().currentStep;
    const K = this.store.get().P.length;
    this.tapeEl.innerHTML = "";
    const visible = this.path.slice(0, cstep + 1);
    const tail = visible.slice(-64);
    for (const s of tail) {
      const cell = document.createElement("div");
      cell.className = "mc-tape__cell";
      cell.style.background = this.color(s);
      cell.title = `state ${s}`;
      this.tapeEl.appendChild(cell);
    }
    // empirical frequencies over the path so far
    const counts = new Array<number>(K).fill(0);
    for (const s of visible) counts[s]++;
    const freq = counts.map((c) => c / visible.length);
    this.freqBars.update(freq, this.pi);
  }

  private drawMeta(): void {
    const { chainKey } = this.store.get();
    const irr = this.chain.isIrreducible();
    const aper = irr ? this.chain.isAperiodic() : false;
    const verdict = !irr
      ? "reducible — limit depends on the start; no single π for all μ₀"
      : aper
        ? "ergodic — μₜ → π geometrically at rate λ⋆"
        : "periodic — μₜ oscillates; only the time-average converges to π";
    this.metaEl.innerHTML =
      `<strong>${chainKey}</strong>: ${verdict}.<br>` +
      `λ⋆ = <span class="rl-mono">${this.lambdaStar.toFixed(4)}</span> · ` +
      `spectral gap = <span class="rl-mono">${(1 - this.lambdaStar).toFixed(4)}</span> · ` +
      `π = (${this.pi.map((x) => x.toFixed(3)).join(", ")})`;
  }
}

customElements.define("convergence-lab", ConvergenceLab);
