/**
 * V2 — First-Visit vs Every-Visit Walkthrough.
 * Animates one trajectory at ~600ms/step. Two side-by-side gridworlds track
 * visit counts and value estimates (FV left, EV right). Below: running V̂(0,0)
 * convergence chart over the past 500 trajectories.
 * Width 800, Height 460.
 */
import * as d3 from "d3";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { rollout, type Step } from "../mdp/rollout";
import { computeReturns } from "../monte-carlo/returns";
import { mulberry32 } from "../importance-sampling/gaussian";
import { createPanel } from "./PanelChrome";
import { makeValueColorScale, textColorOn } from "./value-scale";
import { rc, idx, GRID_SIZE } from "../mdp/types";
import { prefersReducedMotion } from "./base";

const mdp = buildGridworld();
const uniform = uniformPolicy(mdp);

const W = 780, H_CHART = 140, MARGIN = { t: 28, r: 14, b: 36, l: 46 };
const CELL = 64, GAP = 6, PAD = 12;
const GRID_SVG = CELL * GRID_SIZE + GAP * (GRID_SIZE - 1) + 2 * PAD;
const TRUE_V = -0.4205;
const HISTORY_LEN = 500;
const STEP_MS = 600;

const NS_SVG = "http://www.w3.org/2000/svg";

interface MCState {
  V: number[];
  visits: number[];
  history: number[];   // V(0,0) over trajectories
  episodeCount: number;
}

function freshState(): MCState {
  return {
    V: new Array(mdp.nS).fill(0),
    visits: new Array(mdp.nS).fill(0),
    history: [],
    episodeCount: 0,
  };
}

function buildGridSVG(container: HTMLElement, label: string): {
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  cells: d3.Selection<SVGRectElement, unknown, null, undefined>[];
  counts: d3.Selection<SVGTextElement, unknown, null, undefined>[];
  vLabels: d3.Selection<SVGTextElement, unknown, null, undefined>[];
  agent: d3.Selection<SVGCircleElement, unknown, null, undefined>;
} {
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:inline-block;vertical-align:top;";
  const heading = document.createElement("p");
  heading.style.cssText = "margin:0 0 6px 12px;font-family:var(--rl-mono);font-size:11px;font-weight:600;";
  heading.textContent = label;
  wrap.appendChild(heading);
  container.appendChild(wrap);

  const svgEl = document.createElementNS(NS_SVG, "svg") as SVGSVGElement;
  svgEl.setAttribute("viewBox", `0 0 ${GRID_SVG} ${GRID_SVG}`);
  svgEl.setAttribute("width", `${GRID_SVG}`);
  svgEl.style.maxWidth = "100%";
  svgEl.style.height = "auto";
  wrap.appendChild(svgEl);

  const svg = d3.select(svgEl);
  const cells: d3.Selection<SVGRectElement, unknown, null, undefined>[] = [];
  const counts: d3.Selection<SVGTextElement, unknown, null, undefined>[] = [];
  const vLabels: d3.Selection<SVGTextElement, unknown, null, undefined>[] = [];

  for (let s = 0; s < mdp.nS; s++) {
    const { r, c } = rc(s);
    const x = PAD + c * (CELL + GAP);
    const y = PAD + r * (CELL + GAP);
    const cx = x + CELL / 2, cy = y + CELL / 2;

    const cell = svg.append("rect").attr("x", x).attr("y", y)
      .attr("width", CELL).attr("height", CELL).attr("rx", 4)
      .attr("fill", "var(--rl-surface)").attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
    cells.push(cell);

    if (mdp.terminals[s]) {
      svg.append("text").attr("x", cx).attr("y", cy + 4)
        .attr("text-anchor", "middle").attr("font-size", 18)
        .text(s === idx(1, 1) ? "☠" : "⚑");
    }

    const vLabel = svg.append("text").attr("x", cx).attr("y", cy + 4)
      .attr("text-anchor", "middle").attr("font-family", "var(--rl-mono)")
      .attr("font-size", 11).attr("fill", "var(--rl-ink)").text("");
    vLabels.push(vLabel);

    const count = svg.append("text").attr("x", x + 5).attr("y", y + 13)
      .attr("font-family", "var(--rl-mono)").attr("font-size", 9)
      .attr("fill", "var(--rl-ink-muted)").text("");
    counts.push(count);
  }

  // Agent circle
  const agent = svg.append("circle").attr("r", 10).attr("fill", "var(--mc-on-policy)")
    .attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", 0);
  agent.node();

  return { svg, cells, counts, vLabels, agent };
}

class FirstVsEveryVisitWalkthrough extends HTMLElement {
  private fv: MCState = freshState();
  private ev: MCState = freshState();
  private colorScale = makeValueColorScale([-1, 0.5]);
  private rng = mulberry32(1);
  private animTraj: Step[] | null = null;
  private animStep = 0;
  private animTimer: ReturnType<typeof setTimeout> | null = null;
  private playing = false;
  private fvGrid!: ReturnType<typeof buildGridSVG>;
  private evGrid!: ReturnType<typeof buildGridSVG>;
  private chartSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private xScale!: d3.ScaleLinear<number, number>;
  private yScale!: d3.ScaleLinear<number, number>;
  private fvPath!: d3.Selection<SVGPathElement, unknown, null, undefined>;
  private evPath!: d3.Selection<SVGPathElement, unknown, null, undefined>;
  private fvDot!: d3.Selection<SVGCircleElement, unknown, null, undefined>;
  private evDot!: d3.Selection<SVGCircleElement, unknown, null, undefined>;
  private episodeLabel!: d3.Selection<SVGTextElement, unknown, null, undefined>;
  private animFvVisited = new Set<number>();
  private animEvVisited = new Set<number>();
  private animReturns: number[] = [];

  connectedCallback() {
    const { panel, body } = createPanel({ id: "first-vs-every-visit-walkthrough", heavy: false });
    panel.style.maxWidth = `${W + 20}px`;

    // ── Grid row ──────────────────────────────────────────────────────────────
    const gridRow = document.createElement("div");
    gridRow.style.cssText = `display:flex;gap:24px;align-items:flex-start;padding:8px 12px;`;

    this.fvGrid = buildGridSVG(gridRow, "First-Visit MC");
    this.evGrid = buildGridSVG(gridRow, "Every-Visit MC");

    // Controls
    const ctrl = document.createElement("div");
    ctrl.style.cssText = "display:flex;gap:8px;align-items:center;margin-top:4px;";

    const btnNext = document.createElement("button");
    btnNext.className = "rl-btn";
    btnNext.textContent = "▶ Sample trajectory";
    btnNext.addEventListener("click", () => this.startNextTraj());

    const btnAuto = document.createElement("button");
    btnAuto.className = "rl-btn";
    btnAuto.textContent = "Auto";
    btnAuto.addEventListener("click", () => {
      this.playing = !this.playing;
      btnAuto.textContent = this.playing ? "⏸ Pause" : "Auto";
      if (this.playing) this.autoPlay();
    });

    const btnReset = document.createElement("button");
    btnReset.className = "rl-btn";
    btnReset.textContent = "Reset";
    btnReset.addEventListener("click", () => {
      this.stopAnim();
      this.playing = false;
      btnAuto.textContent = "Auto";
      this.fv = freshState();
      this.ev = freshState();
      this.animTraj = null;
      this.rng = mulberry32(Math.floor(Math.random() * 9999));
      this.updateAllGrids();
      this.updateChart();
    });

    ctrl.append(btnNext, btnAuto, btnReset);

    body.appendChild(gridRow);
    body.appendChild(ctrl);

    // ── Convergence chart ─────────────────────────────────────────────────────
    const chartWrap = document.createElement("div");
    chartWrap.style.cssText = "padding:0 12px 8px;";

    const chartTitle = document.createElement("p");
    chartTitle.style.cssText = "margin:8px 0 4px;font-family:var(--rl-mono);font-size:10px;color:var(--rl-ink-muted);";
    chartTitle.textContent = `V̂(0,0) over last ${HISTORY_LEN} trajectories — reference at ${TRUE_V}`;
    chartWrap.appendChild(chartTitle);

    const CW = W - 24, CH = H_CHART;
    const PW = CW - MARGIN.l - MARGIN.r, PH = CH - MARGIN.t - MARGIN.b;

    const cSvgEl = document.createElementNS(NS_SVG, "svg") as SVGSVGElement;
    cSvgEl.setAttribute("viewBox", `0 0 ${CW} ${CH}`);
    cSvgEl.setAttribute("width", `${CW}`);
    cSvgEl.style.maxWidth = "100%";
    cSvgEl.style.height = "auto";
    chartWrap.appendChild(cSvgEl);

    this.chartSvg = d3.select(cSvgEl);
    const g = this.chartSvg.append("g").attr("transform", `translate(${MARGIN.l},${MARGIN.t})`);

    this.xScale = d3.scaleLinear([0, HISTORY_LEN - 1], [0, PW]);
    this.yScale = d3.scaleLinear([-0.8, 0.2], [PH, 0]);

    // Axes
    g.append("g").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(this.xScale).ticks(5).tickSize(-PH))
      .call(ax => ax.selectAll("line").attr("stroke", "var(--rl-border)").attr("stroke-dasharray", "2,2"))
      .call(ax => ax.select(".domain").attr("stroke", "var(--rl-border)"));
    g.append("g")
      .call(d3.axisLeft(this.yScale).ticks(5).tickSize(-PW))
      .call(ax => ax.selectAll("line").attr("stroke", "var(--rl-border)").attr("stroke-dasharray", "2,2"))
      .call(ax => ax.select(".domain").attr("stroke", "var(--rl-border)"));

    // Reference line
    g.append("line").attr("x1", 0).attr("y1", this.yScale(TRUE_V)).attr("x2", PW).attr("y2", this.yScale(TRUE_V))
      .attr("stroke", "var(--mc-truth)").attr("stroke-width", 1.5).attr("stroke-dasharray", "5,3")
      .attr("opacity", 0.7);
    g.append("text").attr("x", PW - 2).attr("y", this.yScale(TRUE_V) - 4).attr("text-anchor", "end")
      .attr("font-family", "var(--rl-mono)").attr("font-size", 9).attr("fill", "var(--mc-truth)")
      .text(`true V = ${TRUE_V}`);

    // Lines and dots
    this.fvPath = g.append("path").attr("fill", "none").attr("stroke", "var(--mc-first-visit)").attr("stroke-width", 1.5);
    this.evPath = g.append("path").attr("fill", "none").attr("stroke", "var(--mc-every-visit)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,2");
    this.fvDot = g.append("circle").attr("r", 4).attr("fill", "var(--mc-first-visit)").attr("opacity", 0);
    this.evDot = g.append("circle").attr("r", 4).attr("fill", "var(--mc-every-visit)").attr("opacity", 0);

    // Legend
    const legend = g.append("g").attr("transform", `translate(${PW - 160}, 4)`);
    legend.append("line").attr("x1", 0).attr("y1", 6).attr("x2", 18).attr("y2", 6).attr("stroke", "var(--mc-first-visit)").attr("stroke-width", 2);
    legend.append("text").attr("x", 22).attr("y", 10).attr("font-family", "var(--rl-mono)").attr("font-size", 9).attr("fill", "var(--mc-first-visit)").text("first-visit");
    legend.append("line").attr("x1", 0).attr("y1", 22).attr("x2", 18).attr("y2", 22).attr("stroke", "var(--mc-every-visit)").attr("stroke-width", 2).attr("stroke-dasharray", "4,2");
    legend.append("text").attr("x", 22).attr("y", 26).attr("font-family", "var(--rl-mono)").attr("font-size", 9).attr("fill", "var(--mc-every-visit)").text("every-visit");

    this.episodeLabel = this.chartSvg.append("text").attr("x", 8).attr("y", 12)
      .attr("font-family", "var(--rl-mono)").attr("font-size", 9).attr("fill", "var(--rl-ink-muted)");

    body.appendChild(chartWrap);
    this.appendChild(panel);
    this.updateAllGrids();
  }

  private getCellX(s: number) {
    const { c } = rc(s);
    return PAD + c * (CELL + GAP) + CELL / 2;
  }
  private getCellY(s: number) {
    const { r } = rc(s);
    return PAD + r * (CELL + GAP) + CELL / 2;
  }

  private updateGridDisplay(grid: ReturnType<typeof buildGridSVG>, state: MCState, highlightSet?: Set<number>, currentS?: number) {
    for (let s = 0; s < mdp.nS; s++) {
      if (mdp.terminals[s]) {
        grid.cells[s].attr("fill", "var(--rl-surface-2, #ede8df)");
        grid.vLabels[s].text("");
        grid.counts[s].text("");
        continue;
      }
      const v = state.V[s];
      const col = this.colorScale(v);
      grid.cells[s].attr("fill", col).attr("stroke", highlightSet?.has(s) ? "var(--mc-on-policy)" : "var(--rl-border)")
        .attr("stroke-width", highlightSet?.has(s) ? 2.5 : 1);
      grid.vLabels[s].attr("fill", textColorOn(col)).text(state.visits[s] > 0 ? v.toFixed(3) : "");
      grid.counts[s].text(state.visits[s] > 0 ? `n=${state.visits[s]}` : "");
    }
    if (currentS !== undefined && !mdp.terminals[currentS]) {
      grid.agent.attr("cx", this.getCellX(currentS)).attr("cy", this.getCellY(currentS)).attr("opacity", 1);
    } else {
      grid.agent.attr("opacity", 0);
    }
  }

  private updateAllGrids(currentS?: number) {
    this.updateGridDisplay(this.fvGrid, this.fv, this.animFvVisited, currentS);
    this.updateGridDisplay(this.evGrid, this.ev, this.animEvVisited, currentS);
  }

  private updateChart() {
    const fhist = this.fv.history.slice(-HISTORY_LEN);
    const ehist = this.ev.history.slice(-HISTORY_LEN);
    const maxLen = Math.max(fhist.length, ehist.length);
    if (maxLen < 2) return;

    const line = d3.line<number>().x((_, i) => this.xScale(i)).y(d => this.yScale(d));
    if (fhist.length > 1) this.fvPath.attr("d", line(fhist) || "");
    if (ehist.length > 1) this.evPath.attr("d", line(ehist) || "");

    if (fhist.length > 0) {
      this.fvDot.attr("cx", this.xScale(fhist.length - 1)).attr("cy", this.yScale(fhist[fhist.length - 1])).attr("opacity", 1);
    }
    if (ehist.length > 0) {
      this.evDot.attr("cx", this.xScale(ehist.length - 1)).attr("cy", this.yScale(ehist[ehist.length - 1])).attr("opacity", 1);
    }
    this.episodeLabel.text(`ep ${this.fv.episodeCount}`);
  }

  private applyStep(step: number, traj: Step[], returns: number[]) {
    const s = traj[step].s;
    // First-Visit
    if (!this.animFvVisited.has(s)) {
      this.animFvVisited.add(s);
      this.fv.visits[s] += 1;
      this.fv.V[s] += (returns[step] - this.fv.V[s]) / this.fv.visits[s];
    }
    // Every-Visit
    this.animEvVisited.add(s);
    this.ev.visits[s] += 1;
    this.ev.V[s] += (returns[step] - this.ev.V[s]) / this.ev.visits[s];
  }

  private finishEpisode() {
    this.fv.episodeCount += 1;
    this.ev.episodeCount += 1;
    this.fv.history.push(this.fv.V[0]);
    this.ev.history.push(this.ev.V[0]);
    if (this.fv.history.length > HISTORY_LEN) this.fv.history.shift();
    if (this.ev.history.length > HISTORY_LEN) this.ev.history.shift();
    this.updateChart();
  }

  private startNextTraj() {
    this.stopAnim();
    const steps = rollout(mdp, uniform, 0, 200, this.rng);
    if (steps.length === 0) { this.startNextTraj(); return; }
    this.animTraj = steps;
    this.animStep = 0;
    this.animFvVisited = new Set();
    this.animEvVisited = new Set();
    this.animReturns = computeReturns(steps.map(st => st.r), mdp.gamma);
    this.stepAnim();
  }

  private stepAnim() {
    const traj = this.animTraj;
    if (!traj) return;
    if (this.animStep < traj.length) {
      this.applyStep(this.animStep, traj, this.animReturns);
      const currentS = traj[this.animStep].s;
      this.updateAllGrids(currentS);
      this.animStep++;
      const delay = prefersReducedMotion() ? 50 : STEP_MS;
      this.animTimer = setTimeout(() => this.stepAnim(), delay);
    } else {
      this.finishEpisode();
      this.updateAllGrids();
      if (this.playing) {
        this.animTimer = setTimeout(() => this.startNextTraj(), 400);
      }
    }
  }

  private autoPlay() {
    if (this.playing) this.startNextTraj();
  }

  private stopAnim() {
    if (this.animTimer !== null) clearTimeout(this.animTimer);
    this.animTimer = null;
  }

  disconnectedCallback() { this.stopAnim(); }
}
customElements.define("first-vs-every-visit-walkthrough", FirstVsEveryVisitWalkthrough);
