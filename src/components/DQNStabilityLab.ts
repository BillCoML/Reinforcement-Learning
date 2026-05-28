/**
 * V6 — DQN Stability Lab (Centerpiece).
 * Six synchronized panels on the 7×7 gridworld:
 *   A) Configuration selector (4 ablations)
 *   B) Live Q-value heatmap (uses pre-computed q_grid from trace JSON)
 *   C) Training loss curve (log scale)
 *   D) Q-value trace for selected state (click a cell)
 *   E) Parameter norm over training
 *   F) Bellman error diagnostic
 *
 * If trace files are not present (training script not yet run), shows a
 * "Training data not available" notice with synthetic preview data.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { loadTrace, qGridAtCheckpoint } from "../dqn/onnx-runtime";
import type { DQNTrace } from "../dqn/onnx-runtime";
import { buildGridworld7x7, idx7, rc7, wallCells7x7 } from "../mdp/gridworld7x7";
import { valueIteration } from "../dp/algorithms";
import { makeValueColorScale } from "./value-scale";

const W = 960;
const H = 920;
const H_PLACEHOLDER = 320; // compact height when training data absent

type Config = "naive" | "target" | "replay" | "full";
const CONFIGS: Config[] = ["naive", "target", "replay", "full"];
const CONFIG_LABELS: Record<Config, string> = {
  naive: "Naive Q-learning",
  target: "+Target network",
  replay: "+Replay buffer",
  full: "Full DQN",
};
const CONFIG_COLORS: Record<Config, string> = {
  naive: "var(--triad-warning)",
  target: "var(--dqn-target)",
  replay: "var(--dqn-replay)",
  full: "var(--dqn-online)",
};

const ENV = "7x7";
const N_STATES = 49;
const N_ACTIONS = 4;
const CELL_PX = 24;
const CELL_GAP = 2;

class DQNStabilityLab extends HTMLElement {
  private activeConfig: Config = "full";
  private currentStep = 0;
  private selectedState = 0; // idx7(0,0)
  private traces: Partial<Record<Config, DQNTrace>> = {};
  private dataAvailable = false;
  private trueQStar: number[][] = []; // [nS][nA]
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null;
  private svgEl: SVGSVGElement | null = null;
  private stepSlider: HTMLInputElement | null = null;
  private sliderRow: HTMLElement | null = null;
  private maxSteps = 800;

  connectedCallback() {
    const { panel, body } = createPanel({ id: "dqn-stability-lab" });
    panel.classList.add("breakout");

    // Config selector
    const configRow = document.createElement("div");
    configRow.className = "rl-controls-row";
    CONFIGS.forEach((cfg) => {
      const btn = document.createElement("button");
      btn.className = `rl-btn${cfg === this.activeConfig ? " active" : ""}`;
      btn.dataset.cfg = cfg;
      btn.textContent = CONFIG_LABELS[cfg];
      btn.style.borderColor = CONFIG_COLORS[cfg];
      btn.addEventListener("click", () => {
        this.activeConfig = cfg;
        configRow.querySelectorAll(".rl-btn").forEach((b) =>
          b.classList.toggle("active", (b as HTMLElement).dataset.cfg === cfg));
        this.redraw();
      });
      configRow.appendChild(btn);
    });
    body.appendChild(configRow);

    // Step scrubber (hidden until data loads)
    const sliderRow = document.createElement("div");
    sliderRow.className = "rl-controls-row";
    sliderRow.innerHTML = `<label class="rl-label">Training step:</label>`;
    sliderRow.style.display = "none";
    this.sliderRow = sliderRow;
    this.stepSlider = document.createElement("input");
    this.stepSlider.type = "range";
    this.stepSlider.min = "0";
    this.stepSlider.max = String(this.maxSteps);
    this.stepSlider.value = "0";
    this.stepSlider.style.width = "300px";
    this.stepSlider.addEventListener("input", () => {
      this.currentStep = parseInt(this.stepSlider!.value);
      this.redrawDynamic();
    });
    sliderRow.appendChild(this.stepSlider);
    const stepLabel = document.createElement("span");
    stepLabel.id = "lab-step-label";
    stepLabel.style.cssText = "margin-left:8px;font-size:11px;color:var(--rl-ink-muted)";
    stepLabel.textContent = "step 0";
    sliderRow.appendChild(stepLabel);
    body.appendChild(sliderRow);

    // SVG — starts compact; expands when data loads
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H_PLACEHOLDER}`);
    svgEl.style.width = "100%"; svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    this.svgEl = svgEl;
    this.svg = d3.select(svgEl);
    this.computeTrueQStar();
    this.loadData().then(() => this.redraw());
    this.drawPlaceholder();
  }

  private computeTrueQStar() {
    const mdp = buildGridworld7x7();
    const { V } = valueIteration(mdp);
    this.trueQStar = Array.from({ length: N_STATES }, (_, s) =>
      Array.from({ length: N_ACTIONS }, (_, a) =>
        mdp.terminals[s] ? 0 :
        mdp.r[s][a] + mdp.gamma * mdp.P[s][a].reduce((acc, p, ns) => acc + p * V[ns], 0),
      ),
    );
  }

  private async loadData() {
    let loaded = false;
    for (const cfg of CONFIGS) {
      try {
        const trace = await loadTrace(cfg, ENV, 0);
        this.traces[cfg] = trace;
        if (trace.checkpointSteps.length > 0) {
          this.maxSteps = Math.max(this.maxSteps, trace.checkpointSteps[trace.checkpointSteps.length - 1]);
        }
        loaded = true;
      } catch {
        // Training data not yet available
      }
    }
    this.dataAvailable = loaded;
    if (this.stepSlider) this.stepSlider.max = String(this.maxSteps);
    if (this.sliderRow) this.sliderRow.style.display = loaded ? "" : "none";
  }

  private drawPlaceholder() {
    if (!this.svg) return;
    this.svg.append("g").attr("class", "placeholder")
      .append("text")
      .attr("x", W / 2).attr("y", H / 2)
      .attr("text-anchor", "middle").attr("font-size", "14px")
      .attr("fill", "var(--rl-ink-muted)")
      .text("Loading DQN Stability Lab…");
  }

  private redraw() {
    if (!this.svg || !this.svgEl) return;
    this.svgEl.setAttribute("viewBox", `0 0 ${W} ${this.dataAvailable ? H : H_PLACEHOLDER}`);
    this.svg.selectAll("*").remove();
    if (this.dataAvailable) this.drawLabels();
    this.redrawDynamic();
  }

  private drawLabels() {
    if (!this.svg) return;
    const labels = [
      { x: 10, y: 14, text: "Panel B — Q-value heatmap" },
      { x: 330, y: 14, text: "Panel C — TD loss (log scale)" },
      { x: 620, y: 14, text: "Panel E — ‖θ‖ parameter norm" },
      { x: 10, y: H / 2 + 14, text: "Panel D — Q-values at selected state (click a cell above)" },
      { x: 620, y: H / 2 + 14, text: "Panel F — Bellman error" },
    ];
    labels.forEach(({ x, y, text }) => {
      this.svg!.append("text").attr("x", x).attr("y", y)
        .attr("class", "rl-label").attr("font-size", "11px").text(text);
    });
  }

  private redrawDynamic() {
    if (!this.svg) return;
    this.svg.selectAll(".dynamic").remove();

    const label = document.getElementById("lab-step-label");
    if (label) label.textContent = `step ${this.currentStep}`;

    const trace = this.traces[this.activeConfig];
    if (!trace || !this.dataAvailable) {
      this.drawSyntheticPanels();
      return;
    }

    this.drawHeatmap(trace);
    this.drawLossCurve(trace);
    this.drawParamNorm(trace);
    this.drawQTrace(trace);
    this.drawBellmanError(trace);
  }

  private drawHeatmap(trace: DQNTrace) {
    const qGrid = qGridAtCheckpoint(trace, this.currentStep);
    if (!qGrid) return;

    const maxQ = qGrid.map((qs) => Math.max(...qs));
    const scale = makeValueColorScale([d3.min(maxQ) ?? -1, d3.max(maxQ) ?? 1]);

    const ox = 10, oy = 24;
    const g = this.svg!.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);

    for (let s = 0; s < N_STATES; s++) {
      const { r, c } = rc7(s);
      const cx = c * (CELL_PX + CELL_GAP);
      const cy = r * (CELL_PX + CELL_GAP);
      const isWall = wallCells7x7.has(s);
      const bg = isWall ? "#374151" : scale(maxQ[s]);
      const rect = g.append("rect").attr("x", cx).attr("y", cy)
        .attr("width", CELL_PX).attr("height", CELL_PX).attr("rx", 2)
        .attr("fill", bg).attr("stroke", s === this.selectedState ? "var(--fa-neural)" : "#e5e7eb")
        .attr("stroke-width", s === this.selectedState ? 2 : 0.5)
        .style("cursor", "pointer");

      rect.on("click", () => {
        if (!isWall) {
          this.selectedState = s;
          this.redrawDynamic();
        }
      });
    }
  }

  private drawLossCurve(trace: DQNTrace) {
    const ox = 330, oy = 24, cw = 280, ch = (H / 2) - 50;
    const g = this.svg!.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);
    const data = trace.lossPerStep;
    if (!data.length) return;

    const dec = Math.max(1, Math.floor(data.length / 200));
    const pts = data.filter((_, i) => i % dec === 0);
    const xScale = d3.scaleLinear().domain([0, data.length]).range([40, cw]);
    const yMax = d3.max(pts.filter(v => v > 0)) ?? 1;
    const yMin = d3.min(pts.filter(v => v > 0)) ?? 0.001;
    const yScale = d3.scaleLog().domain([Math.max(yMin, 1e-4), yMax]).range([ch, 0]).clamp(true);

    g.append("g").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(xScale).ticks(4));
    g.append("g").attr("transform", "translate(40,0)").call(d3.axisLeft(yScale).ticks(3));

    const stepMark = Math.min(this.currentStep, data.length - 1);
    const line = d3.line<number>()
      .x((_, i) => xScale(i * dec))
      .y(d => yScale(Math.max(d, 1e-4)))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(pts.slice(0, Math.ceil(stepMark / dec)))
      .attr("d", line).attr("fill", "none")
      .attr("stroke", CONFIG_COLORS[this.activeConfig]).attr("stroke-width", 1.5);

    // Current step marker
    if (stepMark > 0) {
      const val = data[stepMark];
      g.append("circle").attr("cx", xScale(stepMark)).attr("cy", yScale(Math.max(val, 1e-4))).attr("r", 3)
        .attr("fill", CONFIG_COLORS[this.activeConfig]);
    }
  }

  private drawParamNorm(trace: DQNTrace) {
    const ox = 620, oy = 24, cw = W - ox - 16, ch = (H / 2) - 50;
    const g = this.svg!.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);
    const data = trace.paramNormPerStep;
    if (!data.length) return;

    const dec = Math.max(1, Math.floor(data.length / 200));
    const pts = data.filter((_, i) => i % dec === 0);
    const stepMark = Math.min(this.currentStep, data.length - 1);
    const visiblePts = pts.slice(0, Math.ceil(stepMark / dec));

    const xScale = d3.scaleLinear().domain([0, data.length]).range([40, cw]);
    const yScale = d3.scaleLinear().domain([0, d3.max(pts) ?? 10]).range([ch, 0]);

    g.append("g").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(xScale).ticks(4));
    g.append("g").attr("transform", "translate(40,0)").call(d3.axisLeft(yScale).ticks(4));

    if (visiblePts.length > 1) {
      const line = d3.line<number>().x((_, i) => xScale(i * dec)).y(d => yScale(d)).curve(d3.curveMonotoneX);
      g.append("path").datum(visiblePts).attr("d", line).attr("fill", "none")
        .attr("stroke", CONFIG_COLORS[this.activeConfig]).attr("stroke-width", 2);
    }
  }

  private drawQTrace(trace: DQNTrace) {
    const ox = 10, oy = H / 2 + 24, cw = 590, ch = (H / 2) - 60;
    const g = this.svg!.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);

    if (this.selectedState >= trace.qTracesPerState.length) return;
    const data = trace.qTracesPerState[this.selectedState];
    if (!data?.length) return;

    const dec = Math.max(1, Math.floor(data.length / (4 * N_ACTIONS * 50)));
    const nSteps = Math.floor(data.length / N_ACTIONS);

    const xScale = d3.scaleLinear().domain([0, nSteps]).range([40, cw]);
    const allVals = Array.from({ length: N_ACTIONS }, (_, a) =>
      data.filter((_, i) => i % (dec * N_ACTIONS) === a).filter(v => isFinite(v))
    );
    const allFlat = allVals.flat();
    const yScale = d3.scaleLinear()
      .domain([d3.min(allFlat) ?? -1, d3.max(allFlat) ?? 1])
      .range([ch, 0]);

    g.append("g").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(xScale).ticks(5));
    g.append("g").attr("transform", "translate(40,0)").call(d3.axisLeft(yScale).ticks(4));

    const ACTION_COLORS =["var(--mdp-action-up)", "var(--mdp-action-right)", "var(--mdp-action-down)", "var(--mdp-action-left)"];

    // True Q* reference lines
    if (this.trueQStar[this.selectedState]) {
      this.trueQStar[this.selectedState].forEach((qstar, a) => {
        if (!isFinite(qstar)) return;
        g.append("line").attr("x1", 40).attr("y1", yScale(qstar))
          .attr("x2", cw).attr("y2", yScale(qstar))
          .attr("stroke", ACTION_COLORS[a]).attr("stroke-width", 0.5)
          .attr("stroke-dasharray", "4,4").attr("opacity", 0.5);
      });
    }

    const stepMark = Math.min(this.currentStep, nSteps - 1);
    Array.from({ length: N_ACTIONS }, (_, a) => {
      const pts: number[] = [];
      for (let t = 0; t < nSteps; t++) pts.push(data[t * N_ACTIONS + a]);
      const visiblePts = pts.slice(0, stepMark + 1);
      if (visiblePts.length < 2) return;
      const stepDec = Math.max(1, Math.floor(visiblePts.length / 200));
      const sampled = visiblePts.filter((_, i) => i % stepDec === 0);
      const line = d3.line<number>().x((_, i) => xScale(i * stepDec)).y(d => yScale(d)).curve(d3.curveMonotoneX);
      g.append("path").datum(sampled).attr("d", line).attr("fill", "none")
        .attr("stroke", ACTION_COLORS[a]).attr("stroke-width", 1.5);
    });

    const { r, c } = rc7(this.selectedState);
    g.append("text").attr("x", cw / 2).attr("y", -6)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "var(--rl-ink)")
      .text(`Q-values at state (${r},${c}) over training`);
  }

  private drawBellmanError(trace: DQNTrace) {
    const ox = 620, oy = H / 2 + 24, cw = W - ox - 16, ch = (H / 2) - 60;
    const g = this.svg!.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);
    const data = trace.bellmanErrorPerStep;
    if (!data.length) return;

    const dec = Math.max(1, Math.floor(data.length / 200));
    const pts = data.filter((_, i) => i % dec === 0);
    const stepMark = Math.min(this.currentStep, data.length - 1);
    const visiblePts = pts.slice(0, Math.ceil(stepMark / dec));

    const xScale = d3.scaleLinear().domain([0, data.length]).range([40, cw]);
    const yScale = d3.scaleLinear().domain([0, d3.max(pts) ?? 1]).range([ch, 0]);

    g.append("g").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(xScale).ticks(4));
    g.append("g").attr("transform", "translate(40,0)").call(d3.axisLeft(yScale).ticks(4));

    if (visiblePts.length > 1) {
      const line = d3.line<number>().x((_, i) => xScale(i * dec)).y(d => yScale(d)).curve(d3.curveMonotoneX);
      g.append("path").datum(visiblePts).attr("d", line).attr("fill", "none")
        .attr("stroke", CONFIG_COLORS[this.activeConfig]).attr("stroke-width", 2);
    }
    g.append("text").attr("x", cw / 2).attr("y", -6)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "var(--rl-ink)")
      .text("Bellman error (avg across states/actions)");
  }

  private drawSyntheticPanels() {
    if (!this.svg) return;
    const g = this.svg.append("g").attr("class", "dynamic");

    // Panel B label
    g.append("text").attr("x", 10).attr("y", 14)
      .attr("class", "rl-label").attr("font-size", "11px")
      .text("Panel B — Q* heatmap (true value iteration)");

    // Synthetic heatmap using pre-computed trueQStar
    const ox = 10, oy = 24;
    const maxQ = Array.from({ length: N_STATES }, (_, s) =>
      this.trueQStar[s] ? Math.max(...this.trueQStar[s]) : 0,
    );
    const scale = makeValueColorScale([d3.min(maxQ) ?? -1, d3.max(maxQ) ?? 1]);

    for (let s = 0; s < N_STATES; s++) {
      const { r, c } = rc7(s);
      const cx = ox + c * (CELL_PX + CELL_GAP);
      const cy = oy + r * (CELL_PX + CELL_GAP);
      const isWall = wallCells7x7.has(s);
      const isGoal1 = s === idx7(6, 6), isGoal2 = s === idx7(0, 6);
      const bg = isWall ? "#374151" : isGoal1 ? "#15803d" : isGoal2 ? "#0e7490" : scale(maxQ[s]);
      g.append("rect").attr("x", cx).attr("y", cy)
        .attr("width", CELL_PX).attr("height", CELL_PX).attr("rx", 2)
        .attr("fill", bg).attr("stroke", "#e5e7eb").attr("stroke-width", 0.5);
    }

    // Notice box — positioned beside the heatmap
    const nx = 270, ny = 24, nw = W - nx - 16, nh = 80;
    g.append("rect").attr("x", nx).attr("y", ny).attr("width", nw).attr("height", nh)
      .attr("rx", 6).attr("fill", "none").attr("stroke", "var(--rl-border)").attr("stroke-width", 1.5);
    g.append("text").attr("x", nx + nw / 2).attr("y", ny + 28)
      .attr("text-anchor", "middle").attr("font-size", "13px").attr("font-weight", "600")
      .attr("fill", "var(--rl-ink)").text("Training data not available");
    g.append("text").attr("x", nx + nw / 2).attr("y", ny + 50)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "var(--rl-ink-muted)")
      .text("Run  scripts/train_dqn.py  to generate ONNX checkpoints and trace JSON.");
    g.append("text").attr("x", nx + nw / 2).attr("y", ny + 68)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "var(--rl-ink-muted)")
      .text("Showing true Q* (value iteration) in Panel B as a preview.");
  }
}

customElements.define("dqn-stability-lab", DQNStabilityLab);
