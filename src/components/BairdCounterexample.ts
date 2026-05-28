/**
 * V3 — Baird's Counterexample.
 * Four panels:
 *   A) 7-state MDP schematic
 *   B) θ bar chart updating step by step
 *   C) log‖θ‖ vs iteration (linear-y → straight line = exponential divergence)
 *   D) deadly-triad selector (three checkboxes)
 * Centerpiece breakout: width 960, height 540.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { bairdCounterexample } from "../dqn/baird";
import { mulberry32 } from "../importance-sampling/gaussian";

const W = 960;
const H = 540;
const MAX_STEPS = 2000;
const STEP_DISPLAY_INC = 10;

type TriadConfig = { offPolicy: boolean; bootstrap: boolean };

function runBaird(cfg: TriadConfig, seed: number) {
  return bairdCounterexample(MAX_STEPS, {
    alpha: 0.01,
    onPolicy: !cfg.offPolicy,
    bootstrap: cfg.bootstrap,
    rng: mulberry32(seed),
  });
}

class BairdCounterexample extends HTMLElement {
  private useBootstrap = true;
  private useOffPolicy = true;
  private currentStep = 0;
  private normHistory: number[] = [];
  private normHistoryOnPol: number[] = [];
  private normHistoryMC: number[] = [];
  private thetaHistory: Float64Array[] = [];
  private svgEl: SVGSVGElement | null = null;
  private stepTimer: number | null = null;

  connectedCallback() {
    const { panel, body } = createPanel({ id: "baird-counterexample" });
    panel.classList.add("breakout");

    // Controls row
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <span class="rl-label" style="margin-right:12px">Triad:</span>
      <label style="margin-right:12px"><input type="checkbox" id="cb-fa" checked disabled> Function approx.</label>
      <label style="margin-right:12px"><input type="checkbox" id="cb-bootstrap" checked> Bootstrapping</label>
      <label style="margin-right:12px"><input type="checkbox" id="cb-offpolicy" checked> Off-policy</label>
      <button class="rl-btn" id="baird-run" style="margin-left:16px">Run</button>
      <button class="rl-btn" id="baird-reset" style="margin-left:6px">Reset</button>
      <span id="baird-step-label" style="margin-left:16px;font-size:11px;color:var(--rl-ink-muted)"></span>`;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    this.svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    this.svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svgEl.style.width = "100%"; this.svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(this.svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    this.computeData();
    this.drawStatic();
    this.updateDynamic();

    panel.querySelector("#cb-bootstrap")!.addEventListener("change", () => {
      this.useBootstrap = (panel.querySelector("#cb-bootstrap") as HTMLInputElement).checked;
      this.resetRun();
    });
    panel.querySelector("#cb-offpolicy")!.addEventListener("change", () => {
      this.useOffPolicy = (panel.querySelector("#cb-offpolicy") as HTMLInputElement).checked;
      this.resetRun();
    });
    panel.querySelector("#baird-run")!.addEventListener("click", () => this.startAnimation());
    panel.querySelector("#baird-reset")!.addEventListener("click", () => this.resetRun());
  }

  disconnectedCallback() {
    if (this.stepTimer !== null) clearInterval(this.stepTimer);
  }

  private computeData() {
    const full = runBaird({ offPolicy: true, bootstrap: true }, 0);
    this.normHistory = full.normHistory;
    this.thetaHistory = full.thetaHistory;
    const onPol = runBaird({ offPolicy: false, bootstrap: true }, 0);
    this.normHistoryOnPol = onPol.normHistory;
    const mc = runBaird({ offPolicy: true, bootstrap: false }, 0);
    this.normHistoryMC = mc.normHistory;
  }

  private resetRun() {
    if (this.stepTimer !== null) { clearInterval(this.stepTimer); this.stepTimer = null; }
    this.currentStep = 0;
    this.computeDataForCurrent();
    this.updateDynamic();
  }

  private computeDataForCurrent() {
    const cfg: TriadConfig = { offPolicy: this.useOffPolicy, bootstrap: this.useBootstrap };
    const result = runBaird(cfg, 0);
    this.normHistory = result.normHistory;
    this.thetaHistory = result.thetaHistory;
    const onPol = runBaird({ offPolicy: false, bootstrap: true }, 0);
    this.normHistoryOnPol = onPol.normHistory;
    const mc = runBaird({ offPolicy: true, bootstrap: false }, 0);
    this.normHistoryMC = mc.normHistory;
  }

  private startAnimation() {
    if (this.stepTimer !== null) return;
    this.stepTimer = window.setInterval(() => {
      this.currentStep = Math.min(this.currentStep + STEP_DISPLAY_INC, MAX_STEPS);
      this.updateDynamic();
      if (this.currentStep >= MAX_STEPS) {
        clearInterval(this.stepTimer!);
        this.stepTimer = null;
      }
    }, 20);
  }

  private drawStatic() {
    if (!this.svgEl) return;
    const svg = d3.select(this.svgEl);
    svg.selectAll("*").remove();

    // Panel A: 7-state MDP schematic
    this.drawMDPSchematic(svg, 16, 16, 200, H - 32);

    // Panel C label
    svg.append("text").attr("x", 380).attr("y", 14)
      .attr("class", "rl-label").text("log ‖θ‖ vs iteration");
    svg.append("text").attr("x", 220).attr("y", 14)
      .attr("class", "rl-label").text("θ bar chart");
  }

  private drawMDPSchematic(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    ox: number, oy: number, w: number, h: number,
  ) {
    const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
    g.append("text").attr("x", w / 2).attr("y", 12)
      .attr("text-anchor", "middle").attr("class", "rl-label").text("Baird's 7-state MDP");

    // Positions for states 0–5 in a circle, state 6 in center
    const cx = w / 2, cy = h / 2 + 10, rx = 72, ry = 60;
    const positions: [number, number][] = Array.from({ length: 6 }, (_, i) => {
      const angle = (i / 6) * 2 * Math.PI - Math.PI / 2;
      return [cx + rx * Math.cos(angle), cy + ry * Math.sin(angle)];
    });
    positions.push([cx, cy]); // state 6

    const R = 14;
    // Draw solid action arrows (→ state 6)
    for (let i = 0; i < 6; i++) {
      const [sx, sy] = positions[i];
      const [tx, ty] = positions[6];
      const dx = tx - sx, dy = ty - sy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const ex = sx + dx / dist * (dist - R - 2);
      const ey = sy + dy / dist * (dist - R - 2);
      g.append("line")
        .attr("x1", sx + dx / dist * (R + 2)).attr("y1", sy + dy / dist * (R + 2))
        .attr("x2", ex).attr("y2", ey)
        .attr("stroke", "var(--dqn-target)").attr("stroke-width", 1.5)
        .attr("marker-end", "url(#arrow-solid)");
    }

    // Self-loop on state 6 (solid → stays)
    g.append("path")
      .attr("d", `M ${cx + R + 2} ${cy} A 12 12 0 1 1 ${cx + 1} ${cy - R - 2}`)
      .attr("fill", "none").attr("stroke", "var(--dqn-target)").attr("stroke-width", 1.5);

    // Dashed action arrows (just annotate with a curve)
    for (let i = 0; i < 6; i++) {
      const [sx, sy] = positions[i];
      // Just draw a small dashed indicator
      g.append("line").attr("x1", sx).attr("y1", sy)
        .attr("x2", sx + 8).attr("y2", sy + 8)
        .attr("stroke", "var(--fa-linear)").attr("stroke-width", 1)
        .attr("stroke-dasharray", "3,2");
    }

    // Draw state circles
    positions.forEach(([x, y], i) => {
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", R)
        .attr("fill", i === 6 ? "var(--dqn-target)" : "var(--fa-linear)")
        .attr("stroke", "white").attr("stroke-width", 1.5);
      g.append("text").attr("x", x).attr("y", y + 4)
        .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "white")
        .text(`s${i + 1}`);
    });

    // Arrow marker def
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "arrow-solid")
      .attr("viewBox", "0 0 10 10").attr("refX", 5).attr("refY", 5)
      .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
      .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", "var(--dqn-target)");

    // Legend
    g.append("line").attr("x1", 4).attr("y1", h - 32).attr("x2", 20).attr("y2", h - 32)
      .attr("stroke", "var(--dqn-target)").attr("stroke-width", 2);
    g.append("text").attr("x", 24).attr("y", h - 28).attr("font-size", "10px")
      .attr("fill", "var(--rl-ink)").text("solid → s₇ (ρ=2)");
    g.append("line").attr("x1", 4).attr("y1", h - 16).attr("x2", 20).attr("y2", h - 16)
      .attr("stroke", "var(--fa-linear)").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,2");
    g.append("text").attr("x", 24).attr("y", h - 12).attr("font-size", "10px")
      .attr("fill", "var(--rl-ink)").text("dashed → s₁..₆ (ρ=0)");
  }

  private updateDynamic() {
    if (!this.svgEl) return;
    const svg = d3.select(this.svgEl);
    svg.selectAll(".dynamic").remove();

    const step = this.currentStep;
    const stepLabel = this.svgEl.closest(".lesson-panel")?.querySelector("#baird-step-label");
    if (stepLabel) stepLabel.textContent = `step ${step}`;

    // Panel B: theta bar chart
    this.drawThetaBar(svg, 220, 24, 140, H - 40, step);

    // Panel C: norm trace
    this.drawNormTrace(svg, 374, 24, W - 374 - 16, H - 40, step);
  }

  private drawThetaBar(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    ox: number, oy: number, w: number, h: number,
    step: number,
  ) {
    const thetaIdx = Math.floor(step / 100);
    const theta = this.thetaHistory[Math.min(thetaIdx, this.thetaHistory.length - 1)]
      ?? new Float64Array(8).fill(1);

    const norm = Math.sqrt(Array.from(theta).reduce((a, v) => a + v * v, 0));
    const isDiverging = this.useOffPolicy && this.useBootstrap && norm > 50;

    const g = svg.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);

    // Norm badge
    g.append("text").attr("x", w / 2).attr("y", 12)
      .attr("text-anchor", "middle").attr("font-size", "12px").attr("font-weight", "600")
      .attr("fill", isDiverging ? "var(--triad-warning)" : "var(--dqn-online)")
      .text(`‖θ‖ = ${norm.toFixed(1)}`);

    const barH = (h - 30) / 8 - 3;
    const maxAbs = Math.max(1, Math.max(...Array.from(theta).map(Math.abs)));
    const xScale = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([10, w - 4]);

    theta.forEach((v, i) => {
      const y = 22 + i * (barH + 3);
      const fill = isDiverging ? "var(--triad-warning)" : "var(--fa-linear)";
      g.append("rect").attr("x", xScale(Math.min(v, 0))).attr("y", y)
        .attr("width", Math.abs(xScale(v) - xScale(0))).attr("height", barH)
        .attr("rx", 2).attr("fill", fill).attr("opacity", 0.8);
      g.append("text").attr("x", 6).attr("y", y + barH / 2 + 4)
        .attr("font-size", "9px").attr("fill", "var(--rl-ink-muted)")
        .text(`θ${i + 1}`);
      g.append("text").attr("x", w - 2).attr("y", y + barH / 2 + 4)
        .attr("text-anchor", "end").attr("font-size", "9px").attr("fill", "var(--rl-ink)")
        .text(v.toFixed(1));
    });
  }

  private drawNormTrace(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    ox: number, oy: number, w: number, h: number,
    step: number,
  ) {
    const g = svg.append("g").attr("class", "dynamic").attr("transform", `translate(${ox},${oy})`);

    const fullTrace = this.normHistory.slice(0, step + 1);
    const onPolTrace = this.normHistoryOnPol.slice(0, step + 1);
    const mcTrace = this.normHistoryMC.slice(0, step + 1);

    const maxNorm = Math.max(10, d3.max(fullTrace) ?? 10, d3.max(onPolTrace) ?? 10, d3.max(mcTrace) ?? 10);
    const xScale = d3.scaleLinear().domain([0, MAX_STEPS]).range([44, w]);
    const yScale = d3.scaleLinear().domain([0, maxNorm]).range([h - 24, 0]);

    g.append("g").attr("transform", `translate(0,${h - 24})`).call(d3.axisBottom(xScale).ticks(5));
    g.append("g").attr("transform", "translate(44,0)").call(d3.axisLeft(yScale).ticks(4)
      .tickFormat(d => (+d >= 1000 ? `${+(+d / 1000).toFixed(1)}k` : `${+d}`)));

    g.append("text").attr("x", w / 2).attr("y", h + 12).attr("text-anchor", "middle")
      .attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)").text("iteration");
    g.append("text").attr("x", -h / 2).attr("y", -8)
      .attr("transform", "rotate(-90)").attr("text-anchor", "middle")
      .attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)").text("‖θ‖");

    const isDiverging = this.useOffPolicy && this.useBootstrap;

    const drawTrace = (data: number[], color: string) => {
      if (data.length < 2) return;
      const dec = Math.max(1, Math.floor(data.length / 300));
      const pts = data.filter((_, i) => i % dec === 0);
      const line = d3.line<number>().x((_, i) => xScale(i * dec)).y(d => yScale(d));
      g.append("path").datum(pts).attr("d", line)
        .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("opacity", 0.85);
    };

    drawTrace(onPolTrace, "var(--dqn-online)");   // on-policy: green (bounded)
    drawTrace(mcTrace, "var(--fa-linear)");         // MC: cyan (converges)
    drawTrace(fullTrace, isDiverging ? "var(--triad-warning)" : "var(--dqn-online)");

    // Legend
    const lx = w - 130, ly = 4;
    [
      { color: isDiverging ? "var(--triad-warning)" : "var(--dqn-online)", label: "off-policy TD (full triad)" },
      { color: "var(--dqn-online)", label: "on-policy TD" },
      { color: "var(--fa-linear)", label: "MC (no bootstrap)" },
    ].forEach(({ color, label }, i) => {
      g.append("line").attr("x1", lx).attr("y1", ly + i * 16).attr("x2", lx + 16).attr("y2", ly + i * 16)
        .attr("stroke", color).attr("stroke-width", 2);
      g.append("text").attr("x", lx + 20).attr("y", ly + i * 16 + 4)
        .attr("font-size", "9px").attr("fill", "var(--rl-ink)").text(label);
    });
  }
}

customElements.define("dqn-baird-counterexample", BairdCounterexample);
