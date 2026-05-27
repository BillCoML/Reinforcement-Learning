/**
 * V6 — TD(λ) Trace Visualizer.
 * Two-panel animation: left=eligibility trace heatmap (amber intensity),
 * right=V-value heatmap. Slider for λ. Plays through trace_snapshots from JSON.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface LambdaRun {
  history: number[];
  trace_snapshots: number[][];
  final_V: number[];
}

interface LambdaData {
  lambda_vals: number[];
  n_episodes: number;
  alpha: number;
  V_true_start: number;
  runs: Record<string, LambdaRun>;
}

async function loadData(): Promise<LambdaData> {
  const res = await fetch("/data/td/td_lambda_traces.json");
  return res.json();
}

const CELL = 64, GAP = 5, GRID = 3;
const GRID_PX = GRID * (CELL + GAP) - GAP;

function traceColor(e: number, maxE: number): string {
  if (maxE === 0) return "#f8fafc";
  const t = Math.min(e / maxE, 1);
  // Amber: low=white, high=amber
  const r = Math.round(255);
  const g = Math.round(255 - t * 120);
  const b = Math.round(255 - t * 220);
  return `rgb(${r},${g},${b})`;
}

function valueColor(v: number): string {
  if (v >= 0) {
    const t = Math.min(v / 0.8, 1);
    return `rgb(${Math.round(235 - t * 80)},255,${Math.round(235 - t * 80)})`;
  }
  const t = Math.min(Math.abs(v) / 0.8, 1);
  return `rgb(255,${Math.round(220 - t * 120)},${Math.round(220 - t * 120)})`;
}

class TDLambdaTraceVisualizer extends HTMLElement {
  private lambdaIdx = 1; // default λ=0.5
  private snapshotIdx = 0;
  private data: LambdaData | null = null;
  private timer: number | null = null;
  private svgEl: SVGSVGElement | null = null;
  private setStatus: ((t: string) => void) | null = null;

  connectedCallback() {
    this.build();
    loadData().then(d => {
      this.data = d;
      this.redraw();
    });
  }

  disconnectedCallback() { this.stop(); }

  private stop() {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "td-lambda-trace-visualizer", heavy: true });
    this.setStatus = setStatus;

    const lambdaOpts = ["0.0 (=TD(0))", "0.5", "1.0 (=MC)"];
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label">λ:</label>
      ${lambdaOpts.map((l, i) => `<button class="rl-btn ${i === 1 ? "is-active" : ""}" data-li="${i}">${l}</button>`).join("")}
      <button class="rl-btn" id="tl-play" style="margin-left:8px">▶ Play</button>
      <button class="rl-btn" id="tl-reset">↺ Reset</button>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const W = 780, H = 320;
    this.svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    this.svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svgEl.style.width = "100%";
    this.svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(this.svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    ctrl.querySelectorAll("button[data-li]").forEach(btn => {
      btn.addEventListener("click", () => {
        this.stop();
        ctrl.querySelectorAll("button[data-li]").forEach(b => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        this.lambdaIdx = +(btn as HTMLButtonElement).dataset.li!;
        this.snapshotIdx = 0;
        (panel.querySelector("#tl-play") as HTMLButtonElement).textContent = "▶ Play";
        this.redraw();
      });
    });

    const playBtn = panel.querySelector("#tl-play") as HTMLButtonElement;
    playBtn.addEventListener("click", () => {
      if (this.timer !== null) {
        this.stop(); playBtn.textContent = "▶ Play";
      } else {
        playBtn.textContent = "⏸ Pause";
        this.timer = window.setInterval(() => {
          if (!this.data) return;
          const run = this.currentRun();
          if (!run) return;
          this.snapshotIdx = (this.snapshotIdx + 1) % run.trace_snapshots.length;
          if (this.snapshotIdx === 0) { this.stop(); playBtn.textContent = "▶ Play"; }
          this.redraw();
        }, 500);
      }
    });

    panel.querySelector("#tl-reset")!.addEventListener("click", () => {
      this.stop(); this.snapshotIdx = 0; playBtn.textContent = "▶ Play"; this.redraw();
    });

    setStatus("loading…");
    this.redraw();
  }

  private currentRun(): LambdaRun | null {
    if (!this.data) return null;
    const keys = ["0.0", "0.5", "1.0"];
    return this.data.runs[keys[this.lambdaIdx]] ?? null;
  }

  private redraw() {
    if (!this.svgEl) return;
    const svg = d3.select(this.svgEl);
    svg.selectAll("*").remove();
    const W = 780, H = 320;
    this.draw(svg, W, H);
  }

  private draw(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, W: number, H: number) {
    const run = this.currentRun();
    const lambdaLabels = ["0.0 — TD(0)", "0.5", "1.0 — MC"];
    const lambdaLabel = lambdaLabels[this.lambdaIdx];
    const halfW = W / 2;
    const PX = 24, PY = 28;

    // Divider
    svg.append("line").attr("x1", halfW).attr("x2", halfW).attr("y1", 0).attr("y2", H)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    // Panel labels
    svg.append("text").attr("x", halfW / 2).attr("y", 16)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "var(--td-trace)").text(`Eligibility traces e(s) — λ=${lambdaLabel}`);
    svg.append("text").attr("x", halfW + halfW / 2).attr("y", 16)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "var(--td-lambda)").text("V̂(s) after all episodes");

    if (!run) {
      svg.append("text").attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", "#94a3b8").text("loading…");
      return;
    }

    const traces = run.trace_snapshots;
    const snap = traces[this.snapshotIdx] ?? new Array(9).fill(0);
    const finalV = run.final_V;

    if (this.setStatus) {
      this.setStatus!(`snapshot ${this.snapshotIdx + 1}/${traces.length}, λ=${lambdaLabel}`);
    }

    const maxE = Math.max(...snap, 0.01);
    const gridOX = PX + (halfW - PX * 2 - GRID_PX) / 2;
    const gridOY = PY;

    // Left panel: trace heatmap
    for (let s = 0; s < 9; s++) {
      const row = Math.floor(s / 3), col = s % 3;
      const cx = gridOX + col * (CELL + GAP), cy = gridOY + row * (CELL + GAP);
      const e = snap[s];
      svg.append("rect").attr("x", cx).attr("y", cy).attr("width", CELL).attr("height", CELL)
        .attr("fill", traceColor(e, maxE)).attr("rx", 4)
        .attr("stroke", "#e2e8f0").attr("stroke-width", 1);
      svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 - 4)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#475569").text(`s${s}`);
      svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 10)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
        .attr("fill", "#1e293b").text(e.toFixed(3));
    }

    // Amber color scale legend
    const legX = gridOX, legY = gridOY + GRID_PX + 12;
    for (let i = 0; i < 30; i++) {
      const e = (i / 29) * maxE;
      svg.append("rect").attr("x", legX + i * 4).attr("y", legY).attr("width", 4).attr("height", 10)
        .attr("fill", traceColor(e, maxE));
    }
    svg.append("text").attr("x", legX).attr("y", legY + 20)
      .attr("font-size", 8).attr("fill", "#94a3b8").text("0");
    svg.append("text").attr("x", legX + 120).attr("y", legY + 20)
      .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8")
      .text(maxE.toFixed(2));
    svg.append("text").attr("x", legX + 60).attr("y", legY + 20)
      .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#94a3b8").text("trace value");

    // Right panel: V heatmap
    const vGridOX = halfW + PX + (halfW - PX * 2 - GRID_PX) / 2;
    const vGridOY = PY;

    for (let s = 0; s < 9; s++) {
      const row = Math.floor(s / 3), col = s % 3;
      const cx = vGridOX + col * (CELL + GAP), cy = vGridOY + row * (CELL + GAP);
      const v = finalV[s];
      svg.append("rect").attr("x", cx).attr("y", cy).attr("width", CELL).attr("height", CELL)
        .attr("fill", valueColor(v)).attr("rx", 4).attr("stroke", "#e2e8f0").attr("stroke-width", 1);
      svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 - 4)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#475569").text(`s${s}`);
      svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 10)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
        .attr("fill", "#1e293b").text(v.toFixed(2));
    }

    // V(0,0) convergence note
    const Vs0 = finalV[0];
    svg.append("text").attr("x", vGridOX + GRID_PX / 2).attr("y", vGridOY + GRID_PX + 20)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "var(--td-lambda)")
      .text(`V̂(s₀)=${Vs0.toFixed(3)}, true=${-0.4205}`);

    // History curve (small, bottom of right panel)
    const histY = vGridOY + GRID_PX + 36;
    const histH = H - histY - 8;
    const histW = halfW - PX * 2;
    const histX = halfW + PX;

    if (run.history.length > 1 && histH > 20) {
      const hist = run.history;
      const xS = d3.scaleLinear([0, hist.length - 1], [histX, histX + histW]);
      const yS = d3.scaleLinear(
        [Math.min(...hist) * 1.05, Math.max(...hist, 0.01) * 1.05],
        [histY + histH, histY],
      );
      svg.append("line").attr("x1", histX).attr("x2", histX + histW)
        .attr("y1", yS(-0.4205)).attr("y2", yS(-0.4205))
        .attr("stroke", "#94a3b8").attr("stroke-width", 1).attr("stroke-dasharray", "2,2");
      const stride = Math.max(1, Math.floor(hist.length / 100));
      const pts = Array.from({ length: Math.ceil(hist.length / stride) }, (_, i) => hist[i * stride]);
      svg.append("path").datum(pts)
        .attr("d", d3.line<number>().x((_, i) => xS(i * stride)).y(d => yS(d)).curve(d3.curveMonotoneX))
        .attr("fill", "none").attr("stroke", "var(--td-lambda)").attr("stroke-width", 1.2);
      svg.append("text").attr("x", histX + histW / 2).attr("y", histY - 2)
        .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#94a3b8").text("V̂(s₀) per episode");
    }
  }
}

customElements.define("td-lambda-trace-visualizer", TDLambdaTraceVisualizer);
