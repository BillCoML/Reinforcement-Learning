/**
 * V2 — TD(0) Explorer.
 * Four-panel view: A=gridworld V̂ heatmap, B=V(0,0) convergence trace,
 * C=single-update animation, D=step-size sensitivity bar chart.
 * Reads pre-computed JSON; no live algorithm runs.
 * Width 920, height 540.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { tdZero } from "../td/td-zero";
import { mulberry32 } from "../importance-sampling/gaussian";

const CELL = 60, GAP = 4, GRID = 3;
const GAMMA = 0.9;
const TRUE_V_START = -0.4205;

const mdp = buildGridworld({ slippery: false, gamma: GAMMA });
const uniform = uniformPolicy(mdp);

// Color scale for heatmap: negative (red) → zero (white) → positive (green)
function valueColor(v: number): string {
  if (v >= 0) {
    const t = Math.min(v / 1, 1);
    const g = Math.round(80 + t * 110);
    return `rgb(${Math.round(235 - t * 80)},${g + 60},${Math.round(235 - t * 80)})`;
  } else {
    const t = Math.min(Math.abs(v) / 1, 1);
    return `rgb(${Math.round(220 + t * 35)},${Math.round(220 - t * 120)},${Math.round(220 - t * 120)})`;
  }
}

function runTDZero(nEpisodes: number, alpha: number, seed: number): Float64Array {
  const { V } = tdZero(mdp, uniform, nEpisodes, alpha, { rng: mulberry32(seed) });
  return V;
}

function runHistory(nEpisodes: number, alpha: number, seed: number): Float64Array {
  const { history } = tdZero(mdp, uniform, nEpisodes, alpha, { rng: mulberry32(seed) });
  return history;
}

class TDZeroExplorer extends HTMLElement {
  private alphaIdx = 2; // default α=0.1
  private nEp = 2000;
  private seed = 42;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "td-zero-explorer", heavy: true });

    const alphas = [0.01, 0.05, 0.1, 0.2];
    const nOptions = [100, 500, 2000, 5000];

    // Controls
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label">α:
        <select id="tz-alpha" class="rl-select">
          ${alphas.map((a, i) => `<option value="${i}" ${i === this.alphaIdx ? "selected" : ""}>${a}</option>`).join("")}
        </select>
      </label>
      <label class="rl-label">Episodes:
        <select id="tz-nep" class="rl-select">
          ${nOptions.map((n, i) => `<option value="${i}" ${i === 2 ? "selected" : ""}>${n}</option>`).join("")}
        </select>
      </label>
      <button class="rl-btn" id="tz-run">▶ Run</button>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const W = 920, H = 440;
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);

    const render = () => {
      const alpha = alphas[this.alphaIdx];
      const V = runTDZero(this.nEp, alpha, this.seed);
      const history = runHistory(this.nEp, alpha, this.seed);
      setStatus(`α=${alpha}, N=${this.nEp}`);
      this.draw(svg, V, history, alpha);
    };

    panel.querySelector("#tz-alpha")!.addEventListener("change", (e) => {
      this.alphaIdx = +(e.target as HTMLSelectElement).value;
    });
    panel.querySelector("#tz-nep")!.addEventListener("change", (e) => {
      const idx = +(e.target as HTMLSelectElement).value;
      this.nEp = nOptions[idx];
    });
    panel.querySelector("#tz-run")!.addEventListener("click", render);

    render();
  }

  private draw(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    V: Float64Array,
    history: Float64Array,
    alpha: number,
  ) {
    svg.selectAll("*").remove();

    // Panel A: Gridworld heatmap (top-left)
    const AX = 20, AY = 30;
    const gridSize = GRID * (CELL + GAP) - GAP;

    svg.append("text").attr("x", AX + gridSize / 2).attr("y", AY - 8)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "var(--td-td0)").text("V̂(s) — TD(0) estimate");

    for (let s = 0; s < 9; s++) {
      const row = Math.floor(s / 3), col = s % 3;
      const cx = AX + col * (CELL + GAP);
      const cy = AY + row * (CELL + GAP);
      const isTerminal = mdp.terminals[s];
      const color = isTerminal ? (V[s] >= 0 ? "#bbf7d0" : "#fca5a5") : valueColor(V[s]);
      svg.append("rect").attr("x", cx).attr("y", cy).attr("width", CELL).attr("height", CELL)
        .attr("fill", color).attr("rx", 4).attr("stroke", "#e2e8f0").attr("stroke-width", 1);
      svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 - 4)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#334155")
        .text(`s${s}`);
      svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 10)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
        .attr("fill", "#1e293b")
        .text(isTerminal ? (s === 4 ? "pit" : "goal") : V[s].toFixed(2));
    }

    // Color legend
    const legX = AX + gridSize + 12, legY = AY;
    svg.append("text").attr("x", legX).attr("y", legY - 4)
      .attr("font-size", 9).attr("fill", "#94a3b8").text("neg");
    svg.append("text").attr("x", legX + 10).attr("y", legY - 4)
      .attr("font-size", 9).attr("fill", "#94a3b8").attr("text-anchor", "end").text("pos");
    for (let i = 0; i < 20; i++) {
      const v = -0.8 + (i / 19) * 1.6;
      svg.append("rect").attr("x", legX + (i / 19) * 60 - 30 + 20).attr("y", legY)
        .attr("width", 3).attr("height", 12).attr("fill", valueColor(v));
    }

    // Panel B: V(0,0) convergence trace (top-right)
    const BX = 260, BY = 30, BW = 300, BH = 200;

    svg.append("text").attr("x", BX + BW / 2).attr("y", BY - 8)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "var(--td-td0)").text("V̂(s₀) convergence");

    const marginB = { left: 40, right: 10, top: 10, bottom: 30 };
    const iW = BW - marginB.left - marginB.right;
    const iH = BH - marginB.top - marginB.bottom;

    const xScale = d3.scaleLinear([0, history.length - 1], [BX + marginB.left, BX + marginB.left + iW]);
    const allVals = Array.from(history);
    const yMin = Math.min(TRUE_V_START - 0.1, d3.min(allVals) ?? -1);
    const yMax = Math.max(0.1, d3.max(allVals) ?? 0.2);
    const yScale = d3.scaleLinear([yMin, yMax], [BY + marginB.top + iH, BY + marginB.top]);

    // Axes
    svg.append("line").attr("x1", BX + marginB.left).attr("x2", BX + marginB.left + iW)
      .attr("y1", yScale(0)).attr("y2", yScale(0))
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1).attr("stroke-dasharray", "3,2");

    // True V reference
    svg.append("line").attr("x1", BX + marginB.left).attr("x2", BX + marginB.left + iW)
      .attr("y1", yScale(TRUE_V_START)).attr("y2", yScale(TRUE_V_START))
      .attr("stroke", "#94a3b8").attr("stroke-width", 1).attr("stroke-dasharray", "4,3");
    svg.append("text").attr("x", BX + marginB.left + iW - 2).attr("y", yScale(TRUE_V_START) - 3)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#94a3b8").text("V*(s₀)");

    // Smooth with a stride
    const stride = Math.max(1, Math.floor(history.length / 200));
    const smoothed = Array.from({ length: Math.ceil(history.length / stride) }, (_, i) => {
      const start = i * stride;
      const end = Math.min(start + stride, history.length);
      let sum = 0; for (let j = start; j < end; j++) sum += history[j];
      return sum / (end - start);
    });
    const smoothedX = smoothed.map((_, i) => i * stride);

    svg.append("path")
      .datum(smoothed)
      .attr("d", d3.line<number>().x((_, i) => xScale(smoothedX[i])).y(d => yScale(d)).curve(d3.curveMonotoneX))
      .attr("fill", "none").attr("stroke", "var(--td-td0)").attr("stroke-width", 1.5);

    // X axis label
    svg.append("text").attr("x", BX + marginB.left + iW / 2).attr("y", BY + BH - 4)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8").text("episode");
    svg.append("text").attr("x", BX + marginB.left - 2).attr("y", yScale(TRUE_V_START))
      .attr("font-size", 9).attr("fill", "#64748b").attr("text-anchor", "end")
      .text(TRUE_V_START.toFixed(2));

    // Y ticks
    for (const tick of [yMin, 0, yMax]) {
      svg.append("text").attr("x", BX + marginB.left - 3).attr("y", yScale(tick) + 3)
        .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8")
        .text(tick.toFixed(1));
    }

    // Final V annotation
    const finalV = history[history.length - 1];
    svg.append("circle").attr("cx", xScale(history.length - 1)).attr("cy", yScale(finalV))
      .attr("r", 3).attr("fill", "var(--td-td0)");
    svg.append("text").attr("x", xScale(history.length - 1) - 4).attr("y", yScale(finalV) - 5)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "var(--td-td0)")
      .text(finalV.toFixed(3));

    // Panel C: Step-size sensitivity (bottom, full width)
    const CX = 20, CY = 270, CW = 860, CH = 150;

    svg.append("text").attr("x", CX + CW / 2).attr("y", CY - 8)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "#1e293b").text("Step-size sensitivity: final RMSE vs α (20 trials, N=2000)");

    const alphaVals = [0.01, 0.05, 0.1, 0.2];
    // Pre-compute RMSE over 20 seeds
    const rmseByAlpha = alphaVals.map(a => {
      const errs = Array.from({ length: 20 }, (_, t) => {
        const { V: Vt } = tdZero(mdp, uniform, 2000, a, { rng: mulberry32(t) });
        return Vt[0] - TRUE_V_START;
      });
      return Math.sqrt(errs.map(e => e * e).reduce((a, b) => a + b) / errs.length);
    });

    const marginC = { left: 50, right: 20, top: 15, bottom: 35 };
    const cW = CW - marginC.left - marginC.right;
    const cH = CH - marginC.top - marginC.bottom;

    const xC = d3.scaleBand(alphaVals.map(String), [CX + marginC.left, CX + marginC.left + cW]).padding(0.3);
    const yC = d3.scaleLinear([0, Math.max(...rmseByAlpha) * 1.15], [CY + marginC.top + cH, CY + marginC.top]);

    // Y axis
    svg.append("line").attr("x1", CX + marginC.left).attr("x2", CX + marginC.left)
      .attr("y1", CY + marginC.top).attr("y2", CY + marginC.top + cH)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    // Grid lines
    for (const tick of yC.ticks(4)) {
      svg.append("line")
        .attr("x1", CX + marginC.left).attr("x2", CX + marginC.left + cW)
        .attr("y1", yC(tick)).attr("y2", yC(tick))
        .attr("stroke", "#f1f5f9").attr("stroke-width", 1);
      svg.append("text").attr("x", CX + marginC.left - 4).attr("y", yC(tick) + 3)
        .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#94a3b8")
        .text(tick.toFixed(3));
    }

    // Bars
    alphaVals.forEach((a, i) => {
      const isActive = Math.abs(a - [0.01, 0.05, 0.1, 0.2][this.alphaIdx]) < 0.001;
      svg.append("rect")
        .attr("x", xC(String(a))!)
        .attr("y", yC(rmseByAlpha[i]))
        .attr("width", xC.bandwidth())
        .attr("height", yC(0) - yC(rmseByAlpha[i]))
        .attr("fill", isActive ? "var(--td-td0)" : "#94a3b8")
        .attr("rx", 2);
      svg.append("text")
        .attr("x", xC(String(a))! + xC.bandwidth() / 2)
        .attr("y", yC(rmseByAlpha[i]) - 3)
        .attr("text-anchor", "middle").attr("font-size", 9)
        .attr("fill", isActive ? "var(--td-td0)" : "#64748b")
        .text(rmseByAlpha[i].toFixed(3));
      svg.append("text")
        .attr("x", xC(String(a))! + xC.bandwidth() / 2)
        .attr("y", CY + marginC.top + cH + 14)
        .attr("text-anchor", "middle").attr("font-size", 10)
        .attr("fill", isActive ? "var(--td-td0)" : "#475569")
        .attr("font-weight", isActive ? "600" : "400")
        .text(`α=${a}`);
    });

    svg.append("text").attr("x", CX + marginC.left + cW / 2).attr("y", CY + CH - 2)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8")
      .text("Smaller α → less variance but slower. Larger α → faster but noisier.");

    // Panel D: Single-step update illustration (top-right, below B)
    const DX = 600, DY = 30, DW = 300;
    svg.append("text").attr("x", DX + DW / 2).attr("y", DY - 8)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "#1e293b").text("One TD update");

    // Pick a representative step from a trajectory
    const s0 = 0, r0 = -1, sp0 = 3;
    const Vs0 = V[s0], Vsp0 = V[sp0];
    const target = r0 + GAMMA * Vsp0;
    const delta = target - Vs0;
    const newV = Vs0 + alpha * delta;

    const rowH = 28;
    const labels: [string, string, string][] = [
      ["V(s₀)  =", Vs0.toFixed(3), "current estimate"],
      ["r      =", r0.toFixed(1), "observed reward"],
      ["γV(s₁) =", (GAMMA * Vsp0).toFixed(3), `${GAMMA}×${Vsp0.toFixed(3)}`],
      ["target =", target.toFixed(3), "r + γV(s₁)"],
      ["δ      =", delta.toFixed(3), "target − V(s₀)"],
      ["αδ     =", (alpha * delta).toFixed(3), `${alpha}×${delta.toFixed(3)}`],
      ["V'(s₀) =", newV.toFixed(3), "updated value"],
    ];
    labels.forEach(([lab, val, note], i) => {
      const y = DY + 14 + i * rowH;
      const isResult = i === 6;
      svg.append("text").attr("x", DX + 8).attr("y", y)
        .attr("font-size", 11).attr("font-family", "var(--rl-font-mono)")
        .attr("fill", isResult ? "var(--td-td0)" : "#475569")
        .attr("font-weight", isResult ? "600" : "400")
        .text(lab);
      svg.append("text").attr("x", DX + 108).attr("y", y)
        .attr("font-size", 11).attr("font-family", "var(--rl-font-mono)")
        .attr("fill", isResult ? "var(--td-td0)" : "#1e293b")
        .attr("font-weight", isResult ? "600" : "400")
        .text(val);
      svg.append("text").attr("x", DX + 160).attr("y", y)
        .attr("font-size", 9).attr("fill", "#94a3b8").text(note);
    });

    // Divider line before result
    svg.append("line").attr("x1", DX + 4).attr("x2", DX + DW - 4)
      .attr("y1", DY + 14 + 5 * rowH + 4).attr("y2", DY + 14 + 5 * rowH + 4)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);
  }
}

customElements.define("td-zero-explorer", TDZeroExplorer);
