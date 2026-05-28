/**
 * V2 — Linear FA Convergence.
 * Four panels: A) heatmap comparison, B) θ trajectory, C) learning curve, D) feature slider.
 * Interactive: slider changes feature representation (row-col-bias → quadratic → one-hot).
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { linearTDPrediction, featurizeRowColBias, featurizeOneHot, featurizeQuadratic } from "../dqn/linear-fa";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { mulberry32 } from "../importance-sampling/gaussian";
import { makeValueColorScale, textColorOn } from "./value-scale";

const W = 880;
const H = 460;
const CELL = 40;
const GAP = 4;
const PAD = 10;

type FeatureMode = "linear" | "quadratic" | "onehot";

const FEATURE_LABELS: Record<FeatureMode, string> = {
  linear: "φ(s) = (r, c, 1)  — plane",
  quadratic: "φ(s) = (r, c, r², c², rc, 1)  — quadratic",
  onehot: "φ(s) = one-hot  — tabular (exact)",
};

function runFA(mode: FeatureMode, seed: number, nEpisodes: number, alpha: number) {
  const mdp = buildGridworld({ gamma: 0.9 });
  const policy = uniformPolicy(mdp);
  const rng = mulberry32(seed);
  const featurize = mode === "linear"
    ? (s: number) => featurizeRowColBias(3, s)
    : mode === "quadratic"
    ? (s: number) => featurizeQuadratic(3, s)
    : (s: number) => featurizeOneHot(9, s);
  return linearTDPrediction(mdp, policy, featurize, nEpisodes, alpha, { rng, randomStart: true });
}

function computeVApprox(mode: FeatureMode, theta: Float64Array): number[] {
  return Array.from({ length: 9 }, (_, s) => {
    const r = Math.floor(s / 3), c = s % 3;
    if (mode === "linear") return theta[0] * r + theta[1] * c + theta[2];
    if (mode === "quadratic") return theta[0]*r + theta[1]*c + theta[2]*r*r + theta[3]*c*c + theta[4]*r*c + theta[5];
    return theta[s]; // one-hot
  });
}

class LinearFAConvergence extends HTMLElement {
  private mode: FeatureMode = "linear";
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null;
  private trueV: number[] = [];

  connectedCallback() {
    const mdp = buildGridworld({ gamma: 0.9 });
    this.trueV = policyEvaluationExact(mdp, uniformPolicy(mdp));

    const { panel, body } = createPanel({ id: "linear-fa-convergence" });

    // Controls
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label" style="margin-right:8px">Feature representation:</label>
      <select class="rl-select" id="fa-mode-select">
        <option value="linear">${FEATURE_LABELS.linear}</option>
        <option value="quadratic">${FEATURE_LABELS.quadratic}</option>
        <option value="onehot">${FEATURE_LABELS.onehot}</option>
      </select>`;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%"; svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    this.svg = d3.select(svgEl);
    this.redraw();

    panel.querySelector("#fa-mode-select")!.addEventListener("change", (e) => {
      this.mode = (e.target as HTMLSelectElement).value as FeatureMode;
      this.redraw();
    });
  }

  private redraw() {
    if (!this.svg) return;
    this.svg.selectAll("*").remove();
    const { theta, history } = runFA(this.mode, 0, 8000, 0.02);
    this.drawHeatmapComparison(theta);
    this.drawLearningCurve(history, theta);
  }

  private drawHeatmapComparison(theta: Float64Array) {
    if (!this.svg) return;
    const vApprox = computeVApprox(this.mode, theta);
    const scale = makeValueColorScale([-0.7, 0.1]);
    const mdp = buildGridworld();

    // True V panel (left)
    const drawHalf = (
      label: string,
      values: number[],
      ox: number,
      oy: number,
    ) => {
      const g = this.svg!.append("g").attr("transform", `translate(${ox},${oy})`);
      g.append("text").attr("x", (3 * CELL + 2 * GAP) / 2).attr("y", -6)
        .attr("text-anchor", "middle").attr("class", "rl-label").text(label);
      for (let s = 0; s < 9; s++) {
        const r = Math.floor(s / 3), c = s % 3;
        const cx = c * (CELL + GAP);
        const cy = r * (CELL + GAP);
        const bg = mdp.terminals[s] ? (s === 4 ? "#dc2626" : "#15803d") : scale(values[s]);
        g.append("rect").attr("x", cx).attr("y", cy)
          .attr("width", CELL).attr("height", CELL).attr("rx", 3)
          .attr("fill", bg).attr("stroke", "#e5e7eb").attr("stroke-width", 1);
        if (!mdp.terminals[s]) {
          g.append("text")
            .attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 4)
            .attr("text-anchor", "middle").attr("font-size", "10px")
            .attr("fill", textColorOn(bg))
            .text(values[s].toFixed(3));
        }
      }
    };

    const trueLabel = "True V^π";
    const approxLabel = FEATURE_LABELS[this.mode].split("—")[0].trim();
    drawHalf(trueLabel, this.trueV, PAD, 24);
    drawHalf("Approx V_θ (" + approxLabel + ")", vApprox, PAD + 3 * CELL + 3 * GAP + 24, 24);

    // MSE badge
    const mse = this.trueV.reduce((acc, v, s) => {
      if (buildGridworld().terminals[s]) return acc;
      return acc + (v - vApprox[s]) ** 2;
    }, 0) / 7;
    this.svg.append("text")
      .attr("x", W / 2).attr("y", 175)
      .attr("text-anchor", "middle").attr("font-size", "12px")
      .attr("fill", mse < 0.005 ? "var(--dqn-online)" : "var(--rl-ink-muted)")
      .text(`WMSE = ${mse.toFixed(4)}`);
  }

  private drawLearningCurve(history: Float64Array[], _finalTheta: Float64Array) {
    if (!this.svg) return;
    const mdp = buildGridworld();
    const nonTerm = Array.from({ length: 9 }, (_, s) => s).filter(s => !mdp.terminals[s]);

    const mseHistory = history.map(th => {
      return nonTerm.reduce((acc, s) => {
        const vApp = computeVApprox(this.mode, th);
        return acc + (this.trueV[s] - vApp[s]) ** 2;
      }, 0) / nonTerm.length;
    });

    const ox = W / 2 + 20, oy = 20;
    const cw = W - ox - PAD, ch = H - oy - 40;
    const g = this.svg.append("g").attr("transform", `translate(${ox},${oy})`);
    g.append("text").attr("x", cw / 2).attr("y", -4)
      .attr("text-anchor", "middle").attr("class", "rl-label").text("Learning curve (WMSE vs episode)");

    const xScale = d3.scaleLinear().domain([0, mseHistory.length]).range([40, cw]);
    const yMax = Math.min(1, d3.max(mseHistory) ?? 1);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([ch, 0]);

    g.append("g").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(xScale).ticks(5));
    g.append("g").attr("transform", "translate(40,0)").call(d3.axisLeft(yScale).ticks(4));

    const color = this.mode === "onehot" ? "var(--fa-tabular)"
      : this.mode === "quadratic" ? "var(--fa-neural)"
      : "var(--fa-linear)";

    const line = d3.line<number>().x((_, i) => xScale(i)).y(d => yScale(d)).curve(d3.curveMonotoneX);
    // Downsample for display
    const step = Math.max(1, Math.floor(mseHistory.length / 200));
    const sampled = mseHistory.filter((_, i) => i % step === 0);
    g.append("path")
      .datum(sampled)
      .attr("d", line)
      .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2);

    // Final WMSE annotation
    const finalMse = mseHistory[mseHistory.length - 1];
    g.append("text").attr("x", cw - 5).attr("y", yScale(finalMse) - 4)
      .attr("text-anchor", "end").attr("font-size", "10px").attr("fill", color)
      .text(`final: ${finalMse.toFixed(4)}`);
  }
}

customElements.define("dqn-linear-fa-convergence", LinearFAConvergence);
