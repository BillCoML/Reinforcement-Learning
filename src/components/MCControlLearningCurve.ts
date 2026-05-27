/**
 * V4 — MC Control Learning Curves.
 * Compares three algorithms: MC Exploring Starts, MC ε=0.1, MC GLIE.
 * Shows max Q(s₀, ·) ≈ V(s₀) over 50 000 episodes, 5 seeds each.
 * Toggle: all-seeds view vs mean ± SD band.
 * Data loaded from /data/mc/control_learning_curves.json (pre-computed).
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface CurveData {
  episodeInterval: number;
  nEpisodes: number;
  refOptimal: number;
  refEpsSoft: number;
  mcES: number[][];
  mcEps01: number[][];
  mcGlie: number[][];
}

const W = 660, H = 340;
const M = { top: 24, right: 120, bottom: 40, left: 50 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;

const ALGO_COLOR = {
  es:    "var(--mc-on-policy)",   // green
  eps01: "var(--mc-off-policy)",  // orange
  glie:  "var(--mc-running-mean)", // purple
};

const ALGO_LABEL = {
  es:    "MC Exploring Starts",
  eps01: "MC ε = 0.1",
  glie:  "MC GLIE  ε = 1/√n",
};

// mean and SD over seeds per x-point
function stats(curves: number[][]): { means: number[]; sds: number[] } {
  const T = curves[0].length;
  const means = new Array(T).fill(0);
  const sds   = new Array(T).fill(0);
  for (let t = 0; t < T; t++) {
    const vs = curves.map(c => c[t]);
    const m = vs.reduce((a, b) => a + b, 0) / vs.length;
    const sd = Math.sqrt(vs.reduce((a, v) => a + (v - m) ** 2, 0) / vs.length);
    means[t] = m;
    sds[t] = sd;
  }
  return { means, sds };
}

class MCControlLearningCurve extends HTMLElement {
  private showBands = false;

  connectedCallback() { this.build(); }

  private async build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({
      id: "mc-control-curves",
      heavy: true,
      mobileNotice: "Best viewed on a wider screen.",
    });

    // Controls
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label">
        <input type="checkbox" id="mc-bands-check"> Show mean ± SD bands
      </label>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.style.height = "auto";
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    body.appendChild(wrap);

    this.appendChild(panel);

    // Load data
    let data: CurveData;
    try {
      const resp = await fetch("/data/mc/control_learning_curves.json");
      data = await resp.json();
    } catch {
      d3.select(svgEl).append("text").attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle").attr("font-size", 11)
        .attr("fill", "var(--rl-ink-muted)")
        .text("Could not load data — start the dev server.");
      setStatus("data load failed");
      return;
    }

    const bandsCheck = ctrl.querySelector<HTMLInputElement>("#mc-bands-check")!;
    bandsCheck.addEventListener("change", () => {
      this.showBands = bandsCheck.checked;
      this.draw(svgEl, data);
    });

    this.draw(svgEl, data);
    setStatus(`${data.nEpisodes.toLocaleString()} ep · interval ${data.episodeInterval} · 5 seeds`);
  }

  private draw(svgEl: SVGSVGElement, data: CurveData) {
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);
    const interval = data.episodeInterval;
    const T = data.mcES[0].length;
    const xDomain = (T - 1) * interval;

    const xScale = d3.scaleLinear().domain([0, xDomain]).range([0, PW]);
    const yScale = d3.scaleLinear().domain([0, data.refOptimal * 1.07]).range([PH, 0]).nice();

    // Grid lines
    g.append("g").attr("class", "rl-grid")
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-PW).tickFormat(() => ""))
      .call(g2 => g2.selectAll(".tick line").attr("stroke", "var(--rl-border)").attr("opacity", 0.4))
      .call(g2 => g2.select(".domain").remove());

    // Axes
    g.append("g").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xScale).ticks(6)
        .tickFormat((d) => `${(+d / 1000).toFixed(0)}k`))
      .selectAll("text").style("font-size", "9px");
    g.append("g").call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    // Axis labels
    g.append("text").attr("x", PW / 2).attr("y", PH + 32)
      .attr("text-anchor", "middle").attr("font-size", 9.5)
      .attr("fill", "var(--rl-ink-muted)").text("episodes");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -PH / 2).attr("y", -38)
      .attr("text-anchor", "middle").attr("font-size", 9.5)
      .attr("fill", "var(--rl-ink-muted)").text("max Q(s₀,·)");

    // Reference lines
    const refs = [
      { v: data.refOptimal, label: `V*(0,0) = ${data.refOptimal}`, dash: "5,3", color: "var(--mc-on-policy)" },
      { v: data.refEpsSoft, label: `Q^ε(0,0) = ${data.refEpsSoft}`, dash: "3,3", color: "var(--mc-off-policy)" },
    ];
    for (const ref of refs) {
      g.append("line")
        .attr("x1", 0).attr("x2", PW)
        .attr("y1", yScale(ref.v)).attr("y2", yScale(ref.v))
        .attr("stroke", ref.color).attr("stroke-width", 1.1)
        .attr("stroke-dasharray", ref.dash).attr("opacity", 0.55);
      g.append("text")
        .attr("x", PW + 3).attr("y", yScale(ref.v) + 4)
        .attr("font-size", 8.5).attr("fill", ref.color).attr("opacity", 0.7)
        .text(ref.label);
    }

    // Draw curves
    const algos: Array<{ key: "es" | "eps01" | "glie"; curves: number[][] }> = [
      { key: "es",    curves: data.mcES },
      { key: "eps01", curves: data.mcEps01 },
      { key: "glie",  curves: data.mcGlie },
    ];

    for (const { key, curves } of algos) {
      const color = ALGO_COLOR[key];

      if (this.showBands) {
        const { means, sds } = stats(curves);
        const area = d3.area<number>()
          .x((_, i) => xScale(i * interval))
          .y0((_, i) => yScale(means[i] - sds[i]))
          .y1((_, i) => yScale(means[i] + sds[i]))
          .curve(d3.curveMonotoneX);
        g.append("path").datum(means)
          .attr("d", area)
          .attr("fill", color).attr("opacity", 0.12);
        const line = d3.line<number>()
          .x((_, i) => xScale(i * interval)).y(yScale)
          .curve(d3.curveMonotoneX);
        g.append("path").datum(means)
          .attr("d", line)
          .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2.2);
      } else {
        const line = d3.line<number>()
          .x((_, i) => xScale(i * interval)).y(yScale)
          .curve(d3.curveMonotoneX);
        for (let seed = 0; seed < curves.length; seed++) {
          g.append("path").datum(curves[seed])
            .attr("d", line)
            .attr("fill", "none")
            .attr("stroke", color)
            .attr("stroke-width", seed === 0 ? 1.8 : 1.1)
            .attr("opacity", seed === 0 ? 0.85 : 0.45);
        }
      }
    }

    // Legend
    const legendY = 4;
    for (const { key } of algos) {
      const lx = M.left + PW + 4;
      const ly = legendY + algos.indexOf(algos.find(a => a.key === key)!) * 36;
      const lg = svg.append("g").attr("transform", `translate(${lx},${M.top + ly})`);
      lg.append("line").attr("x1", 0).attr("y1", 6).attr("x2", 18).attr("y2", 6)
        .attr("stroke", ALGO_COLOR[key]).attr("stroke-width", 2.2);
      lg.append("text").attr("x", 21).attr("y", 10)
        .attr("font-size", 9).attr("fill", "var(--rl-ink)")
        .text(ALGO_LABEL[key]);
    }
  }
}

customElements.define("mc-control-learning-curve", MCControlLearningCurve);
