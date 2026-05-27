/**
 * V7 — Maximization Bias Demo.
 * Single panel: running "left-from-A" fraction chart.
 * Loads pre-computed JSON curves; shows mean curve vs optimal 5% reference.
 * Optionally plays the animation with a seed selector.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface MaxBiasData {
  n_episodes: number;
  epsilon: number;
  alpha: number;
  optimal_left_fraction: number;
  mean_left_fraction: number;
  individual_curves: number[][];
  average_curve: number[];
}

async function loadData(): Promise<MaxBiasData> {
  const res = await fetch("/data/td/max_bias_demo.json");
  return res.json();
}

class MaximizationBiasDemo extends HTMLElement {
  private data: MaxBiasData | null = null;
  private showIndividual = false;
  private svgEl: SVGSVGElement | null = null;

  connectedCallback() {
    this.build();
    loadData().then(d => { this.data = d; this.redraw(); });
  }

  private svgW = 820;
  private svgH = 320;

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "maximization-bias-demo" });

    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <button class="rl-btn" id="mb-toggle">Show individual seeds</button>
      <span style="margin-left:12px;font-size:11px;color:var(--rl-ink-muted)" id="mb-info"></span>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    this.svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    this.svgEl.setAttribute("viewBox", `0 0 ${this.svgW} ${this.svgH}`);
    this.svgEl.style.width = "100%";
    this.svgEl.style.maxWidth = `${this.svgW}px`;
    wrap.appendChild(this.svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    panel.querySelector("#mb-toggle")!.addEventListener("click", () => {
      this.showIndividual = !this.showIndividual;
      (panel.querySelector("#mb-toggle") as HTMLButtonElement).textContent =
        this.showIndividual ? "Show mean only" : "Show individual seeds";
      this.redraw();
    });

    setStatus("loading…");
    this.redraw();
  }

  private redraw() {
    if (!this.svgEl) return;
    const svg = d3.select(this.svgEl);
    svg.selectAll("*").remove();
    this.draw(svg);
  }

  private draw(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const W = this.svgW, H = this.svgH;
    const margin = { left: 52, right: 20, top: 28, bottom: 40 };
    const iW = W - margin.left - margin.right;
    const iH = H - margin.top - margin.bottom;

    svg.append("text").attr("x", margin.left + iW / 2).attr("y", 16)
      .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 600)
      .attr("fill", "#1e293b").text("Left-from-A fraction — Q-learning exhibits maximization bias");

    if (!this.data) {
      svg.append("text").attr("x", margin.left + iW / 2).attr("y", margin.top + iH / 2)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#94a3b8").text("loading…");
      return;
    }

    const avg = this.data.average_curve;
    const opt = this.data.optimal_left_fraction; // 0.05
    const N = this.data.n_episodes;
    const stride = Math.ceil(N / avg.length);

    const xS = d3.scaleLinear([0, (avg.length - 1) * stride], [margin.left, margin.left + iW]);
    const yS = d3.scaleLinear([0, 1], [margin.top + iH, margin.top]);

    // Axes
    svg.append("line").attr("x1", margin.left).attr("x2", margin.left + iW)
      .attr("y1", margin.top + iH).attr("y2", margin.top + iH)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);
    svg.append("line").attr("x1", margin.left).attr("x2", margin.left)
      .attr("y1", margin.top).attr("y2", margin.top + iH)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    // Y grid + labels
    for (const tick of [0, 0.25, 0.5, 0.75, 1.0]) {
      svg.append("line").attr("x1", margin.left).attr("x2", margin.left + iW)
        .attr("y1", yS(tick)).attr("y2", yS(tick))
        .attr("stroke", "#f1f5f9").attr("stroke-width", 1);
      svg.append("text").attr("x", margin.left - 4).attr("y", yS(tick) + 3)
        .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#94a3b8")
        .text(`${Math.round(tick * 100)}%`);
    }

    // Optimal reference line at 5%
    svg.append("line").attr("x1", margin.left).attr("x2", margin.left + iW)
      .attr("y1", yS(opt)).attr("y2", yS(opt))
      .attr("stroke", "#22c55e").attr("stroke-width", 1.5).attr("stroke-dasharray", "5,3");
    svg.append("text").attr("x", margin.left + iW - 4).attr("y", yS(opt) - 3)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#22c55e")
      .text(`optimal ${(opt * 100).toFixed(0)}%`);

    // Individual curves (faint)
    if (this.showIndividual) {
      this.data.individual_curves.forEach(curve => {
        svg.append("path").datum(curve)
          .attr("d", d3.line<number>().x((_, i) => xS(i * stride)).y(d => yS(d)).curve(d3.curveMonotoneX))
          .attr("fill", "none").attr("stroke", "var(--td-qlearning)").attr("stroke-width", 0.5)
          .attr("opacity", 0.25);
      });
    }

    // Mean curve
    svg.append("path").datum(avg)
      .attr("d", d3.line<number>().x((_, i) => xS(i * stride)).y(d => yS(d)).curve(d3.curveMonotoneX))
      .attr("fill", "none").attr("stroke", "var(--td-qlearning)").attr("stroke-width", 2.5);

    // Annotation: mean left fraction
    const meanFrac = this.data.mean_left_fraction;
    const lastPt = avg[avg.length - 1];
    svg.append("circle").attr("cx", xS((avg.length - 1) * stride)).attr("cy", yS(lastPt))
      .attr("r", 4).attr("fill", "var(--td-qlearning)");
    svg.append("text").attr("x", xS((avg.length - 1) * stride) - 6).attr("y", yS(lastPt) - 6)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "var(--td-qlearning)")
      .attr("font-weight", 600).text(`mean=${(meanFrac * 100).toFixed(1)}%`);

    // Bias annotation box
    const midX = margin.left + iW * 0.15;
    const midY = yS(0.3);
    svg.append("rect").attr("x", midX - 4).attr("y", midY - 14).attr("width", 220).attr("height", 48)
      .attr("fill", "#fff7ed").attr("rx", 4).attr("stroke", "var(--td-qlearning)").attr("stroke-width", 1);
    svg.append("text").attr("x", midX + 2).attr("y", midY + 2)
      .attr("font-size", 10).attr("fill", "var(--td-qlearning)").attr("font-weight", 600)
      .text("Maximization bias:");
    svg.append("text").attr("x", midX + 2).attr("y", midY + 16)
      .attr("font-size", 9).attr("fill", "#92400e")
      .text(`Q-learning goes left ≈${(meanFrac * 100).toFixed(0)}% (should be ${(opt * 100).toFixed(0)}%)`);
    svg.append("text").attr("x", midX + 2).attr("y", midY + 28)
      .attr("font-size", 9).attr("fill", "#92400e")
      .text("max of noisy Q > expected max");

    // X axis label
    svg.append("text").attr("x", margin.left + iW / 2).attr("y", H - 6)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8")
      .text("episodes");

    // X ticks
    for (const ep of xS.ticks(5)) {
      svg.append("text").attr("x", xS(ep)).attr("y", margin.top + iH + 14)
        .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#94a3b8")
        .text(ep.toLocaleString());
    }

    // Y axis label
    svg.append("text")
      .attr("x", 12).attr("y", margin.top + iH / 2)
      .attr("transform", `rotate(-90, 12, ${margin.top + iH / 2})`)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8")
      .text("P(left from A)");
  }
}

customElements.define("maximization-bias-demo", MaximizationBiasDemo);
