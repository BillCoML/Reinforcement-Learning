/**
 * Panel D — KL divergence trace over training.
 * Pink trace + dashed TRPO budget line at δ=0.01.
 */
import * as d3 from "d3";
import type { PPOIterationLog } from "../../ppo/types";

const TRPO_BUDGET = 0.01;

export class PanelKLTrace extends HTMLElement {
  private svg!: d3.Selection<SVGSVGElement, undefined, null, undefined>;
  private readonly W = 380;
  private readonly H = 160;

  connectedCallback() {
    this.innerHTML = "";
    const title = document.createElement("p");
    title.style.cssText = "font-size:12px;font-weight:600;margin:0 0 4px;color:var(--rl-ink-muted);text-align:center;";
    title.textContent = "D — KL Divergence Trace";
    this.appendChild(title);

    this.svg = d3.create("svg").attr("viewBox", `0 0 ${this.W} ${this.H}`)
      .style("width", "100%").style("height", "auto");
    this.appendChild(this.svg.node()!);
    this.renderEmpty();
  }

  update(logs: readonly PPOIterationLog[]) {
    this.svg.selectAll("*").remove();
    if (logs.length === 0) { this.renderEmpty(); return; }

    const W = this.W, H = this.H;
    const pad = { l: 56, r: 20, t: 10, b: 30 };
    const kls = logs.map((l) => l.meanKL);
    const yMax = Math.max(...kls, TRPO_BUDGET * 1.5);

    const xScale = d3.scaleLinear().domain([0, logs.length - 1]).range([pad.l, W - pad.r]);
    const yScale = d3.scaleLinear().domain([0, yMax * 1.15]).range([H - pad.b, pad.t]);

    this.svg.append("g").attr("transform", `translate(0,${H - pad.b})`).call(d3.axisBottom(xScale).ticks(5));
    this.svg.append("g").attr("transform", `translate(${pad.l},0)`).call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".3f")));
    this.svg.append("text").attr("x", W / 2).attr("y", H - 1).attr("text-anchor", "middle")
      .attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text("Iteration");
    this.svg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 14)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text("Mean KL");

    // TRPO budget line.
    if (TRPO_BUDGET <= yMax) {
      this.svg.append("line")
        .attr("x1", pad.l).attr("y1", yScale(TRPO_BUDGET))
        .attr("x2", W - pad.r).attr("y2", yScale(TRPO_BUDGET))
        .attr("stroke", "#888").attr("stroke-width", 1).attr("stroke-dasharray", "4,3");
      this.svg.append("text").attr("x", W - pad.r - 2).attr("y", yScale(TRPO_BUDGET) - 3)
        .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#888").text("TRPO δ=0.01");
    }

    const line = d3.line<PPOIterationLog>().x((_, i) => xScale(i)).y((d) => yScale(d.meanKL));
    this.svg.append("path").datum(logs).attr("fill", "none")
      .attr("stroke", "#be185d").attr("stroke-width", 2).attr("d", line);

    const last = kls[kls.length - 1];
    this.svg.append("text").attr("x", W - pad.r).attr("y", yScale(last) - 4)
      .attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#be185d")
      .text(last.toFixed(5));
  }

  private renderEmpty() {
    const W = this.W, H = this.H;
    this.svg.selectAll("*").remove();
    this.svg.append("text").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", "#ccc").text("KL trace appears here");
  }
}

customElements.define("ppo-panel-kl-trace", PanelKLTrace);
