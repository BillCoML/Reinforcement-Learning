/**
 * Panel C — Clip fraction over training.
 */
import * as d3 from "d3";
import type { PPOIterationLog } from "../../ppo/types";

export class PanelClipFraction extends HTMLElement {
  private svg!: d3.Selection<SVGSVGElement, undefined, null, undefined>;
  private readonly W = 380;
  private readonly H = 160;

  connectedCallback() {
    this.innerHTML = "";
    const title = document.createElement("p");
    title.style.cssText = "font-size:12px;font-weight:600;margin:0 0 4px;color:var(--rl-ink-muted);text-align:center;";
    title.textContent = "C — Clip Fraction";
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
    const pad = { l: 48, r: 16, t: 10, b: 30 };
    const clips = logs.map((l) => l.clipFraction);
    const yMax = Math.max(...clips, 0.05);

    const xScale = d3.scaleLinear().domain([0, logs.length - 1]).range([pad.l, W - pad.r]);
    const yScale = d3.scaleLinear().domain([0, yMax * 1.15]).range([H - pad.b, pad.t]);

    this.svg.append("g").attr("transform", `translate(0,${H - pad.b})`).call(d3.axisBottom(xScale).ticks(5));
    this.svg.append("g").attr("transform", `translate(${pad.l},0)`).call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".0%")));
    this.svg.append("text").attr("x", W / 2).attr("y", H - 1).attr("text-anchor", "middle")
      .attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text("Iteration");
    this.svg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 12)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text("Clip frac (last epoch)");

    const area = d3.area<PPOIterationLog>()
      .x((_, i) => xScale(i)).y0(yScale(0)).y1((d) => yScale(d.clipFraction))
      .curve(d3.curveStepAfter);
    this.svg.append("path").datum(logs).attr("fill", "rgba(234,88,12,0.15)").attr("d", area);

    const line = d3.line<PPOIterationLog>().x((_, i) => xScale(i)).y((d) => yScale(d.clipFraction)).curve(d3.curveStepAfter);
    this.svg.append("path").datum(logs).attr("fill", "none")
      .attr("stroke", "#ea580c").attr("stroke-width", 2).attr("d", line);

    // Latest value.
    const last = clips[clips.length - 1];
    this.svg.append("text").attr("x", W - pad.r).attr("y", yScale(last) - 4)
      .attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#ea580c")
      .text(`${(last * 100).toFixed(1)}%`);
  }

  private renderEmpty() {
    const W = this.W, H = this.H;
    this.svg.selectAll("*").remove();
    this.svg.append("text").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", "#ccc").text("Clip fraction appears here");
  }
}

customElements.define("ppo-panel-clip-fraction", PanelClipFraction);
