/**
 * Panel B — Learning curves: V(start), surrogate value, moving-avg return.
 */
import * as d3 from "d3";
import type { PPOIterationLog } from "../../ppo/types";

export class PanelLearningCurves extends HTMLElement {
  private svg!: d3.Selection<SVGSVGElement, undefined, null, undefined>;
  private readonly W = 380;
  private readonly H = 200;

  connectedCallback() {
    this.innerHTML = "";
    const title = document.createElement("p");
    title.style.cssText = "font-size:12px;font-weight:600;margin:0 0 4px;color:var(--rl-ink-muted);text-align:center;";
    title.textContent = "B — Learning Curves";
    this.appendChild(title);

    this.svg = d3.create("svg").attr("viewBox", `0 0 ${this.W} ${this.H}`)
      .style("width", "100%").style("height", "auto");
    this.appendChild(this.svg.node()!);

    const legend = document.createElement("div");
    legend.style.cssText = "display:flex;gap:12px;font-size:10px;flex-wrap:wrap;margin-top:3px;";
    [
      { color: "#7c3aed", label: "V(s₀) exact" },
      { color: "#0e7490", label: "Surrogate" },
    ].forEach(({ color, label }) => {
      const item = document.createElement("div");
      item.style.cssText = "display:flex;gap:4px;align-items:center;";
      const dot = document.createElement("span");
      dot.style.cssText = `background:${color};width:10px;height:3px;display:inline-block;border-radius:2px;`;
      item.appendChild(dot);
      item.appendChild(Object.assign(document.createElement("span"), { textContent: label, style: "color:var(--rl-ink-muted);" }));
      legend.appendChild(item);
    });
    this.appendChild(legend);
    this.renderEmpty();
  }

  update(logs: readonly PPOIterationLog[]) {
    this.svg.selectAll("*").remove();
    if (logs.length === 0) { this.renderEmpty(); return; }

    const W = this.W, H = this.H;
    const pad = { l: 48, r: 16, t: 12, b: 30 };

    const vStarts = logs.map((l) => l.vStart);
    const surrs = logs.map((l) => l.surrogateValue);
    const allY = [...vStarts, ...surrs];
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const yPad = (yMax - yMin) * 0.1 || 0.05;

    const xScale = d3.scaleLinear().domain([0, logs.length - 1]).range([pad.l, W - pad.r]);
    const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([H - pad.b, pad.t]);

    this.svg.append("g").attr("transform", `translate(0,${H - pad.b})`).call(d3.axisBottom(xScale).ticks(5));
    this.svg.append("g").attr("transform", `translate(${pad.l},0)`).call(d3.axisLeft(yScale).ticks(4));
    this.svg.append("text").attr("x", W / 2).attr("y", H - 1).attr("text-anchor", "middle")
      .attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text("Iteration");

    const lineGen = (key: "vStart" | "surrogateValue", color: string, dash?: string) => {
      const l = d3.line<PPOIterationLog>().x((_, i) => xScale(i)).y((d) => yScale(d[key]));
      this.svg.append("path").datum(logs).attr("fill", "none")
        .attr("stroke", color).attr("stroke-width", 2)
        .attr("stroke-dasharray", dash || "none").attr("d", l);
    };

    lineGen("vStart", "#7c3aed");
    lineGen("surrogateValue", "#0e7490", "4,3");

    // Final value label.
    const lastV = vStarts[vStarts.length - 1];
    this.svg.append("text").attr("x", W - pad.r).attr("y", yScale(lastV) - 4)
      .attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#7c3aed")
      .attr("font-weight", "600").text(lastV.toFixed(4));
  }

  private renderEmpty() {
    const W = this.W, H = this.H;
    this.svg.selectAll("*").remove();
    this.svg.append("text").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", "#ccc").text("Run PPO to see learning curves");
  }
}

customElements.define("ppo-panel-learning-curves", PanelLearningCurves);
