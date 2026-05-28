/**
 * Panel E — Trajectory length histogram. Bar chart of steps to terminal
 * from Monte Carlo rollouts at the current alpha.
 */
import * as d3 from "d3";

export class PanelLengthHistogram extends HTMLElement {
  connectedCallback() {
    const W = 280, H = 200;
    const m = { top: 24, right: 12, bottom: 36, left: 48 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    this.appendChild(svgEl);

    const root = d3.select(svgEl).append("g").attr("transform", `translate(${m.left},${m.top})`);
    (this as any)._root = root;
    (this as any)._iW = iW;
    (this as any)._iH = iH;

    root.append("text").attr("x", iW / 2).attr("y", -10).attr("text-anchor", "middle")
      .attr("class", "chart-title").style("font-size", "11px").attr("fill", "var(--rl-ink)")
      .text("Trajectory lengths");

    const xAxisG = root.append("g").attr("transform", `translate(0,${iH})`);
    const yAxisG = root.append("g");
    (this as any)._xAxisG = xAxisG;
    (this as any)._yAxisG = yAxisG;

    root.append("text").attr("x", iW / 2).attr("y", iH + 30).attr("text-anchor", "middle")
      .style("font-size", "9px").attr("fill", "var(--rl-ink-muted)").text("Steps");
  }

  update(histogram: number[], maxSteps: number) {
    const root = (this as any)._root as d3.Selection<SVGGElement, unknown, null, undefined>;
    const iW = (this as any)._iW as number;
    const iH = (this as any)._iH as number;

    root.selectAll("rect").remove();

    const N = histogram.length;
    const xScale = d3.scaleLinear().domain([0, maxSteps]).range([0, iW]);
    const yScale = d3.scaleLinear().domain([0, Math.max(...histogram) * 1.15 + 1e-6]).range([iH, 0]);
    const binW = iW / N;

    (this as any)._xAxisG.call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .selectAll("text").style("font-size", "8px");
    (this as any)._yAxisG.call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".2f")).tickSize(3))
      .selectAll("text").style("font-size", "8px");

    histogram.forEach((v, i) => {
      root.append("rect")
        .attr("x", i * binW).attr("y", yScale(v))
        .attr("width", Math.max(1, binW - 0.5)).attr("height", iH - yScale(v))
        .attr("fill", "var(--maxent-soft)").attr("opacity", 0.65);
    });
  }
}

customElements.define("esl-panel-length-histogram", PanelLengthHistogram);
