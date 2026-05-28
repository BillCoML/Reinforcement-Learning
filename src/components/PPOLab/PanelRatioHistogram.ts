/**
 * Panel E — Probability ratio histogram.
 * Shows distribution of r_t = π_new(a|s) / π_old(a|s) across the current batch.
 * Clip band [1-ε, 1+ε] shaded. Samples outside band rendered in orange.
 */
import * as d3 from "d3";

export class PanelRatioHistogram extends HTMLElement {
  private svg!: d3.Selection<SVGSVGElement, undefined, null, undefined>;
  private readonly W = 380;
  private readonly H = 220;

  connectedCallback() {
    this.innerHTML = "";
    const title = document.createElement("p");
    title.style.cssText = "font-size:12px;font-weight:600;margin:0 0 4px;color:var(--rl-ink-muted);text-align:center;";
    title.textContent = "E — Ratio Histogram";
    this.appendChild(title);

    this.svg = d3.create("svg").attr("viewBox", `0 0 ${this.W} ${this.H}`)
      .style("width", "100%").style("height", "auto");
    this.appendChild(this.svg.node()!);
    this.renderEmpty();
  }

  update(ratios: Float64Array, clipEps: number) {
    this.svg.selectAll("*").remove();
    if (!ratios || ratios.length === 0) { this.renderEmpty(); return; }

    const W = this.W, H = this.H;
    const pad = { l: 44, r: 16, t: 12, b: 36 };

    const rArr = Array.from(ratios);
    const rMin = Math.min(...rArr), rMax = Math.max(...rArr);
    const xMin = Math.min(rMin, 1 - clipEps) - 0.05;
    const xMax = Math.max(rMax, 1 + clipEps) + 0.05;

    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([pad.l, W - pad.r]);
    const nBins = 20;
    const bins = d3.bin().domain([xMin, xMax]).thresholds(nBins)(rArr);
    const yMax = Math.max(...bins.map((b) => b.length));
    const yScale = d3.scaleLinear().domain([0, yMax * 1.15]).range([H - pad.b, pad.t]);

    // Clip region background.
    this.svg.append("rect")
      .attr("x", xScale(1 - clipEps)).attr("y", pad.t)
      .attr("width", xScale(1 + clipEps) - xScale(1 - clipEps))
      .attr("height", H - pad.t - pad.b)
      .attr("fill", "rgba(251,191,36,0.2)");

    this.svg.append("g").attr("transform", `translate(0,${H - pad.b})`).call(d3.axisBottom(xScale).ticks(7));
    this.svg.append("g").attr("transform", `translate(${pad.l},0)`).call(d3.axisLeft(yScale).ticks(4));
    this.svg.append("text").attr("x", W / 2).attr("y", H - 1).attr("text-anchor", "middle")
      .attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text("r = π_new(a|s) / π_old(a|s)");

    // Clip band boundary lines.
    [1 - clipEps, 1 + clipEps].forEach((x) => {
      this.svg.append("line")
        .attr("x1", xScale(x)).attr("y1", pad.t)
        .attr("x2", xScale(x)).attr("y2", H - pad.b)
        .attr("stroke", "#b45309").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,2");
    });

    // Bars colored by whether they fall inside or outside the clip band.
    bins.forEach((bin) => {
      const bx0 = xScale(bin.x0!);
      const bx1 = xScale(bin.x1!);
      const midX = (bin.x0! + bin.x1!) / 2;
      const isClipped = midX < 1 - clipEps || midX > 1 + clipEps;
      this.svg.append("rect")
        .attr("x", bx0 + 1).attr("y", yScale(bin.length))
        .attr("width", Math.max(0, bx1 - bx0 - 2))
        .attr("height", H - pad.b - yScale(bin.length))
        .attr("fill", isClipped ? "#ea580c" : "#7c3aed")
        .attr("opacity", isClipped ? 0.8 : 0.6);
    });

    // r=1 line.
    this.svg.append("line")
      .attr("x1", xScale(1)).attr("y1", pad.t)
      .attr("x2", xScale(1)).attr("y2", H - pad.b)
      .attr("stroke", "#1c1e22").attr("stroke-width", 1).attr("stroke-dasharray", "2,2");
    this.svg.append("text").attr("x", xScale(1)).attr("y", pad.t - 2)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#1c1e22").text("r=1");

    // Legend.
    const clipCount = rArr.filter((r) => r < 1 - clipEps || r > 1 + clipEps).length;
    const clipPct = ((clipCount / rArr.length) * 100).toFixed(1);
    this.svg.append("text").attr("x", W - pad.r).attr("y", pad.t + 10)
      .attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", "#ea580c")
      .text(`Clipped: ${clipPct}% (${clipCount}/${rArr.length})`);
  }

  private renderEmpty() {
    const W = this.W, H = this.H;
    this.svg.selectAll("*").remove();
    this.svg.append("text").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", "#ccc").text("Ratio histogram appears here");
  }
}

customElements.define("ppo-panel-ratio-histogram", PanelRatioHistogram);
