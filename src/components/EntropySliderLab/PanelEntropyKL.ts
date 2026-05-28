/**
 * Panel C — Entropy / KL. Two curves over the alpha axis:
 * mean H(pi*(.|s)) (pink) and KL(pi* || uniform) (cyan).
 */
import * as d3 from "d3";

export class PanelEntropyKL extends HTMLElement {
  private xScale!: d3.ScaleLogarithmic<number, number>;
  private yScale!: d3.ScaleLinear<number, number>;
  private cursorLine!: d3.Selection<SVGLineElement, unknown, null, undefined>;

  connectedCallback() {
    const W = 380, H = 220;
    const m = { top: 24, right: 20, bottom: 44, left: 52 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    this.appendChild(svgEl);

    const root = d3.select(svgEl).append("g").attr("transform", `translate(${m.left},${m.top})`);
    (this as any)._root = root;
    (this as any)._iH = iH;

    this.xScale = d3.scaleLog().range([0, iW]);
    this.yScale = d3.scaleLinear().range([iH, 0]);

    const xAxisG = root.append("g").attr("transform", `translate(0,${iH})`);
    const yAxisG = root.append("g");
    (this as any)._xAxisG = xAxisG;
    (this as any)._yAxisG = yAxisG;

    root.append("text").attr("x", iW / 2).attr("y", iH + 38).attr("text-anchor", "middle")
      .attr("class", "axis-label").style("font-size", "10px").text("α (log scale)");
    root.append("text").attr("transform", "rotate(-90)").attr("y", -42).attr("x", -iH / 2)
      .attr("text-anchor", "middle").attr("class", "axis-label").style("font-size", "10px").text("Nats");

    // Max entropy reference: log(4)
    const H_max = Math.log(4);
    root.append("line").attr("x1", 0).attr("x2", iW)
      .attr("y1", this.yScale(H_max)).attr("y2", this.yScale(H_max))
      .attr("stroke", "var(--rl-border)").attr("stroke-dasharray", "3 2").attr("stroke-width", 0.8);
    root.append("text").attr("x", iW - 2).attr("y", this.yScale(H_max) - 3).attr("text-anchor", "end")
      .style("font-size", "8px").attr("fill", "var(--rl-ink-muted)").text("log(4)");

    this.cursorLine = root.append("line")
      .attr("stroke", "var(--rl-ink-muted)").attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");

    // Legend
    const leg = root.append("g").attr("transform", `translate(4,4)`);
    [{ color: "var(--maxent-entropy)", label: "H(π*(·|s)) mean" },
     { color: "var(--maxent-likelihood)", label: "KL(π* ∥ uniform)" }].forEach((d, i) => {
      leg.append("line").attr("x1", 0).attr("x2", 14).attr("y1", i * 15 + 7).attr("y2", i * 15 + 7)
        .attr("stroke", d.color).attr("stroke-width", 2);
      leg.append("text").attr("x", 18).attr("y", i * 15 + 11).style("font-size", "8px")
        .attr("fill", "var(--rl-ink-muted)").text(d.label);
    });
  }

  setFullData(alphas: number[], meanEntropies: number[], klToUniform: number[]) {
    const root = (this as any)._root as d3.Selection<SVGGElement, unknown, null, undefined>;
    const iH = (this as any)._iH as number;

    this.xScale.domain([d3.min(alphas)!, d3.max(alphas)!]);
    const H_max = Math.log(4);
    this.yScale.domain([-0.05, H_max * 1.1]);

    (this as any)._xAxisG.call(d3.axisBottom(this.xScale).ticks(5, ".2~g").tickSize(3))
      .selectAll("text").style("font-size", "9px");
    (this as any)._yAxisG.call(d3.axisLeft(this.yScale).ticks(5).tickSize(3))
      .selectAll("text").style("font-size", "9px");

    const lineGen = d3.line<[number, number]>().x(d => this.xScale(d[0])).y(d => this.yScale(d[1]));
    const hPts = alphas.map((a, i) => [a, meanEntropies[i]] as [number, number]);
    const klPts = alphas.map((a, i) => [a, klToUniform[i]] as [number, number]);

    const h_sel = root.select(".ekl-entropy");
    if (h_sel.empty()) {
      root.insert("path", "line").datum(hPts).attr("class", "ekl-entropy")
        .attr("fill", "none").attr("stroke", "var(--maxent-entropy)").attr("stroke-width", 2).attr("d", lineGen);
    } else {
      h_sel.datum(hPts).attr("d", lineGen);
    }

    const kl_sel = root.select(".ekl-kl");
    if (kl_sel.empty()) {
      root.insert("path", "line").datum(klPts).attr("class", "ekl-kl")
        .attr("fill", "none").attr("stroke", "var(--maxent-likelihood)").attr("stroke-width", 2).attr("d", lineGen);
    } else {
      kl_sel.datum(klPts).attr("d", lineGen);
    }

    this.cursorLine.attr("y1", 0).attr("y2", iH);
  }

  updateCursor(alpha: number) {
    const x = this.xScale(alpha);
    this.cursorLine.attr("x1", x).attr("x2", x);
  }
}

customElements.define("esl-panel-entropy-kl", PanelEntropyKL);
