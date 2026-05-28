/**
 * Panel B — Value Bifurcation. V_soft[start] (violet) and V_pi[start] (green)
 * vs alpha. Both curves over the full alpha range; a vertical line marks current alpha.
 * Pink horizontal reference line at V^pi = 0.7217 (the L10 softmax cap).
 */
import * as d3 from "d3";

const CAP_V = 0.7217;
const V_STAR = 0.7290;

export class PanelValueBifurcation extends HTMLElement {
  private xScale!: d3.ScaleLogarithmic<number, number>;
  private yScale!: d3.ScaleLinear<number, number>;
  private cursorLine!: d3.Selection<SVGLineElement, unknown, null, undefined>;
  private cursorLabel!: d3.Selection<SVGTextElement, unknown, null, undefined>;

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
    this.xScale = d3.scaleLog().range([0, iW]);
    this.yScale = d3.scaleLinear().range([iH, 0]);

    const xAxisG = root.append("g").attr("transform", `translate(0,${iH})`);
    const yAxisG = root.append("g");
    root.append("text").attr("x", iW / 2).attr("y", iH + 38).attr("text-anchor", "middle")
      .attr("class", "axis-label").style("font-size", "10px").text("α (log scale)");
    root.append("text").attr("transform", "rotate(-90)").attr("y", -42).attr("x", -iH / 2)
      .attr("text-anchor", "middle").attr("class", "axis-label").style("font-size", "10px").text("Value");

    // Store refs for update
    (this as any)._xAxisG = xAxisG;
    (this as any)._yAxisG = yAxisG;
    (this as any)._iW = iW;
    (this as any)._iH = iH;
    (this as any)._root = root;

    // Cursor line (added last so it's on top)
    this.cursorLine = root.append("line")
      .attr("stroke", "var(--rl-ink-muted)").attr("stroke-width", 1.2)
      .attr("stroke-dasharray", "3 2").attr("y1", 0);
    this.cursorLabel = root.append("text")
      .attr("class", "annot").attr("fill", "var(--rl-ink-muted)").style("font-size", "9px");
  }

  setFullData(alphas: number[], V_softs: number[], V_pis: number[]) {
    const root = (this as any)._root as d3.Selection<SVGGElement, unknown, null, undefined>;
    const iW = (this as any)._iW as number;
    const iH = (this as any)._iH as number;

    this.xScale.domain([d3.min(alphas)!, d3.max(alphas)!]);
    const allVs = [...V_softs, ...V_pis];
    this.yScale.domain([Math.min(0, d3.min(allVs)!) - 0.02, d3.max(allVs)! + 0.05]);

    (this as any)._xAxisG.call(d3.axisBottom(this.xScale).ticks(5, ".2~g").tickSize(3))
      .selectAll("text").style("font-size", "9px").style("font-family", "var(--rl-font-mono)");
    (this as any)._yAxisG.call(d3.axisLeft(this.yScale).ticks(5).tickSize(3))
      .selectAll("text").style("font-size", "9px").style("font-family", "var(--rl-font-mono)");

    const lineGen = d3.line<[number, number]>().x(d => this.xScale(d[0])).y(d => this.yScale(d[1]));

    // V_soft curve (violet)
    const softPts = alphas.map((a, i) => [a, V_softs[i]] as [number, number]);
    const existingSoft = root.select(".bifur-vsoft");
    if (existingSoft.empty()) {
      root.insert("path", ".bifur-cursor").datum(softPts).attr("class", "bifur-vsoft")
        .attr("fill", "none").attr("stroke", "var(--maxent-soft)").attr("stroke-width", 2)
        .attr("d", lineGen);
    } else {
      existingSoft.datum(softPts).attr("d", lineGen);
    }

    // V_pi curve (green)
    const piPts = alphas.map((a, i) => [a, V_pis[i]] as [number, number]);
    const existingPi = root.select(".bifur-vpi");
    if (existingPi.empty()) {
      root.insert("path", ".bifur-cursor").datum(piPts).attr("class", "bifur-vpi")
        .attr("fill", "none").attr("stroke", "var(--maxent-hard)").attr("stroke-width", 2)
        .attr("d", lineGen);
    } else {
      existingPi.datum(piPts).attr("d", lineGen);
    }

    // V* dashed
    if (root.select(".bifur-vstar").empty()) {
      root.insert("line", ".bifur-cursor").attr("class", "bifur-vstar")
        .attr("x1", 0).attr("x2", iW)
        .attr("y1", this.yScale(V_STAR)).attr("y2", this.yScale(V_STAR))
        .attr("stroke", "var(--maxent-hard)").attr("stroke-width", 1).attr("stroke-dasharray", "4 3").attr("opacity", 0.5);
      root.insert("text", ".bifur-cursor").attr("class", "bifur-vstar-label")
        .attr("x", iW - 2).attr("y", this.yScale(V_STAR) - 3).attr("text-anchor", "end")
        .style("font-size", "8px").attr("fill", "var(--maxent-hard)").text("V*");
    }

    // L10 cap pink reference
    if (root.select(".bifur-cap").empty()) {
      root.insert("line", ".bifur-cursor").attr("class", "bifur-cap")
        .attr("x1", 0).attr("x2", iW)
        .attr("y1", this.yScale(CAP_V)).attr("y2", this.yScale(CAP_V))
        .attr("stroke", "var(--maxent-cap)").attr("stroke-width", 1.2).attr("stroke-dasharray", "3 2");
      root.insert("text", ".bifur-cursor").attr("class", "bifur-cap-label")
        .attr("x", 2).attr("y", this.yScale(CAP_V) - 3).style("font-size", "8px")
        .attr("fill", "var(--maxent-cap)").text("L10 cap 0.722");
    }

    // Legend
    if (root.select(".bifur-legend").empty()) {
      const leg = root.append("g").attr("class", "bifur-legend").attr("transform", `translate(${iW - 120},4)`);
      [{ color: "var(--maxent-soft)", label: "V_soft (fixed pt)" },
       { color: "var(--maxent-hard)", label: "Vᵯⁿ (true return)" }].forEach((d, i) => {
        leg.append("line").attr("x1", 0).attr("x2", 14).attr("y1", i * 16 + 7).attr("y2", i * 16 + 7)
          .attr("stroke", d.color).attr("stroke-width", 2);
        leg.append("text").attr("x", 18).attr("y", i * 16 + 11).style("font-size", "8px")
          .attr("fill", "var(--rl-ink-muted)").text(d.label);
      });
    }

    this.cursorLine.attr("y2", iH);
  }

  updateCursor(alpha: number) {
    const iH = (this as any)._iH as number;
    const x = this.xScale(alpha);
    this.cursorLine.attr("x1", x).attr("x2", x).attr("y2", iH);
    this.cursorLabel.attr("x", x + 3).attr("y", 10).text(`α=${alpha.toFixed(3)}`);
  }
}

customElements.define("esl-panel-value-bifurcation", PanelValueBifurcation);
