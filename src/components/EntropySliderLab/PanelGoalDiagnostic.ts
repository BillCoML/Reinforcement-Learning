/**
 * Panel D — Goal-reach diagnostic. Stacked bars: green=goal, gray=timeout.
 * Threshold line at 10% goal-reach probability.
 */
import * as d3 from "d3";

export class PanelGoalDiagnostic extends HTMLElement {
  connectedCallback() {
    const W = 280, H = 200;
    const m = { top: 24, right: 16, bottom: 28, left: 16 };
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
      .text("Goal-reach probability");
  }

  update(goalReachProb: number, timeoutProb: number, pitReachProb: number) {
    const root = (this as any)._root as d3.Selection<SVGGElement, unknown, null, undefined>;
    const iW = (this as any)._iW as number;
    const iH = (this as any)._iH as number;

    root.selectAll(".gd-bar, .gd-label, .gd-threshold").remove();

    // Single stacked bar
    const barW = iW * 0.6;
    const barX = (iW - barW) / 2;

    // Goal (green)
    const goalH = goalReachProb * iH;
    root.append("rect").attr("class", "gd-bar")
      .attr("x", barX).attr("y", iH - goalH)
      .attr("width", barW).attr("height", goalH)
      .attr("fill", "var(--maxent-hard)").attr("opacity", 0.8);

    // Pit (small, gray-blue)
    const pitH = pitReachProb * iH;
    root.append("rect").attr("class", "gd-bar")
      .attr("x", barX).attr("y", iH - goalH - pitH)
      .attr("width", barW).attr("height", pitH)
      .attr("fill", "#475569").attr("opacity", 0.6);

    // Timeout (red)
    const timeoutH = timeoutProb * iH;
    root.append("rect").attr("class", "gd-bar")
      .attr("x", barX).attr("y", iH - goalH - pitH - timeoutH)
      .attr("width", barW).attr("height", timeoutH)
      .attr("fill", "var(--maxent-failure)").attr("opacity", 0.7);

    // Percentage labels
    if (goalReachProb > 0.05) {
      root.append("text").attr("class", "gd-label")
        .attr("x", barX + barW / 2).attr("y", iH - goalH / 2 + 4)
        .attr("text-anchor", "middle").style("font-size", "11px").attr("fill", "white").attr("font-weight", "bold")
        .text(`${(goalReachProb * 100).toFixed(0)}%`);
    }
    if (timeoutProb > 0.05) {
      root.append("text").attr("class", "gd-label")
        .attr("x", barX + barW / 2).attr("y", iH - goalH - pitH - timeoutH / 2 + 4)
        .attr("text-anchor", "middle").style("font-size", "11px").attr("fill", "white").attr("font-weight", "bold")
        .text(`${(timeoutProb * 100).toFixed(0)}%`);
    }

    // Labels below bar
    root.append("text").attr("class", "gd-label")
      .attr("x", barX).attr("y", iH + 16).style("font-size", "9px").attr("fill", "var(--maxent-hard)")
      .text("■ Goal");
    root.append("text").attr("class", "gd-label")
      .attr("x", barX + barW / 2).attr("y", iH + 16).attr("text-anchor", "middle").style("font-size", "9px").attr("fill", "var(--maxent-failure)")
      .text("■ Timeout");

    // 10% threshold line
    const thresh10 = 0.10 * iH;
    root.append("line").attr("class", "gd-threshold")
      .attr("x1", barX - 4).attr("x2", barX + barW + 4)
      .attr("y1", iH - thresh10).attr("y2", iH - thresh10)
      .attr("stroke", "var(--maxent-failure)").attr("stroke-dasharray", "3 2").attr("stroke-width", 1);
    root.append("text").attr("class", "gd-threshold")
      .attr("x", barX + barW + 6).attr("y", iH - thresh10 + 4)
      .style("font-size", "8px").attr("fill", "var(--maxent-failure)").text("10%");
  }
}

customElements.define("esl-panel-goal-diagnostic", PanelGoalDiagnostic);
