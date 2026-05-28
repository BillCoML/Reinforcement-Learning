/**
 * Panel A — Policy Heatmap. 3×3 gridworld with Boltzmann policy bars in each cell.
 * Background tint: V_soft(s) normalized.
 */
import * as d3 from "d3";

const GRID = 3;
const ACTION_NAMES = ["↑", "→", "↓", "←"];
const CELL = 82;

export class PanelPolicyHeatmap extends HTMLElement {
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private root!: d3.Selection<SVGGElement, unknown, null, undefined>;

  connectedCallback() {
    const W = CELL * GRID + 8;
    const H = CELL * GRID + 8;
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    this.appendChild(svgEl);
    this.svg = d3.select(svgEl);
    this.root = this.svg.append("g").attr("transform", "translate(4,4)");
  }

  update(pi: number[][], V_soft: number[]) {
    this.root.selectAll("*").remove();
    const maxV = Math.max(...V_soft.filter((_, s) => s !== 4 && s !== 8));

    for (let s = 0; s < 9; s++) {
      const row = Math.floor(s / GRID), col = s % GRID;
      const cx = col * CELL, cy = row * CELL;
      const g = this.root.append("g").attr("transform", `translate(${cx},${cy})`);

      const isGoal = s === 8, isPit = s === 4;
      const vNorm = maxV > 0 ? Math.max(0, V_soft[s] / maxV) : 0;
      const bg = isGoal ? "#dcfce7" : isPit ? "#fee2e2" : `rgba(124,58,237,${0.04 + vNorm * 0.2})`;

      g.append("rect").attr("width", CELL - 4).attr("height", CELL - 4).attr("rx", 6)
        .attr("fill", bg)
        .attr("stroke", isGoal ? "#15803d" : isPit ? "#dc2626" : "#e5e7eb").attr("stroke-width", 1.5);

      const rr = Math.floor(s / GRID), cc = s % GRID;
      g.append("text").attr("x", 4).attr("y", 13).style("font-size", "8px")
        .style("font-family", "var(--rl-font-mono)").attr("fill", "#9ca3af").text(`(${rr},${cc})`);

      if (isGoal || isPit) {
        g.append("text").attr("x", (CELL - 4) / 2).attr("y", (CELL - 4) / 2 + 4)
          .attr("text-anchor", "middle").style("font-size", "11px").attr("fill", "#6b7280")
          .text(isGoal ? "GOAL" : "PIT");
        continue;
      }

      // 2×2 bars (up/right top row, down/left bottom row)
      const bW = (CELL - 4) / 2 - 4;
      const bMaxH = (CELL - 4) / 2 - 14;
      const positions = [[2, 17], [(CELL - 4) / 2 + 2, 17], [2, (CELL - 4) / 2 + 5], [(CELL - 4) / 2 + 2, (CELL - 4) / 2 + 5]];

      pi[s].forEach((p, a) => {
        const [bx, by] = positions[a];
        const bh = p * bMaxH;
        g.append("rect").attr("x", bx).attr("y", by + bMaxH - bh)
          .attr("width", bW).attr("height", bh)
          .attr("fill", "var(--maxent-soft)").attr("opacity", 0.7);
        g.append("text").attr("x", bx + bW / 2).attr("y", by + bMaxH + 10)
          .attr("text-anchor", "middle").style("font-size", "8px").attr("fill", "#6b7280")
          .text(ACTION_NAMES[a]);
      });
    }
  }
}

customElements.define("esl-panel-policy-heatmap", PanelPolicyHeatmap);
