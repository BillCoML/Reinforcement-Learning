/**
 * Panel A — Policy and value heatmap on the 3×3 gridworld.
 * Action bars at each cell + value background tint.
 */
import * as d3 from "d3";

export class PanelGridworld extends HTMLElement {
  private svg!: d3.Selection<SVGSVGElement, undefined, null, undefined>;
  private readonly W = 280;
  private readonly H = 280;
  private readonly CELL = 80;
  private readonly GRID = 3;
  private readonly PIT = 4;
  private readonly GOAL = 8;

  connectedCallback() {
    this.innerHTML = "";
    const title = document.createElement("p");
    title.style.cssText = "font-size:12px;font-weight:600;margin:0 0 4px;color:var(--rl-ink-muted);text-align:center;";
    title.textContent = "A — Policy & Value";
    this.appendChild(title);

    this.svg = d3.create("svg").attr("viewBox", `0 0 ${this.W} ${this.H}`)
      .style("width", "100%").style("height", "auto");
    this.appendChild(this.svg.node()!);
    this.drawEmpty();
  }

  update(theta: Float64Array, V: Float64Array, nA: number) {
    this.svg.selectAll("*").remove();
    const C = this.CELL;

    const vMin = Math.min(...Array.from(V));
    const vMax = Math.max(...Array.from(V));
    const vColor = d3.scaleSequential(d3.interpolate("#f0f8ff", "#0e7490")).domain([vMin, vMax]);

    for (let s = 0; s < 9; s++) {
      const row = Math.floor(s / this.GRID);
      const col = s % this.GRID;
      const x = col * C, y = row * C;

      // Background tint.
      const isTerminal = s === this.PIT || s === this.GOAL;
      this.svg.append("rect").attr("x", x).attr("y", y).attr("width", C).attr("height", C)
        .attr("fill", isTerminal ? (s === this.GOAL ? "#dcfce7" : "#fee2e2") : vColor(V[s]))
        .attr("stroke", "#ccc").attr("stroke-width", 1);

      // State label.
      this.svg.append("text").attr("x", x + 4).attr("y", y + 12)
        .attr("font-size", 9).attr("fill", "#888").text(`s${s}`);

      // V value.
      this.svg.append("text").attr("x", x + C / 2).attr("y", y + 22)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#1c1e22")
        .attr("font-weight", "600").text(V[s].toFixed(3));

      if (isTerminal) {
        this.svg.append("text").attr("x", x + C / 2).attr("y", y + C / 2 + 6)
          .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", s === this.GOAL ? "#15803d" : "#dc2626")
          .text(s === this.GOAL ? "GOAL" : "PIT");
        continue;
      }

      // Action probability bars (4 directions at center of cell).
      const base = s * nA;
      let maxLogit = -Infinity;
      for (let a = 0; a < nA; a++) if (theta[base + a] > maxLogit) maxLogit = theta[base + a];
      let sm = 0;
      const probs = new Float64Array(nA);
      for (let a = 0; a < nA; a++) { probs[a] = Math.exp(theta[base + a] - maxLogit); sm += probs[a]; }
      for (let a = 0; a < nA; a++) probs[a] /= sm;

      const cx = x + C / 2, cy = y + C / 2;
      const maxBarLen = 24;
      // Up(0), Right(1), Down(2), Left(3).
      const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // dx, dy for each action

      probs.forEach((p, a) => {
        const [dx, dy] = dirs[a];
        const len = p * maxBarLen;
        // Arrow from center toward direction.
        const ex = cx + dx * len, ey = cy + dy * len;
        this.svg.append("line").attr("x1", cx).attr("y1", cy).attr("x2", ex).attr("y2", ey)
          .attr("stroke", "#7c3aed").attr("stroke-width", Math.max(1, p * 5))
          .attr("stroke-linecap", "round");
        // Probability label for dominant action.
        if (p > 0.35) {
          this.svg.append("text").attr("x", cx + dx * (len + 5)).attr("y", cy + dy * (len + 5) + 3)
            .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#7c3aed")
            .text(p.toFixed(2));
        }
      });
    }
  }

  private drawEmpty() {
    const C = this.CELL;
    for (let s = 0; s < 9; s++) {
      const row = Math.floor(s / this.GRID);
      const col = s % this.GRID;
      this.svg.append("rect").attr("x", col * C).attr("y", row * C)
        .attr("width", C).attr("height", C).attr("fill", "#f8f8f8").attr("stroke", "#ccc").attr("stroke-width", 1);
      this.svg.append("text").attr("x", col * C + C / 2).attr("y", row * C + C / 2 + 4)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#bbb").text(`s${s}`);
    }
  }
}

customElements.define("ppo-panel-gridworld", PanelGridworld);
