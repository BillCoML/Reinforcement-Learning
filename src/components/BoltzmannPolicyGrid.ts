/**
 * V4 — Boltzmann Policy Grid. 3×3 grid of bar charts showing pi*(a|s) at the
 * selected alpha. Driven by alpha_sweep.json. Alpha slider shared with EntropySliderLab
 * if present, otherwise standalone.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface SweepEntry {
  alpha: number;
  pi: number[][];   // [state][action]
  V_soft: number[]; // [state]
}

const GRID = 3;
const N_S = 9;
const ACTION_NAMES = ["↑", "→", "↓", "←"];
const CELL_SIZE = 120;

customElements.define(
  "boltzmann-policy-grid",
  class extends HTMLElement {
    private data: SweepEntry[] = [];
    private alphaIdx = 20;

    async connectedCallback() {
      const { panel, body } = createPanel({ id: "boltzmann-policy-grid" });
      this.appendChild(panel);

      // Controls
      const controls = document.createElement("div");
      controls.className = "controls-row";
      controls.innerHTML = `
        <label class="slider-label" style="flex:1">Temperature α
          <input type="range" id="bpg-alpha" min="0" max="60" step="1" value="20" style="width:100%">
          <span id="bpg-alpha-val" style="font-family:var(--rl-font-mono)">—</span>
        </label>
      `;
      body.appendChild(controls);

      // Grid container
      const gridWrap = document.createElement("div");
      gridWrap.style.cssText = "overflow-x:auto;";
      body.appendChild(gridWrap);

      const totalW = CELL_SIZE * GRID + 20;
      const totalH = CELL_SIZE * GRID + 20;

      const svg = d3.select(gridWrap)
        .append("svg")
        .attr("viewBox", `0 0 ${totalW} ${totalH}`)
        .attr("width", "100%")
        .classed("rl-svg", true);


      try {
        const res = await fetch("/data/maxent/alpha_sweep.json");
        this.data = await res.json();
      } catch {
        body.textContent = "Data not available.";
        return;
      }

      const slider = body.querySelector("#bpg-alpha") as HTMLInputElement;
      slider.max = String(this.data.length - 1);
      const alphaLabel = body.querySelector("#bpg-alpha-val") as HTMLElement;

      const draw = (idx: number) => {
        const entry = this.data[idx];
        alphaLabel.textContent = `α = ${entry.alpha.toFixed(4)}`;

        svg.selectAll("*").remove();

        // Draw 3×3 grid of bar charts
        for (let s = 0; s < N_S; s++) {
          const row = Math.floor(s / GRID);
          const col = s % GRID;
          const cx = col * CELL_SIZE + 10;
          const cy = row * CELL_SIZE + 10;
          const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);

          const cellW = CELL_SIZE - 8;
          const cellH = CELL_SIZE - 8;

          // Background tint from V_soft
          const vSoftMax = 0.8; // rough max for normalization
          const vNorm = Math.max(0, Math.min(1, entry.V_soft[s] / vSoftMax));
          const isTerminal = s === 4 || s === 8;
          const bgColor = isTerminal
            ? (s === 8 ? "var(--maxent-hard-tint)" : "var(--maxent-failure-tint)")
            : `rgba(124,58,237,${0.05 + vNorm * 0.18})`;

          g.append("rect").attr("width", cellW).attr("height", cellH)
            .attr("rx", 6).attr("fill", bgColor)
            .attr("stroke", isTerminal ? (s === 8 ? "var(--maxent-hard)" : "var(--maxent-failure)") : "var(--rl-border)")
            .attr("stroke-width", isTerminal ? 2 : 1);

          // State label
          const rr = Math.floor(s / GRID), cc = s % GRID;
          g.append("text").attr("x", 4).attr("y", 13)
            .style("font-size", "9px").style("font-family", "var(--rl-font-mono)")
            .attr("fill", "var(--rl-ink-muted)")
            .text(`(${rr},${cc})`);

          if (isTerminal) {
            g.append("text").attr("x", cellW / 2).attr("y", cellH / 2 + 4)
              .attr("text-anchor", "middle").style("font-size", "12px").attr("fill", "var(--rl-ink-muted)")
              .text(s === 8 ? "GOAL" : "PIT");
            continue;
          }

          // 4 action bars, 2×2 layout
          const pi_s = entry.pi[s];
          const barW = cellW / 2 - 6;
          const barMaxH = cellH / 2 - 14;

          pi_s.forEach((p, a) => {
            const bx = (a % 2) * (cellW / 2) + 3;
            const by_base = Math.floor(a / 2) * (cellH / 2) + 18;
            const bh = p * barMaxH;

            g.append("rect")
              .attr("x", bx).attr("y", by_base + barMaxH - bh)
              .attr("width", barW).attr("height", bh)
              .attr("fill", "var(--maxent-soft)").attr("opacity", 0.7);

            g.append("text").attr("x", bx + barW / 2).attr("y", by_base + barMaxH + 10)
              .attr("text-anchor", "middle").style("font-size", "9px")
              .attr("fill", "var(--rl-ink-muted)").text(ACTION_NAMES[a]);

            g.append("text").attr("x", bx + barW / 2).attr("y", by_base + barMaxH - bh - 2)
              .attr("text-anchor", "middle").style("font-size", "8px")
              .attr("fill", "var(--maxent-soft)")
              .text(p > 0.05 ? p.toFixed(2) : "");
          });
        }
      };

      slider.addEventListener("input", () => draw(+slider.value));
      draw(this.alphaIdx);
    }
  },
);
