/**
 * V5 — Failure Diagnostic. Three synchronized panels parameterized by alpha slider.
 * Panel A: goal/pit/timeout bars. Panel B: trajectory length histogram.
 * Panel C: state-0 action probs with wall-bumping actions highlighted red.
 * Loads maxent_rollouts.json and alpha_sweep.json.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface RolloutEntry {
  alpha: number;
  goalReachProb: number;
  pitReachProb: number;
  timeoutProb: number;
  meanStepsToTerminal: number;
  lengthHistogram: number[];
}

interface SweepEntry {
  alpha: number;
  pi: number[][];
}

const ACTION_NAMES = ["↑ Up", "→ Right", "↓ Down", "← Left"];
// At state 0 (corner), Up and Left are wall-bumpers
const WALL_BUMPING = new Set([0, 3]); // action indices

customElements.define(
  "failure-diagnostic",
  class extends HTMLElement {
    private rollouts: RolloutEntry[] = [];
    private sweep: SweepEntry[] = [];

    async connectedCallback() {
      const { panel, body } = createPanel({ id: "failure-diagnostic" });
      this.appendChild(panel);

      const controls = document.createElement("div");
      controls.className = "controls-row";
      controls.innerHTML = `
        <label class="slider-label" style="flex:1">Temperature α
          <input type="range" id="fd-alpha" min="0" max="60" step="1" value="25" style="width:100%">
          <span id="fd-alpha-val" style="font-family:var(--rl-font-mono)">—</span>
        </label>
      `;
      body.appendChild(controls);

      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:12px;flex-wrap:wrap;";
      body.appendChild(row);

      const panelA = document.createElement("div");
      panelA.style.cssText = "flex:1;min-width:180px;";
      const panelB = document.createElement("div");
      panelB.style.cssText = "flex:1;min-width:180px;";
      const panelC = document.createElement("div");
      panelC.style.cssText = "flex:1;min-width:180px;";
      row.append(panelA, panelB, panelC);

      try {
        const [r, s] = await Promise.all([
          fetch("/data/maxent/maxent_rollouts.json").then(r => r.json()),
          fetch("/data/maxent/alpha_sweep.json").then(r => r.json()),
        ]);
        this.rollouts = r;
        this.sweep = s;
      } catch {
        body.textContent = "Data not available.";
        return;
      }

      const slider = body.querySelector("#fd-alpha") as HTMLInputElement;
      slider.max = String(this.rollouts.length - 1);
      const alphaLabel = body.querySelector("#fd-alpha-val") as HTMLElement;

      const W = 180, H = 220;
      const margin = { top: 28, right: 12, bottom: 36, left: 48 };
      const iW = W - margin.left - margin.right;
      const iH = H - margin.top - margin.bottom;

      const makeSvg = (container: HTMLElement) =>
        d3.select(container).append("svg")
          .attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").classed("rl-svg", true)
          .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      const svgA = makeSvg(panelA);
      const svgB = makeSvg(panelB);
      const svgC = makeSvg(panelC);

      const draw = (idx: number) => {
        const r = this.rollouts[idx];
        const s = this.sweep[idx];
        alphaLabel.textContent = `α = ${r.alpha.toFixed(4)}`;

        // Panel A: Outcome bars
        svgA.selectAll("*").remove();
        svgA.append("text").attr("x", iW / 2).attr("y", -10).attr("text-anchor", "middle")
          .attr("class", "chart-title").style("font-size", "11px").attr("fill", "var(--rl-ink)")
          .text("Outcomes");

        const outcomes = [
          { label: "Goal", value: r.goalReachProb, color: "var(--maxent-hard)" },
          { label: "Timeout", value: r.timeoutProb, color: "var(--maxent-failure)" },
          { label: "Pit", value: r.pitReachProb, color: "var(--maxent-uniform)" },
        ];
        const yA = d3.scaleBand().domain(outcomes.map(o => o.label)).range([0, iH]).padding(0.25);
        const xA = d3.scaleLinear().domain([0, 1]).range([0, iW]);
        svgA.append("g").call(d3.axisBottom(xA).ticks(4).tickSize(3))
          .attr("transform", `translate(0,${iH})`).selectAll("text").style("font-size", "8px");
        svgA.append("g").call(d3.axisLeft(yA).tickSize(3)).selectAll("text").style("font-size", "9px");
        outcomes.forEach(o => {
          svgA.append("rect").attr("y", yA(o.label)!).attr("x", 0)
            .attr("height", yA.bandwidth()).attr("width", xA(o.value))
            .attr("fill", o.color).attr("opacity", 0.75);
          svgA.append("text").attr("x", xA(o.value) + 3).attr("y", yA(o.label)! + yA.bandwidth() / 2 + 4)
            .style("font-size", "9px").attr("fill", "var(--rl-ink)")
            .text((o.value * 100).toFixed(1) + "%");
        });

        // Panel B: Length histogram
        svgB.selectAll("*").remove();
        svgB.append("text").attr("x", iW / 2).attr("y", -10).attr("text-anchor", "middle")
          .attr("class", "chart-title").style("font-size", "11px").attr("fill", "var(--rl-ink)")
          .text("Step histogram");

        const hist = r.lengthHistogram;
        const xB = d3.scaleLinear().domain([0, hist.length]).range([0, iW]);
        const yB = d3.scaleLinear().domain([0, d3.max(hist)! * 1.1 + 1e-6]).range([iH, 0]);
        svgB.append("g").attr("transform", `translate(0,${iH})`)
          .call(d3.axisBottom(xB).ticks(4).tickFormat(d => String(+d * 10)).tickSize(3))
          .selectAll("text").style("font-size", "8px");
        svgB.append("g").call(d3.axisLeft(yB).ticks(4).tickSize(3)).selectAll("text").style("font-size", "8px");
        svgB.append("text").attr("x", iW / 2).attr("y", iH + 30).attr("text-anchor", "middle")
          .style("font-size", "9px").attr("fill", "var(--rl-ink-muted)").text("Steps (×10)");

        hist.forEach((v, i) => {
          svgB.append("rect")
            .attr("x", xB(i)).attr("y", yB(v))
            .attr("width", Math.max(1, xB(1) - xB(0) - 0.5)).attr("height", iH - yB(v))
            .attr("fill", "var(--maxent-soft)").attr("opacity", 0.65);
        });

        // Panel C: Action probs at state 0
        svgC.selectAll("*").remove();
        svgC.append("text").attr("x", iW / 2).attr("y", -10).attr("text-anchor", "middle")
          .attr("class", "chart-title").style("font-size", "11px").attr("fill", "var(--rl-ink)")
          .text("π*(·|start)");

        const pi0 = s.pi[0];
        const yC = d3.scaleBand().domain(ACTION_NAMES).range([0, iH]).padding(0.2);
        const xC = d3.scaleLinear().domain([0, 1]).range([0, iW]);
        svgC.append("g").call(d3.axisBottom(xC).ticks(4).tickSize(3))
          .attr("transform", `translate(0,${iH})`).selectAll("text").style("font-size", "8px");
        svgC.append("g").call(d3.axisLeft(yC).tickSize(3)).selectAll("text").style("font-size", "9px");

        ACTION_NAMES.forEach((name, ai) => {
          const isWall = WALL_BUMPING.has(ai);
          const maxProb = Math.max(...pi0);
          const isHighest = Math.abs(pi0[ai] - maxProb) < 0.001;
          const color = isWall && isHighest
            ? "var(--maxent-failure)"
            : isWall
              ? "var(--maxent-failure-tint)"
              : "var(--maxent-soft)";
          svgC.append("rect")
            .attr("y", yC(name)!).attr("x", 0)
            .attr("height", yC.bandwidth()).attr("width", xC(pi0[ai]))
            .attr("fill", color).attr("opacity", 0.8);
          svgC.append("text").attr("x", xC(pi0[ai]) + 3).attr("y", yC(name)! + yC.bandwidth() / 2 + 4)
            .style("font-size", "9px").attr("fill", isWall ? "var(--maxent-failure)" : "var(--rl-ink)")
            .text(pi0[ai].toFixed(3));
        });

        // Wall-bumper label
        if (r.alpha > 0.08) {
          svgC.append("text").attr("x", iW).attr("y", iH + 30).attr("text-anchor", "end")
            .style("font-size", "8px").attr("fill", "var(--maxent-failure)")
            .text("↑/← = wall-bumpers");
        }
      };

      slider.addEventListener("input", () => draw(+slider.value));
      draw(+slider.value);
    }
  },
);
