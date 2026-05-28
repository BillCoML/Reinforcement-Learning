/**
 * V3 — Soft VI Convergence. Left: V_soft per non-terminal state vs iteration.
 * Right: pi(·|s=0) bar chart animating from uniform to Boltzmann.
 * Loads convergence_traces.json; user picks alpha.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface TraceData {
  V: number[][];        // [iter][state]
  pi_start: number[][]; // [iter][action]
}

const NON_TERMINALS = [0, 1, 2, 3, 5, 6, 7]; // all except 4(pit) and 8(goal)
const ACTION_NAMES = ["↑", "→", "↓", "←"];
const STATE_COLORS = d3.schemeTableau10;

customElements.define(
  "soft-vi-convergence",
  class extends HTMLElement {
    async connectedCallback() {
      const { panel, body } = createPanel({ id: "soft-vi-convergence" });
      this.appendChild(panel);

      const controls = document.createElement("div");
      controls.className = "controls-row";
      controls.innerHTML = `
        <label class="slider-label">Temperature α
          <select id="svc-alpha">
            <option value="0.01">0.01</option>
            <option value="0.05" selected>0.05</option>
            <option value="0.1">0.10</option>
            <option value="0.5">0.50</option>
            <option value="1.0">1.00</option>
          </select>
        </label>
        <label class="slider-label">Iteration
          <input type="range" id="svc-iter" min="0" max="30" step="1" value="30">
          <span id="svc-iter-val">30</span>
        </label>
      `;
      body.appendChild(controls);

      const chartsRow = document.createElement("div");
      chartsRow.style.cssText = "display:flex;gap:12px;";
      body.appendChild(chartsRow);

      let allData: Record<string, TraceData>;
      try {
        const res = await fetch("/data/maxent/convergence_traces.json");
        allData = await res.json();
      } catch {
        body.textContent = "Data not available.";
        return;
      }

      // Left chart: convergence curves
      const leftSvg = d3.select(chartsRow).append("svg")
        .attr("viewBox", "0 0 300 240").attr("width", "50%").classed("rl-svg", true);
      const lm = { top: 24, right: 16, bottom: 40, left: 48 };
      const lW = 300 - lm.left - lm.right;
      const lH = 240 - lm.top - lm.bottom;
      const lg = leftSvg.append("g").attr("transform", `translate(${lm.left},${lm.top})`);

      // Right chart: policy bar chart
      const rightSvg = d3.select(chartsRow).append("svg")
        .attr("viewBox", "0 0 220 240").attr("width", "40%").classed("rl-svg", true);
      const rm = { top: 24, right: 16, bottom: 40, left: 48 };
      const rW = 220 - rm.left - rm.right;
      const rH = 240 - rm.top - rm.bottom;
      const rg = rightSvg.append("g").attr("transform", `translate(${rm.left},${rm.top})`);

      const alphaSelect = body.querySelector("#svc-alpha") as HTMLSelectElement;
      const iterSlider = body.querySelector("#svc-iter") as HTMLInputElement;
      const iterVal = body.querySelector("#svc-iter-val") as HTMLElement;

      function draw(alphaStr: string, iter: number) {
        const trace = allData[alphaStr];
        if (!trace) return;
        const maxIter = trace.V.length - 1;
        iter = Math.min(iter, maxIter);

        lg.selectAll("*").remove();
        rg.selectAll("*").remove();

        // Left: V convergence
        const xL = d3.scaleLinear().domain([0, maxIter]).range([0, lW]);
        const allVs = trace.V.flatMap(v => NON_TERMINALS.map(s => v[s]));
        const yL = d3.scaleLinear().domain([d3.min(allVs)! - 0.05, d3.max(allVs)! + 0.05]).range([lH, 0]);

        lg.append("g").attr("transform", `translate(0,${lH})`)
          .call(d3.axisBottom(xL).ticks(5).tickSize(3))
          .selectAll("text").style("font-size", "9px");
        lg.append("g").call(d3.axisLeft(yL).ticks(5).tickSize(3))
          .selectAll("text").style("font-size", "9px");
        lg.append("text").attr("x", lW / 2).attr("y", lH + 34).attr("text-anchor", "middle")
          .attr("class", "axis-label").style("font-size", "10px").text("Iteration");
        lg.append("text").attr("class", "chart-title").attr("x", lW / 2).attr("y", -8)
          .attr("text-anchor", "middle").style("font-size", "11px")
          .attr("fill", "var(--rl-ink)").text("V_soft per state");

        NON_TERMINALS.forEach((s, ci) => {
          const pts = trace.V.slice(0, iter + 1).map(v => [v[s]]);
          lg.append("path")
            .datum(pts)
            .attr("fill", "none")
            .attr("stroke", STATE_COLORS[ci % STATE_COLORS.length])
            .attr("stroke-width", 1.4)
            .attr("d", d3.line<number[]>().x((_, i) => xL(i)).y(d => yL(d[0])));
        });

        // Iter marker line
        if (iter < maxIter) {
          lg.append("line").attr("x1", xL(iter)).attr("x2", xL(iter))
            .attr("y1", 0).attr("y2", lH)
            .attr("stroke", "var(--rl-ink-muted)").attr("stroke-dasharray", "3 2").attr("stroke-width", 1);
        }

        // Right: policy bar chart at current iter
        const piCur = trace.pi_start[Math.min(iter, trace.pi_start.length - 1)];
        const xR = d3.scaleBand().domain(ACTION_NAMES).range([0, rW]).padding(0.2);
        const yR = d3.scaleLinear().domain([0, 1]).range([rH, 0]);

        rg.append("g").attr("transform", `translate(0,${rH})`)
          .call(d3.axisBottom(xR).tickSize(3))
          .selectAll("text").style("font-size", "11px");
        rg.append("g").call(d3.axisLeft(yR).ticks(5).tickSize(3))
          .selectAll("text").style("font-size", "9px");
        rg.append("text").attr("class", "chart-title").attr("x", rW / 2).attr("y", -8)
          .attr("text-anchor", "middle").style("font-size", "11px")
          .attr("fill", "var(--rl-ink)").text("π*(·|s=start)");

        ACTION_NAMES.forEach((a, ai) => {
          rg.append("rect")
            .attr("x", xR(a)!).attr("y", yR(piCur[ai]))
            .attr("width", xR.bandwidth()).attr("height", rH - yR(piCur[ai]))
            .attr("fill", "var(--maxent-soft)").attr("opacity", 0.75);
          rg.append("text").attr("x", xR(a)! + xR.bandwidth() / 2).attr("y", yR(piCur[ai]) - 3)
            .attr("text-anchor", "middle").style("font-size", "9px").attr("fill", "var(--rl-ink-muted)")
            .text(piCur[ai].toFixed(2));
        });

        // 1/4 uniform reference
        rg.append("line").attr("x1", 0).attr("x2", rW)
          .attr("y1", yR(0.25)).attr("y2", yR(0.25))
          .attr("stroke", "var(--maxent-uniform)").attr("stroke-dasharray", "3 2").attr("stroke-width", 1);
        rg.append("text").attr("x", rW).attr("y", yR(0.25) - 3).attr("text-anchor", "end")
          .style("font-size", "9px").attr("fill", "var(--maxent-uniform)").text("uniform 0.25");
      }

      function update() {
        const iter = +iterSlider.value;
        iterVal.textContent = String(iter);
        draw(alphaSelect.value, iter);
      }

      alphaSelect.addEventListener("change", update);
      iterSlider.addEventListener("input", update);
      update();
    }
  },
);
