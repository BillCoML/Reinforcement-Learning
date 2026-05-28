/**
 * V2 — Objective Surface. Analytic 1-state bandit: J(pi) = pi1*r1 + pi2*r2 + alpha*H(pi).
 * User controls delta_r = r1-r2 and alpha via sliders. Three overlaid curves.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

customElements.define(
  "objective-surface",
  class extends HTMLElement {
    connectedCallback() {
      const { panel, body } = createPanel({ id: "objective-surface" });
      this.appendChild(panel);

      // Controls
      const controls = document.createElement("div");
      controls.className = "controls-row";
      controls.innerHTML = `
        <label class="slider-label">Δr = r₁ − r₂
          <input type="range" id="os-dr" min="-2" max="2" step="0.1" value="1">
          <span id="os-dr-val">1.0</span>
        </label>
        <label class="slider-label">Temperature α
          <input type="range" id="os-alpha" min="0.01" max="1" step="0.01" value="0.2">
          <span id="os-alpha-val">0.20</span>
        </label>
      `;
      body.appendChild(controls);

      const W = 480, H = 280;
      const margin = { top: 24, right: 24, bottom: 48, left: 54 };
      const iW = W - margin.left - margin.right;
      const iH = H - margin.top - margin.bottom;
      const N = 200;

      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      body.appendChild(wrap);

      const root = d3.select(wrap)
        .append("svg")
        .attr("viewBox", `0 0 ${W} ${H}`)
        .attr("width", "100%")
        .classed("rl-svg", true)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, iW]);
      const yScale = d3.scaleLinear().domain([-1.2, 1.5]).range([iH, 0]);

      root.append("g").attr("transform", `translate(0,${iH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSize(4))
        .selectAll("text").style("font-family", "var(--rl-font-mono)").style("font-size", "10px");
      root.append("g").call(d3.axisLeft(yScale).ticks(6).tickSize(4))
        .selectAll("text").style("font-family", "var(--rl-font-mono)").style("font-size", "10px");

      root.append("text").attr("x", iW / 2).attr("y", iH + 38)
        .attr("text-anchor", "middle").attr("class", "axis-label").text("π₁ (probability on action 1)");
      root.append("text").attr("transform", "rotate(-90)").attr("y", -42).attr("x", -iH / 2)
        .attr("text-anchor", "middle").attr("class", "axis-label").text("J(π)");

      // Zero line
      root.append("line").attr("x1", 0).attr("x2", iW)
        .attr("y1", yScale(0)).attr("y2", yScale(0))
        .attr("stroke", "var(--rl-border)").attr("stroke-width", 0.8);

      const lineGen = d3.line<[number, number]>().x(d => xScale(d[0])).y(d => yScale(d[1]));
      const pathReward = root.append("path").attr("fill", "none").attr("stroke", "var(--maxent-hard)").attr("stroke-width", 1.8).attr("stroke-dasharray", "4 3");
      const pathEntropy = root.append("path").attr("fill", "none").attr("stroke", "var(--maxent-entropy)").attr("stroke-width", 1.8).attr("stroke-dasharray", "4 3");
      const pathTotal = root.append("path").attr("fill", "none").attr("stroke", "var(--maxent-soft)").attr("stroke-width", 2.5);
      const dotOptimal = root.append("circle").attr("r", 5).attr("fill", "var(--maxent-soft)");
      const legEl = root.append("g").attr("transform", `translate(${iW - 140},4)`);
      const legData = [
        { color: "var(--maxent-hard)", label: "Reward term", dash: "4 3" },
        { color: "var(--maxent-entropy)", label: "Entropy term (α·H)", dash: "4 3" },
        { color: "var(--maxent-soft)", label: "Total J(π)", dash: "none" },
      ];
      legData.forEach((d, i) => {
        legEl.append("line").attr("x1", 0).attr("x2", 16).attr("y1", i * 16 + 8).attr("y2", i * 16 + 8)
          .attr("stroke", d.color).attr("stroke-width", 1.8).attr("stroke-dasharray", d.dash === "none" ? null : d.dash);
        legEl.append("text").attr("x", 20).attr("y", i * 16 + 12).text(d.label)
          .attr("class", "annot").attr("fill", "var(--rl-ink-muted)");
      });

      function draw(dr: number, alpha: number) {
        const pts = d3.range(N + 1).map(i => i / N);
        const rewardPts: [number, number][] = pts.map(p1 => [p1, p1 * dr]);
        const entropyPts: [number, number][] = pts.map(p1 => {
          const p2 = 1 - p1;
          const H = p1 > 0 && p2 > 0 ? -(p1 * Math.log(p1) + p2 * Math.log(p2)) : 0;
          return [p1, alpha * H];
        });
        const totalPts: [number, number][] = pts.map((p1, i) => [p1, rewardPts[i][1] + entropyPts[i][1]]);

        pathReward.datum(rewardPts).attr("d", lineGen);
        pathEntropy.datum(entropyPts).attr("d", lineGen);
        pathTotal.datum(totalPts).attr("d", lineGen);

        // Optimal p1*: pi1 = exp(r1/alpha) / (exp(r1/alpha) + exp(r2/alpha)) = exp(dr/alpha) / (exp(dr/alpha) + 1)
        const optP1 = Math.exp(dr / alpha) / (Math.exp(dr / alpha) + 1);
        const optJ = optP1 * dr + alpha * (optP1 > 0 && optP1 < 1
          ? -(optP1 * Math.log(optP1) + (1 - optP1) * Math.log(1 - optP1)) : 0);
        dotOptimal.attr("cx", xScale(Math.min(1, Math.max(0, optP1)))).attr("cy", yScale(optJ));
      }

      const drIn = body.querySelector("#os-dr") as HTMLInputElement;
      const drVal = body.querySelector("#os-dr-val") as HTMLElement;
      const alphaIn = body.querySelector("#os-alpha") as HTMLInputElement;
      const alphaVal = body.querySelector("#os-alpha-val") as HTMLElement;

      function update() {
        const dr = +drIn.value;
        const alpha = +alphaIn.value;
        drVal.textContent = dr.toFixed(1);
        alphaVal.textContent = alpha.toFixed(2);
        draw(dr, alpha);
      }

      drIn.addEventListener("input", update);
      alphaIn.addEventListener("input", update);
      update();
    }
  },
);
