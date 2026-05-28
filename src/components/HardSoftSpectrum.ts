/**
 * V1 — Hard/Soft Spectrum. Plots V^pi(start) vs alpha (the true expected return,
 * no entropy bonus) from alpha_sweep.json. Shows the softmax cap from L10 as a
 * pink reference line at V^pi=0.722, alpha≈0.02.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface SweepEntry {
  alpha: number;
  V_pi: number[];
}

const START = 0;

customElements.define(
  "hard-soft-spectrum",
  class extends HTMLElement {
    async connectedCallback() {
      const { panel, body } = createPanel({ id: "hard-soft-spectrum" });
      this.appendChild(panel);

      let data: SweepEntry[];
      try {
        const res = await fetch("/data/maxent/alpha_sweep.json");
        data = await res.json();
      } catch {
        body.textContent = "Data not available.";
        return;
      }

      const W = 540, H = 280;
      const margin = { top: 24, right: 24, bottom: 48, left: 54 };
      const iW = W - margin.left - margin.right;
      const iH = H - margin.top - margin.bottom;

      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      body.appendChild(wrap);

      const svg = d3.select(wrap)
        .append("svg")
        .attr("viewBox", `0 0 ${W} ${H}`)
        .attr("width", "100%")
        .classed("rl-svg", true)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const alphas = data.map(d => d.alpha);

      const xScale = d3.scaleLog().domain([d3.min(alphas)!, d3.max(alphas)!]).range([0, iW]);
      const yScale = d3.scaleLinear().domain([-0.05, 0.76]).range([iH, 0]);

      // Shading: "useful" band (violet tint) between alpha=0 and 0.1
      // "failure" band (red tint) above alpha=0.1
      const failureX = xScale(0.1);
      svg.append("rect")
        .attr("x", 0).attr("y", 0).attr("width", failureX).attr("height", iH)
        .attr("fill", "var(--maxent-soft-tint)").attr("opacity", 0.35);
      svg.append("rect")
        .attr("x", failureX).attr("y", 0).attr("width", iW - failureX).attr("height", iH)
        .attr("fill", "var(--maxent-failure-tint)").attr("opacity", 0.35);

      // Annotations
      svg.append("text").attr("x", failureX / 2).attr("y", 12)
        .attr("text-anchor", "middle").attr("class", "annot")
        .attr("fill", "var(--maxent-soft)").text("Useful");
      svg.append("text").attr("x", failureX + (iW - failureX) / 2).attr("y", 12)
        .attr("text-anchor", "middle").attr("class", "annot")
        .attr("fill", "var(--maxent-failure)").text("Failure");

      // Axes
      svg.append("g").attr("transform", `translate(0,${iH})`)
        .call(d3.axisBottom(xScale).ticks(6, ".3~g").tickSize(4))
        .selectAll("text").style("font-family", "var(--rl-font-mono)").style("font-size", "10px");
      svg.append("g").call(d3.axisLeft(yScale).ticks(6).tickSize(4))
        .selectAll("text").style("font-family", "var(--rl-font-mono)").style("font-size", "10px");

      svg.append("text").attr("x", iW / 2).attr("y", iH + 38)
        .attr("text-anchor", "middle").attr("class", "axis-label").text("Temperature α (log scale)");
      svg.append("text").attr("transform", "rotate(-90)").attr("y", -42).attr("x", -iH / 2)
        .attr("text-anchor", "middle").attr("class", "axis-label").text("Vᵯⁿ(start)");

      // V* reference (green dashed)
      const vStar = 0.7290;
      svg.append("line")
        .attr("x1", 0).attr("x2", iW)
        .attr("y1", yScale(vStar)).attr("y2", yScale(vStar))
        .attr("stroke", "var(--maxent-hard)").attr("stroke-width", 1.2)
        .attr("stroke-dasharray", "5 3").attr("opacity", 0.7);
      svg.append("text").attr("x", iW - 4).attr("y", yScale(vStar) - 4)
        .attr("text-anchor", "end").attr("class", "annot")
        .attr("fill", "var(--maxent-hard)").text("V* = 0.729");

      // L10 cap (pink dashed)
      const capV = 0.7217;
      svg.append("line")
        .attr("x1", 0).attr("x2", iW)
        .attr("y1", yScale(capV)).attr("y2", yScale(capV))
        .attr("stroke", "var(--maxent-cap)").attr("stroke-width", 1.2)
        .attr("stroke-dasharray", "3 3").attr("opacity", 0.8);
      svg.append("text").attr("x", 4).attr("y", yScale(capV) - 4)
        .attr("class", "annot").attr("fill", "var(--maxent-cap)").text("L10 cap 0.722");

      // Main curve: V^pi (green)
      const line = d3.line<{ a: number; v: number }>()
        .x(d => xScale(d.a))
        .y(d => yScale(d.v))
        .curve(d3.curveCatmullRom);

      const pts = data.map((d: SweepEntry) => ({ a: d.alpha, v: d.V_pi[START] }));
      svg.append("path")
        .datum(pts)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "var(--maxent-hard)")
        .attr("stroke-width", 2.5);

      // Alpha=0.02 vertical marker
      svg.append("line")
        .attr("x1", xScale(0.02)).attr("x2", xScale(0.02))
        .attr("y1", 0).attr("y2", iH)
        .attr("stroke", "var(--maxent-cap)").attr("stroke-width", 1)
        .attr("stroke-dasharray", "3 2").attr("opacity", 0.6);
      svg.append("text").attr("x", xScale(0.02) + 4).attr("y", iH - 8)
        .attr("class", "annot").attr("fill", "var(--maxent-cap)").text("α=0.02");
    }
  },
);
