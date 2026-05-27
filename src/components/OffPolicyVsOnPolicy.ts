/**
 * V5 — Off-Policy vs On-Policy MC Comparison.
 * Side-by-side bar chart (mean ± SD) for N ∈ {100, 1000, 10000}.
 * Left: on-policy MC, right: off-policy weighted IS.
 * Data from /data/mc/*.json (pre-computed, 50 trials each).
 * Width 680, Height 300.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 680, H = 300;
const M = { top: 28, right: 16, bottom: 40, left: 52 };
const PW = W / 2 - M.left - M.right - 8;
const PH = H - M.top - M.bottom;

const NS_SVG = "http://www.w3.org/2000/svg";

class OffPolicyVsOnPolicy extends HTMLElement {
  connectedCallback() { this.build(); }

  private async build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "off-vs-on-policy" });

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svgEl = document.createElementNS(NS_SVG, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.style.height = "auto";
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    try {
      const [onData, offData] = await Promise.all([
        fetch("/data/mc/on_policy_convergence.json").then(r => r.json()),
        fetch("/data/mc/off_policy_gridworld.json").then(r => r.json()),
      ]);
      this.draw(svgEl, onData, offData);
      setStatus("on-policy vs off-policy · 50 trials · N ∈ {100, 1000, 10000}");
    } catch {
      d3.select(svgEl).append("text").attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle").attr("font-size", 11)
        .attr("fill", "var(--rl-ink-muted)").text("Data unavailable — run dev server.");
    }
  }

  private draw(
    svgEl: SVGSVGElement,
    onData: { trueValue: number; rows: { N: number; mean: number; std: number }[] },
    offData: { trueValue: number; rows: { N: number; weightedMean: number; weightedSD: number; ordinaryMean: number; ordinarySD: number }[] },
  ) {
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    const Ns = onData.rows.map((r: { N: number }) => r.N);
    const xScale = d3.scaleBand().domain(Ns.map(String)).range([0, PW]).padding(0.3);
    const bw = xScale.bandwidth();

    // ── Left: On-policy ──────────────────────────────────────────────────────
    const trueOn = onData.trueValue;
    const onYMin = trueOn - 0.22;
    const onYMax = trueOn + 0.22;
    const yLeft = d3.scaleLinear().domain([onYMin, onYMax]).range([PH, 0]).nice();

    const gLeft = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    gLeft.append("g").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xScale).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    gLeft.append("g")
      .call(d3.axisLeft(yLeft).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    gLeft.append("text").attr("x", PW / 2).attr("y", PH + 30)
      .attr("text-anchor", "middle").attr("font-size", 9.5)
      .attr("fill", "var(--rl-ink-muted)").text("N (episodes)");

    // True value line
    gLeft.append("line")
      .attr("x1", 0).attr("x2", PW)
      .attr("y1", yLeft(trueOn)).attr("y2", yLeft(trueOn))
      .attr("stroke", "var(--mc-truth)").attr("stroke-width", 1.1)
      .attr("stroke-dasharray", "5,3").attr("opacity", 0.5);
    gLeft.append("text").attr("x", PW - 2).attr("y", yLeft(trueOn) - 4)
      .attr("text-anchor", "end").attr("font-size", 8.5)
      .attr("fill", "var(--mc-truth)").attr("opacity", 0.6).text(`true ${trueOn}`);

    // Bars + error bars
    for (const row of onData.rows) {
      const x = (xScale(String(row.N)) ?? 0) + bw / 2;
      const yMid = yLeft(row.mean);
      const yHi = yLeft(row.mean + row.std);
      const yLo = yLeft(row.mean - row.std);
      gLeft.append("rect")
        .attr("x", x - bw / 2).attr("y", yMid)
        .attr("width", bw).attr("height", Math.abs(yMid - yLeft(trueOn)) || 2)
        .attr("fill", row.mean < trueOn ? "var(--mc-on-policy)" : "var(--mc-on-policy)")
        .attr("opacity", 0.55);

      // bar from 0 to mean
      const y0 = yLeft(trueOn);
      gLeft.append("rect")
        .attr("x", x - bw * 0.35).attr("y", Math.min(y0, yMid))
        .attr("width", bw * 0.7).attr("height", Math.abs(y0 - yMid) || 2)
        .attr("fill", "var(--mc-on-policy)").attr("opacity", 0.65);

      // Error bar (±1 SD)
      gLeft.append("line")
        .attr("x1", x).attr("x2", x).attr("y1", yHi).attr("y2", yLo)
        .attr("stroke", "var(--mc-on-policy)").attr("stroke-width", 1.5);
      gLeft.append("line").attr("x1", x - 4).attr("x2", x + 4).attr("y1", yHi).attr("y2", yHi)
        .attr("stroke", "var(--mc-on-policy)").attr("stroke-width", 1.5);
      gLeft.append("line").attr("x1", x - 4).attr("x2", x + 4).attr("y1", yLo).attr("y2", yLo)
        .attr("stroke", "var(--mc-on-policy)").attr("stroke-width", 1.5);
    }

    gLeft.append("text").attr("x", PW / 2).attr("y", -10)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", "600")
      .attr("fill", "var(--mc-on-policy)").text("On-policy MC (uniform random)");

    // ── Right: Off-policy ────────────────────────────────────────────────────
    const trueOff = offData.trueValue;
    const offYMin = trueOff - 0.55;
    const offYMax = trueOff + 0.55;
    const yRight = d3.scaleLinear().domain([offYMin, offYMax]).range([PH, 0]).nice();

    const offsetX = W / 2 + 8;
    const gRight = svg.append("g").attr("transform", `translate(${offsetX + M.left},${M.top})`);

    gRight.append("g").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xScale).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    gRight.append("g")
      .call(d3.axisLeft(yRight).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    gRight.append("text").attr("x", PW / 2).attr("y", PH + 30)
      .attr("text-anchor", "middle").attr("font-size", 9.5)
      .attr("fill", "var(--rl-ink-muted)").text("N (episodes)");

    // True value line
    gRight.append("line")
      .attr("x1", 0).attr("x2", PW)
      .attr("y1", yRight(trueOff)).attr("y2", yRight(trueOff))
      .attr("stroke", "var(--mc-truth)").attr("stroke-width", 1.1)
      .attr("stroke-dasharray", "5,3").attr("opacity", 0.5);
    gRight.append("text").attr("x", PW - 2).attr("y", yRight(trueOff) - 4)
      .attr("text-anchor", "end").attr("font-size", 8.5)
      .attr("fill", "var(--mc-truth)").attr("opacity", 0.6).text(`true ${trueOff}`);

    // Bars for weighted IS
    for (const row of offData.rows) {
      const x = (xScale(String(row.N)) ?? 0) + bw / 2;
      const wMean = row.weightedMean;
      const wSD = row.weightedSD;
      const yMid = yRight(wMean);
      const yHi = yRight(wMean + wSD);
      const yLo = yRight(wMean - wSD);
      const y0 = yRight(trueOff);

      gRight.append("rect")
        .attr("x", x - bw * 0.35).attr("y", Math.min(y0, yMid))
        .attr("width", bw * 0.7).attr("height", Math.abs(y0 - yMid) || 2)
        .attr("fill", "var(--mc-off-policy)").attr("opacity", 0.65);

      gRight.append("line")
        .attr("x1", x).attr("x2", x).attr("y1", yHi).attr("y2", yLo)
        .attr("stroke", "var(--mc-off-policy)").attr("stroke-width", 1.5);
      gRight.append("line").attr("x1", x - 4).attr("x2", x + 4).attr("y1", yHi).attr("y2", yHi)
        .attr("stroke", "var(--mc-off-policy)").attr("stroke-width", 1.5);
      gRight.append("line").attr("x1", x - 4).attr("x2", x + 4).attr("y1", yLo).attr("y2", yLo)
        .attr("stroke", "var(--mc-off-policy)").attr("stroke-width", 1.5);
    }

    gRight.append("text").attr("x", PW / 2).attr("y", -10)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", "600")
      .attr("fill", "var(--mc-off-policy)").text("Off-policy MC (weighted IS, target=π*)");

    // Divider
    svg.append("line")
      .attr("x1", W / 2 + 4).attr("x2", W / 2 + 4)
      .attr("y1", 12).attr("y2", H - 12)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1).attr("stroke-dasharray", "4,3");
  }
}

customElements.define("off-policy-vs-on-policy", OffPolicyVsOnPolicy);
