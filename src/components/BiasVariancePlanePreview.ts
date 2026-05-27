/**
 * V6 — Bias-Variance Plane Preview.
 * Static schematic: MC at (bias=0, high variance), TD(0) at (moderate bias,
 * low variance), n-step TD as a smooth arc between them.
 * Width 540, Height 320.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 540, H = 320;
const M = { top: 28, right: 100, bottom: 48, left: 60 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;

// n-step TD trade-off curve: as n increases, bias decreases and variance grows.
// Parameterized by n ∈ [1, ∞], x=bias, y=variance.
// n=1: TD(0) → (0.28, 0.10)  n=∞: MC → (0, 1.0)
function curvePoint(t: number): [number, number] {
  // t ∈ [0,1] where t=0 → TD(0), t=1 → MC
  const bias = 0.28 * (1 - t);
  const variance = 0.10 + 0.90 * t * t;
  return [bias, variance];
}

const MARKERS: Array<{ label: string; t: number; dx: number; dy: number; color: string }> = [
  { label: "TD(0)  n=1", t: 0.0, dx: 8,   dy: -14, color: "var(--mc-off-policy)" },
  { label: "2-step",     t: 0.2, dx: 8,   dy: -10, color: "var(--rl-ink-muted)" },
  { label: "4-step",     t: 0.4, dx: 8,   dy: -10, color: "var(--rl-ink-muted)" },
  { label: "8-step",     t: 0.6, dx: 8,   dy: -10, color: "var(--rl-ink-muted)" },
  { label: "MC  n=∞",   t: 1.0, dx: 8,   dy:  12, color: "var(--mc-on-policy)" },
];

class BiasVariancePlanePreview extends HTMLElement {
  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "bias-variance-plane" });

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.style.height = "auto";
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);
    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    const xScale = d3.scaleLinear().domain([0, 0.38]).range([0, PW]);
    const yScale = d3.scaleLinear().domain([0, 1.15]).range([PH, 0]);

    // Light grid
    g.append("g").call(
      d3.axisLeft(yScale).ticks(4).tickSize(-PW).tickFormat(() => "")
    ).call(g2 => g2.selectAll(".tick line").attr("stroke", "var(--rl-border)").attr("opacity", 0.4))
      .call(g2 => g2.select(".domain").remove());
    g.append("g").attr("transform", `translate(0,${PH})`).call(
      d3.axisBottom(xScale).ticks(4).tickSize(-PH).tickFormat(() => "")
    ).call(g2 => g2.selectAll(".tick line").attr("stroke", "var(--rl-border)").attr("opacity", 0.4))
      .call(g2 => g2.select(".domain").remove());

    // Axes
    g.append("g").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xScale).ticks(4).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    g.append("g").call(d3.axisLeft(yScale).ticks(4).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    // Axis labels
    g.append("text").attr("x", PW / 2).attr("y", PH + 36)
      .attr("text-anchor", "middle").attr("font-size", 10.5)
      .attr("fill", "var(--rl-ink-muted)").text("Bias  →");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -PH / 2).attr("y", -44)
      .attr("text-anchor", "middle").attr("font-size", 10.5)
      .attr("fill", "var(--rl-ink-muted)").text("Variance  →");

    // Trade-off curve
    const ts = d3.range(0, 1.001, 0.02);
    const curveLine = d3.line<number>()
      .x(t => xScale(curvePoint(t)[0]))
      .y(t => yScale(curvePoint(t)[1]))
      .curve(d3.curveCatmullRom);

    g.append("path").datum(ts)
      .attr("d", curveLine)
      .attr("fill", "none")
      .attr("stroke", "var(--rl-ink-muted)")
      .attr("stroke-width", 1.8)
      .attr("stroke-dasharray", "none")
      .attr("opacity", 0.45);

    // Shaded region under curve (ideal zone is bottom-left)
    g.append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("width", xScale(0.10)).attr("height", yScale(0.30))
      .attr("fill", "var(--mc-on-policy)").attr("opacity", 0.06);
    g.append("text").attr("x", xScale(0.05)).attr("y", yScale(0.38))
      .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "var(--mc-on-policy)")
      .attr("opacity", 0.5).text("ideal");

    // "Increasing n →" diagonal label
    g.append("text")
      .attr("x", xScale(0.20)).attr("y", yScale(0.42))
      .attr("text-anchor", "middle").attr("font-size", 9)
      .attr("fill", "var(--rl-ink-muted)").attr("transform",
        `rotate(-38,${xScale(0.20)},${yScale(0.42)})`)
      .text("n-step TD (increasing n →)");

    // Points and labels
    for (const mk of MARKERS) {
      const [bx, bv] = curvePoint(mk.t);
      const px = xScale(bx);
      const py = yScale(bv);
      const isMajor = mk.t === 0 || mk.t === 1;

      g.append("circle")
        .attr("cx", px).attr("cy", py)
        .attr("r", isMajor ? 5 : 3.5)
        .attr("fill", mk.color)
        .attr("opacity", isMajor ? 1 : 0.65);

      g.append("text")
        .attr("x", px + mk.dx).attr("y", py + mk.dy)
        .attr("font-size", isMajor ? 10.5 : 9)
        .attr("font-weight", isMajor ? "600" : "400")
        .attr("fill", mk.color)
        .attr("opacity", isMajor ? 1 : 0.65)
        .text(mk.label);
    }

    // Annotations for MC and TD
    const [mcBias, mcVar] = curvePoint(1);
    g.append("text")
      .attr("x", xScale(mcBias) + 10).attr("y", yScale(mcVar) + 26)
      .attr("font-size", 9).attr("fill", "var(--mc-on-policy)").attr("opacity", 0.75)
      .text("zero bias, full-return variance");

    const [tdBias, tdVar] = curvePoint(0);
    g.append("text")
      .attr("x", xScale(tdBias) + 10).attr("y", yScale(tdVar) + 26)
      .attr("font-size", 9).attr("fill", "var(--mc-off-policy)").attr("opacity", 0.75)
      .text("bootstrap bias, one-step variance");

    // Title
    svg.append("text")
      .attr("x", M.left + PW / 2).attr("y", 16)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", "600")
      .attr("font-family", "var(--rl-font-ui)")
      .attr("fill", "var(--rl-ink)")
      .text("Bias–Variance trade-off: MC vs TD");
  }
}

customElements.define("bias-variance-plane-preview", BiasVariancePlanePreview);
