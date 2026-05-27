/**
 * V1 — Model-Free vs Model-Based side-by-side.
 * Static two-column schematic: DP backup (uses P and R) vs MC return (uses τ only).
 * Width 720, Height 280.
 */
import * as d3 from "d3";
import { cssVar } from "./base";

const W = 720, H = 280;
const COL_W = W / 2 - 4;

function arrow(x1: number, y1: number, x2: number, y2: number): string {
  return `M${x1},${y1} L${x2},${y2}`;
}

class ModelFreeVsModelBased extends HTMLElement {
  connectedCallback() {
    const wrap = document.createElement("div");
    wrap.style.cssText = "overflow-x:auto;";
    const svg = d3.select(wrap)
      .append("svg")
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("width", W)
      .attr("style", "max-width:100%;height:auto;display:block;margin:0 auto;")
      .attr("class", "rl-svg");

    const defs = svg.append("defs");
    defs.append("marker").attr("id", "arr").attr("markerWidth", 8).attr("markerHeight", 8)
      .attr("refX", 6).attr("refY", 3).attr("orient", "auto")
      .append("polygon").attr("points", "0 0, 8 3, 0 6").attr("fill", cssVar("--rl-ink") || "#1c1e22");
    defs.append("marker").attr("id", "arr-blue").attr("markerWidth", 8).attr("markerHeight", 8)
      .attr("refX", 6).attr("refY", 3).attr("orient", "auto")
      .append("polygon").attr("points", "0 0, 8 3, 0 6").attr("fill", "#2563eb");
    defs.append("marker").attr("id", "arr-green").attr("markerWidth", 8).attr("markerHeight", 8)
      .attr("refX", 6).attr("refY", 3).attr("orient", "auto")
      .append("polygon").attr("points", "0 0, 8 3, 0 6").attr("fill", "#15803d");

    const ink = cssVar("--rl-ink") || "#1c1e22";
    const surface = cssVar("--rl-surface") || "#f8f6f0";
    const border = cssVar("--rl-border") || "#d6d0c8";
    const mono = `"JetBrains Mono", monospace`;

    // ── Left panel: Model-Based (DP) ──────────────────────────────────────────
    const left = svg.append("g");
    const lx = 16;

    // Panel background
    left.append("rect").attr("x", lx).attr("y", 8).attr("width", COL_W - 24).attr("height", H - 16)
      .attr("rx", 8).attr("fill", surface).attr("stroke", border).attr("stroke-width", 1.5);

    // Header
    left.append("text").attr("x", lx + 12).attr("y", 34).attr("fill", ink)
      .attr("font-family", mono).attr("font-size", 12).attr("font-weight", 600)
      .text("Model-Based (DP)");

    // "Knows:" line with inputs
    left.append("text").attr("x", lx + 12).attr("y", 56).attr("fill", "#6b5e4e")
      .attr("font-family", mono).attr("font-size", 10).text("inputs:");

    // P box
    left.append("rect").attr("x", lx + 58).attr("y", 43).attr("width", 88).attr("height", 20)
      .attr("rx", 3).attr("fill", "#eff6ff").attr("stroke", "#2563eb").attr("stroke-width", 1.5);
    left.append("text").attr("x", lx + 102).attr("y", 57).attr("fill", "#2563eb")
      .attr("font-family", mono).attr("font-size", 10).attr("text-anchor", "middle").text("P(s′|s,a)");

    // R box
    left.append("rect").attr("x", lx + 154).attr("y", 43).attr("width", 80).attr("height", 20)
      .attr("rx", 3).attr("fill", "#f0fdf4").attr("stroke", "#15803d").attr("stroke-width", 1.5);
    left.append("text").attr("x", lx + 194).attr("y", 57).attr("fill", "#15803d")
      .attr("font-family", mono).attr("font-size", 10).attr("text-anchor", "middle").text("R(s,a)");

    // Arrows down from P and R to formula box
    left.append("path").attr("d", arrow(lx + 102, 64, lx + 102, 86))
      .attr("stroke", "#2563eb").attr("stroke-width", 1.5).attr("fill", "none").attr("marker-end", "url(#arr-blue)");
    left.append("path").attr("d", arrow(lx + 194, 64, lx + 150, 86))
      .attr("stroke", "#15803d").attr("stroke-width", 1.5).attr("fill", "none").attr("marker-end", "url(#arr-green)");

    // Bellman backup formula box
    const fbox = left.append("g");
    fbox.append("rect").attr("x", lx + 40).attr("y", 88).attr("width", 200).attr("height", 50)
      .attr("rx", 5).attr("fill", "#fff8f0").attr("stroke", "#b45309").attr("stroke-width", 1.5);
    fbox.append("text").attr("x", lx + 140).attr("y", 107).attr("fill", "#b45309")
      .attr("font-family", mono).attr("font-size", 9.5).attr("font-weight", 600).attr("text-anchor", "middle")
      .text("Bellman Expectation Backup");
    fbox.append("text").attr("x", lx + 140).attr("y", 127).attr("fill", ink)
      .attr("font-family", mono).attr("font-size", 9).attr("text-anchor", "middle")
      .text("V(s) ← Σ P(s′|s,a)[R + γ V(s′)]");

    // Arrow from formula to result
    left.append("path").attr("d", arrow(lx + 140, 139, lx + 140, 162))
      .attr("stroke", ink).attr("stroke-width", 1.5).attr("fill", "none").attr("marker-end", "url(#arr)");

    // Result box
    left.append("rect").attr("x", lx + 82).attr("y", 163).attr("width", 116).attr("height", 26)
      .attr("rx", 4).attr("fill", "#f0fdf4").attr("stroke", "#15803d").attr("stroke-width", 1.5);
    left.append("text").attr("x", lx + 140).attr("y", 181).attr("fill", "#15803d")
      .attr("font-family", mono).attr("font-size", 10).attr("text-anchor", "middle").text("V̂(s) for all s");

    // Tagline
    left.append("text").attr("x", lx + 140).attr("y", 220).attr("fill", "#6b5e4e")
      .attr("font-family", mono).attr("font-size", 9).attr("text-anchor", "middle")
      .text("exact (requires full model)");

    // ── Right panel: Model-Free (MC) ──────────────────────────────────────────
    const rx = W / 2 + 4;
    const right = svg.append("g");

    right.append("rect").attr("x", rx).attr("y", 8).attr("width", COL_W - 24).attr("height", H - 16)
      .attr("rx", 8).attr("fill", surface).attr("stroke", border).attr("stroke-width", 1.5);

    right.append("text").attr("x", rx + 12).attr("y", 34).attr("fill", ink)
      .attr("font-family", mono).attr("font-size", 12).attr("font-weight", 600)
      .text("Model-Free (MC)");

    right.append("text").attr("x", rx + 12).attr("y", 56).attr("fill", "#6b5e4e")
      .attr("font-family", mono).attr("font-size", 10).text("input:");

    // Trajectory box
    right.append("rect").attr("x", rx + 54).attr("y", 43).attr("width", 210).attr("height", 20)
      .attr("rx", 3).attr("fill", "#fff7ed").attr("stroke", "#ea580c").attr("stroke-width", 1.5);
    right.append("text").attr("x", rx + 159).attr("y", 57).attr("fill", "#ea580c")
      .attr("font-family", mono).attr("font-size", 9.5).attr("text-anchor", "middle")
      .text("τ = (s₀, a₀, r₁, s₁, a₁, r₂, …, sᴛ)");

    // Arrow down
    right.append("path").attr("d", arrow(rx + 159, 64, rx + 159, 86))
      .attr("stroke", "#ea580c").attr("stroke-width", 1.5).attr("fill", "none").attr("marker-end", "url(#arr)");

    // Return computation box
    const rbox = right.append("g");
    rbox.append("rect").attr("x", rx + 40).attr("y", 88).attr("width", 238).attr("height", 50)
      .attr("rx", 5).attr("fill", "#fff8f0").attr("stroke", "#b45309").attr("stroke-width", 1.5);
    rbox.append("text").attr("x", rx + 159).attr("y", 107).attr("fill", "#b45309")
      .attr("font-family", mono).attr("font-size", 9.5).attr("font-weight", 600).attr("text-anchor", "middle")
      .text("Compute Return + Incremental Average");
    rbox.append("text").attr("x", rx + 159).attr("y", 127).attr("fill", ink)
      .attr("font-family", mono).attr("font-size", 9).attr("text-anchor", "middle")
      .text("G₀ = Σ γᵗ rₜ₊₁  →  V̂(s₀) += (G₀ − V̂(s₀))/N");

    // Arrow down
    right.append("path").attr("d", arrow(rx + 159, 139, rx + 159, 162))
      .attr("stroke", ink).attr("stroke-width", 1.5).attr("fill", "none").attr("marker-end", "url(#arr)");

    // Result box
    right.append("rect").attr("x", rx + 101).attr("y", 163).attr("width", 116).attr("height", 26)
      .attr("rx", 4).attr("fill", "#f0fdf4").attr("stroke", "#15803d").attr("stroke-width", 1.5);
    right.append("text").attr("x", rx + 159).attr("y", 181).attr("fill", "#15803d")
      .attr("font-family", mono).attr("font-size", 10).attr("text-anchor", "middle").text("V̂(s) for all s");

    right.append("text").attr("x", rx + 159).attr("y", 220).attr("fill", "#6b5e4e")
      .attr("font-family", mono).attr("font-size", 9).attr("text-anchor", "middle")
      .text("unbiased — no model needed");

    // Divider
    svg.append("line").attr("x1", W / 2).attr("y1", 24).attr("x2", W / 2).attr("y2", H - 24)
      .attr("stroke", border).attr("stroke-width", 1).attr("stroke-dasharray", "4,4");

    this.appendChild(wrap);
  }
}
customElements.define("model-free-vs-model-based", ModelFreeVsModelBased);
