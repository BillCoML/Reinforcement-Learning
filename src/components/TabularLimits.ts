/**
 * V1 — Tabular Limits.
 * Three-panel layout: 3×3 gridworld with Q-values, 7×7 gridworld, Atari schematic.
 * Shows the size explosion: 36 → 196 → 10^67914 possible states.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld } from "../mdp/gridworld";
import { idx7, rc7, wallCells7x7 } from "../mdp/gridworld7x7";
import { qLearning } from "../td/q-learning";
import { mulberry32 } from "../importance-sampling/gaussian";
import { makeValueColorScale, textColorOn } from "./value-scale";

const W = 880;
const H = 360;
const CELL = 36;
const GAP = 3;
const PAD = 12;

function gridW(n: number) { return n * CELL + (n - 1) * GAP + 2 * PAD; }

function drawGrid3x3(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  ox: number,
  oy: number,
) {
  const mdp = buildGridworld();
  const result = qLearning(mdp, 3000, 0.1, 0.1, { rng: mulberry32(0) });
  const Q = result.Q;
  const nA = 4;
  const scale = makeValueColorScale([-1, 1]);

  const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
  g.append("text")
    .attr("x", gridW(3) / 2).attr("y", -4)
    .attr("text-anchor", "middle")
    .attr("class", "rl-label")
    .text("3×3 gridworld  |36 Q-values|");

  for (let s = 0; s < 9; s++) {
    const { r, c } = { r: Math.floor(s / 3), c: s % 3 };
    const cx = PAD + c * (CELL + GAP);
    const cy = PAD + r * (CELL + GAP);

    // background
    const maxQ = Math.max(...Array.from({ length: nA }, (_, a) => Q[s * nA + a]));
    const bg = mdp.terminals[s] ? (s === 4 ? "#dc2626" : "#15803d") : scale(maxQ);
    g.append("rect").attr("x", cx).attr("y", cy)
      .attr("width", CELL).attr("height", CELL)
      .attr("rx", 3).attr("fill", bg).attr("stroke", "#e5e7eb").attr("stroke-width", 1);

    if (!mdp.terminals[s]) {
      // show best Q value
      g.append("text")
        .attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("font-size", "9px")
        .attr("fill", textColorOn(bg))
        .text(maxQ.toFixed(2));
    } else {
      g.append("text")
        .attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("fill", "white")
        .text(s === 4 ? "☠" : "⚑");
    }
  }
}

function drawGrid7x7(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  ox: number,
  oy: number,
) {
  const CELL7 = 28;
  const gw = 7 * CELL7 + 6 * GAP + 2 * PAD;
  const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
  g.append("text")
    .attr("x", gw / 2).attr("y", -4)
    .attr("text-anchor", "middle")
    .attr("class", "rl-label")
    .text("7×7 gridworld  |196 Q-values|");

  for (let s = 0; s < 49; s++) {
    const { r, c } = rc7(s);
    const cx = PAD + c * (CELL7 + GAP);
    const cy = PAD + r * (CELL7 + GAP);
    const isWall = wallCells7x7.has(s);
    const isGoal1 = s === idx7(6, 6);
    const isGoal2 = s === idx7(0, 6);
    const bg = isWall ? "#374151" : isGoal1 ? "#15803d" : isGoal2 ? "#0e7490" : "#f3f4f6";
    g.append("rect").attr("x", cx).attr("y", cy)
      .attr("width", CELL7).attr("height", CELL7)
      .attr("rx", 2).attr("fill", bg).attr("stroke", "#e5e7eb").attr("stroke-width", 0.5);
    if (isGoal1 || isGoal2) {
      g.append("text")
        .attr("x", cx + CELL7 / 2).attr("y", cy + CELL7 / 2 + 4)
        .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "white")
        .text(isGoal1 ? "⚑" : "◎");
    }
  }
}

function drawAtariSchematic(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  ox: number,
  oy: number,
) {
  const bw = 160;
  const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
  g.append("text")
    .attr("x", bw / 2).attr("y", -4)
    .attr("text-anchor", "middle")
    .attr("class", "rl-label")
    .text("Atari frames");

  // Draw a little pixel grid
  const pxSize = 4;
  for (let pr = 0; pr < 20; pr++) {
    for (let pc = 0; pc < 20; pc++) {
      const hue = (pr * 20 + pc) * 7;
      g.append("rect")
        .attr("x", PAD + pc * pxSize).attr("y", PAD + pr * pxSize)
        .attr("width", pxSize - 0.5).attr("height", pxSize - 0.5)
        .attr("fill", `hsl(${hue},60%,60%)`);
    }
  }
  g.append("text")
    .attr("x", bw / 2).attr("y", PAD + 20 * pxSize + 18)
    .attr("text-anchor", "middle")
    .attr("font-size", "11px")
    .attr("fill", "var(--rl-ink)")
    .text("84×84×4 pixels");
  g.append("text")
    .attr("x", bw / 2).attr("y", PAD + 20 * pxSize + 33)
    .attr("text-anchor", "middle")
    .attr("font-size", "11px")
    .attr("fill", "var(--triad-warning)")
    .attr("font-weight", "600")
    .text("~10⁶⁷⁹¹⁴ states");
}

function drawSizeChart(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  ox: number,
  oy: number,
  cw: number,
) {
  const data = [
    { label: "3×3", logVal: Math.log10(36) },
    { label: "7×7", logVal: Math.log10(196) },
    { label: "Atari", logVal: 67914 / 1000 }, // scaled for display
  ];
  const maxLog = 70;
  const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
  g.append("text").attr("x", cw / 2).attr("y", -4)
    .attr("text-anchor", "middle").attr("class", "rl-label").text("Q-table size (log scale)");

  data.forEach((d, i) => {
    const barW = (d.logVal / maxLog) * (cw - 80);
    const y = i * 16;
    g.append("rect").attr("x", 40).attr("y", y).attr("width", Math.max(barW, 4))
      .attr("height", 13).attr("rx", 2)
      .attr("fill", i === 2 ? "var(--triad-warning)" : "var(--fa-linear)");
    g.append("text").attr("x", 36).attr("y", y + 10)
      .attr("text-anchor", "end").attr("font-size", "10px").attr("fill", "var(--rl-ink)")
      .text(d.label);
    if (i < 2) {
      g.append("text").attr("x", 44 + Math.max(barW, 4)).attr("y", y + 10)
        .attr("font-size", "10px").attr("fill", "var(--rl-ink)")
        .text(i === 0 ? "36" : "196");
    } else {
      g.append("text").attr("x", 44 + Math.max(barW, 4)).attr("y", y + 10)
        .attr("font-size", "10px").attr("fill", "var(--triad-warning)").attr("font-weight", "600")
        .text("10⁶⁷⁹¹⁴");
    }
  });
}

class TabularLimits extends HTMLElement {
  connectedCallback() {
    const { panel, body } = createPanel({ id: "tabular-limits" });
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);
    const g3w = gridW(3);
    drawGrid3x3(svg, 10, 28);
    const g7w = 7 * 28 + 6 * GAP + 2 * PAD;
    drawGrid7x7(svg, g3w + 20, 28);
    drawAtariSchematic(svg, g3w + g7w + 30, 28);
    const rightX = g3w + g7w + 30 + 190;
    drawSizeChart(svg, rightX, H / 2 - 10, W - rightX - 8);
  }
}

customElements.define("dqn-tabular-limits", TabularLimits);
