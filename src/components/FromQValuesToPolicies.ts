/**
 * V1 — From Q-Values to Policies.
 * Left: 3×3 gridworld with Q-values from Q-learning (greedy arrow overlay).
 * Right: same grid with softmax policy probabilities shown as directional bars.
 * Toggle highlights the difference: indirect (argmax) vs direct representation.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld } from "../mdp/gridworld";
import { qLearning } from "../td/q-learning";
import { mulberry32 } from "../importance-sampling/gaussian";
import { SoftmaxPolicy } from "../pg/softmax-policy";
import { reinforce } from "../pg/reinforce";

const W = 720;
const H = 360;
const CELL = 54;
const GAP = 4;
const PAD = 14;
const G3W = 3 * CELL + 2 * GAP + 2 * PAD;

const ARROW_DIRS: [number, number][] = [[-1,0],[0,1],[1,0],[0,-1]]; // Up Right Down Left

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

function getQValues(): Float64Array {
  const { Q } = qLearning(mdp, 4000, 0.1, 0.1, { rng: mulberry32(7) });
  return Q;
}

function getSoftmaxPolicy(): SoftmaxPolicy {
  const { policy } = reinforce(mdp, 2000, 0.1, { rng: mulberry32(0) });
  return policy;
}

function drawGridBase(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  title: string,
) {
  g.append("text")
    .attr("x", G3W / 2).attr("y", -8)
    .attr("text-anchor", "middle")
    .attr("font-size", "11px").attr("font-weight", "600")
    .attr("fill", "var(--rl-ink)")
    .text(title);

  for (let s = 0; s < 9; s++) {
    const r = Math.floor(s / 3), c = s % 3;
    const x = PAD + c * (CELL + GAP);
    const y = PAD + r * (CELL + GAP);
    const isTerminal = mdp.terminals[s];
    const isPit = isTerminal && s === 4;
    const isGoal = isTerminal && s === 8;
    const fill = isPit ? "#fecaca" : isGoal ? "#bbf7d0" : "#f8fafc";
    g.append("rect")
      .attr("x", x).attr("y", y)
      .attr("width", CELL).attr("height", CELL)
      .attr("rx", 4)
      .attr("fill", fill)
      .attr("stroke", "#cbd5e1").attr("stroke-width", 1.5);

    if (isGoal) {
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 6)
        .attr("text-anchor", "middle").attr("font-size", "20px").text("⚑");
    } else if (isPit) {
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 6)
        .attr("text-anchor", "middle").attr("font-size", "20px").text("☠");
    }
  }
}

function drawQValues(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  Q: Float64Array,
) {
  drawGridBase(g, "Q-learning  ·  greedy policy");

  for (let s = 0; s < 9; s++) {
    if (mdp.terminals[s]) continue;
    const r = Math.floor(s / 3), c = s % 3;
    const cx = PAD + c * (CELL + GAP) + CELL / 2;
    const cy = PAD + r * (CELL + GAP) + CELL / 2;
    const nA = mdp.nA;

    // Small Q-value labels at cardinal positions
    const offsets: [number,number][] = [[0,-14],[14,0],[0,14],[-14,0]];
    let maxQ = -Infinity, bestA = 0;
    for (let a = 0; a < nA; a++) {
      const q = Q[s * nA + a];
      if (q > maxQ) { maxQ = q; bestA = a; }
      g.append("text")
        .attr("x", cx + offsets[a][0]).attr("y", cy + offsets[a][1] + 4)
        .attr("text-anchor", "middle")
        .attr("font-size", "8px")
        .attr("fill", a === bestA ? "var(--pg-theta)" : "#94a3b8")
        .attr("font-weight", a === bestA ? "600" : "400")
        .text(q.toFixed(2));
    }

    // Greedy arrow
    const [dr, dc] = ARROW_DIRS[bestA];
    const len = 10;
    g.append("line")
      .attr("x1", cx).attr("y1", cy)
      .attr("x2", cx + dc * len).attr("y2", cy + dr * len)
      .attr("stroke", "var(--pg-theta)").attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead-pg)");
  }
}

function drawSoftmaxPolicy(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  policy: SoftmaxPolicy,
) {
  drawGridBase(g, "Softmax policy  ·  π_θ(a|s)");

  for (let s = 0; s < 9; s++) {
    if (mdp.terminals[s]) continue;
    const r = Math.floor(s / 3), c = s % 3;
    const ox = PAD + c * (CELL + GAP);
    const oy = PAD + r * (CELL + GAP);
    const probs = policy.probs(s);

    // Draw probability bars: small rects at cardinal positions
    const barConfigs: { dx: number; dy: number; w: number; h: number; a: number }[] = [
      { dx: (CELL - 20) / 2, dy: 2, w: 20, h: 10, a: 0 },   // Up
      { dx: CELL - 12, dy: (CELL - 20) / 2, w: 10, h: 20, a: 1 }, // Right
      { dx: (CELL - 20) / 2, dy: CELL - 12, w: 20, h: 10, a: 2 }, // Down
      { dx: 2, dy: (CELL - 20) / 2, w: 10, h: 20, a: 3 },    // Left
    ];

    const maxProb = Math.max(...Array.from({ length: 4 }, (_, a) => probs[a]));
    for (const cfg of barConfigs) {
      const p = probs[cfg.a];
      const isMax = p === maxProb;
      const fill = isMax ? "var(--pg-softmax)" : "#c4b5fd";
      const fracW = cfg.w > cfg.h ? p * cfg.w : cfg.w;
      const fracH = cfg.h > cfg.w ? p * cfg.h : cfg.h;

      // Background track
      g.append("rect")
        .attr("x", ox + cfg.dx).attr("y", oy + cfg.dy)
        .attr("width", cfg.w).attr("height", cfg.h)
        .attr("rx", 2).attr("fill", "#e2e8f0");
      // Probability fill
      g.append("rect")
        .attr("x", ox + cfg.dx).attr("y", oy + cfg.dy + (cfg.h > cfg.w ? cfg.h - fracH : 0))
        .attr("width", fracW).attr("height", fracH > 0 ? fracH : cfg.h)
        .attr("rx", 2).attr("fill", fill);
      // Probability label
      g.append("text")
        .attr("x", ox + cfg.dx + cfg.w / 2)
        .attr("y", oy + cfg.dy + cfg.h / 2 + 3)
        .attr("text-anchor", "middle")
        .attr("font-size", "7px")
        .attr("fill", isMax ? "white" : "#475569")
        .attr("font-weight", isMax ? "600" : "400")
        .text(p.toFixed(2));
    }
  }
}

class FromQValuesToPolicies extends HTMLElement {
  connectedCallback() {
    const { panel, body } = createPanel({ id: "q-values-to-policies" });
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.alignItems = "center";
    body.style.gap = "12px";

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);

    const svg = d3.select(svgEl);

    // Arrowhead marker
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrowhead-pg")
      .attr("markerWidth", 8).attr("markerHeight", 6)
      .attr("refX", 8).attr("refY", 3)
      .attr("orient", "auto")
      .append("polygon")
      .attr("points", "0 0, 8 3, 0 6")
      .attr("fill", "var(--pg-theta)");

    const gLeft = svg.append("g").attr("transform", "translate(20, 28)");
    const gRight = svg.append("g").attr("transform", `translate(${20 + G3W + 40}, 28)`);

    // Label: argmax indirect
    svg.append("text")
      .attr("x", 20 + G3W / 2).attr("y", H - 12)
      .attr("text-anchor", "middle").attr("font-size", "10px")
      .attr("fill", "#64748b")
      .text("Indirect: π(s) = argmax_a Q(s,a)");

    // Label: softmax direct
    svg.append("text")
      .attr("x", 20 + G3W + 40 + G3W / 2).attr("y", H - 12)
      .attr("text-anchor", "middle").attr("font-size", "10px")
      .attr("fill", "#64748b")
      .text("Direct: π_θ(a|s) = softmax(θ_{s,a})");

    // Vertical divider
    const divX = 20 + G3W + 20;
    svg.append("line")
      .attr("x1", divX).attr("y1", 10).attr("x2", divX).attr("y2", H - 20)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

    const Q = getQValues();
    const policy = getSoftmaxPolicy();

    drawQValues(gLeft, Q);
    drawSoftmaxPolicy(gRight, policy);

    // Toggle button
    const btn = document.createElement("button");
    btn.className = "rl-btn";
    btn.textContent = "Highlight: what changes between the two views?";
    btn.style.fontSize = "12px";
    let highlighted = false;

    const note = document.createElement("div");
    note.style.fontSize = "12px";
    note.style.color = "#475569";
    note.style.maxWidth = "600px";
    note.style.textAlign = "center";
    note.style.display = "none";
    note.textContent =
      "Left (Q-learning): the policy is implicit — one argmax per state. "
      + "Right (softmax): the policy is a probability distribution at every state, "
      + "updated directly. There is no argmax anywhere.";

    btn.addEventListener("click", () => {
      highlighted = !highlighted;
      note.style.display = highlighted ? "block" : "none";
      btn.textContent = highlighted
        ? "Hide explanation"
        : "Highlight: what changes between the two views?";
    });

    body.appendChild(btn);
    body.appendChild(note);
    this.appendChild(panel);
  }
}

customElements.define("pg-from-q-values-to-policies", FromQValuesToPolicies);
