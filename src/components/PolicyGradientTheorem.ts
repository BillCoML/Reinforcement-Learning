/**
 * V3 — The Policy Gradient Theorem.
 * Left: trajectory on 3×3 gridworld with score function direction at each step.
 * Right: same trajectory, score scaled by G_t (positive → amplify, negative → flip).
 * Controls: resample trajectory, average over N trajectories.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld } from "../mdp/gridworld";
import { sampleStep } from "../td/helpers";
import { mulberry32 } from "../importance-sampling/gaussian";
import { SoftmaxPolicy } from "../pg/softmax-policy";
import { computeReturns } from "../monte-carlo/returns";

const W = 720;
const H = 460;
const CELL = 52;
const GAP = 4;
const PAD = 12;
const G3W = 3 * CELL + 2 * GAP + 2 * PAD;

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const ACTION_ARROWS = ["↑", "→", "↓", "←"];
const DELTAS: [number,number][] = [[-1,0],[0,1],[1,0],[0,-1]];

let globalSeed = 0x5AF9;

function sampleTrajectory(policy: SoftmaxPolicy, maxSteps = 25): {
  states: number[]; actions: number[]; rewards: number[];
} {
  const rng = mulberry32(globalSeed++);
  const states: number[] = [], actions: number[] = [], rewards: number[] = [];
  let s = 0;
  for (let t = 0; t < maxSteps; t++) {
    if (mdp.terminals[s]) break;
    const a = policy.sample(s, rng);
    const { sp, r, done } = sampleStep(mdp, s, a, rng);
    states.push(s); actions.push(a); rewards.push(r);
    s = sp;
    if (done) break;
  }
  return { states, actions, rewards };
}

function drawGrid(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  title: string,
  trajectory: { states: number[]; actions: number[]; rewards: number[] },
  policy: SoftmaxPolicy,
  showScaled: boolean,
  nAvg: number,
  allTrajectories: typeof trajectory[],
) {
  g.selectAll("*").remove();
  g.append("text")
    .attr("x", G3W / 2).attr("y", -8)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
    .attr("fill", "var(--rl-ink)").text(title);

  // Draw cells
  for (let s = 0; s < 9; s++) {
    const r0 = Math.floor(s / 3), c0 = s % 3;
    const x = PAD + c0 * (CELL + GAP), y = PAD + r0 * (CELL + GAP);
    const isPit = mdp.terminals[s] && s === 4;
    const isGoal = mdp.terminals[s] && s === 8;
    const fill = isPit ? "#fecaca" : isGoal ? "#bbf7d0" : "#f8fafc";
    g.append("rect").attr("x", x).attr("y", y)
      .attr("width", CELL).attr("height", CELL).attr("rx", 4)
      .attr("fill", fill).attr("stroke", "#cbd5e1").attr("stroke-width", 1.5);
    if (isGoal) {
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 6)
        .attr("text-anchor", "middle").attr("font-size", "18px").text("⚑");
    } else if (isPit) {
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 6)
        .attr("text-anchor", "middle").attr("font-size", "18px").text("☠");
    }
  }

  // Gather per-state gradient from all trajectories
  const gradByState: Map<number, number[]> = new Map();
  const traj = nAvg <= 1 ? [trajectory] : allTrajectories.slice(0, nAvg);

  for (const tr of traj) {
    const Gs = tr.states.length > 0 ? computeReturns(tr.rewards, mdp.gamma) : [];
    for (let t = 0; t < tr.states.length; t++) {
      const st = tr.states[t];
      const at = tr.actions[t];
      const score = policy.scoreFunction(st, at);
      const weight = showScaled ? Gs[t] * Math.pow(mdp.gamma, t) : 1.0;
      if (!gradByState.has(st)) gradByState.set(st, [0, 0, 0, 0]);
      const cur = gradByState.get(st)!;
      for (let a = 0; a < mdp.nA; a++) cur[a] += score[a] * weight;
    }
  }
  // Average
  for (const [, v] of gradByState) {
    for (let a = 0; a < mdp.nA; a++) v[a] /= traj.length;
  }

  // Draw trajectory path arrows
  const { states, actions } = traj[0];
  for (let t = 0; t < states.length - 1; t++) {
    const s = states[t];
    const r0 = Math.floor(s / 3), c0 = s % 3;
    const cx = PAD + c0 * (CELL + GAP) + CELL / 2;
    const cy = PAD + r0 * (CELL + GAP) + CELL / 2;
    const [dr, dc] = DELTAS[actions[t]];
    const len = 14;
    g.append("line")
      .attr("x1", cx).attr("y1", cy)
      .attr("x2", cx + dc * len).attr("y2", cy + dr * len)
      .attr("stroke", "var(--pg-theta)").attr("stroke-width", 2.5)
      .attr("stroke-linecap", "round");
    g.append("text")
      .attr("x", cx - 8).attr("y", cy - 8)
      .attr("font-size", "9px").attr("fill", "var(--pg-theta)").attr("font-weight", "600")
      .text(`t=${t}`);
  }

  // Draw score function bars at visited states
  for (const [s, grad] of gradByState) {
    if (mdp.terminals[s]) continue;
    const r0 = Math.floor(s / 3), c0 = s % 3;
    const ox = PAD + c0 * (CELL + GAP);
    const oy = PAD + r0 * (CELL + GAP);

    const maxAbs = Math.max(...grad.map(Math.abs), 0.01);
    const barH = 6;

    // Mini bar chart: Up (a=0) top, Down (a=2) bottom
    for (const cfg of [
      { a: 0, y: 2 }, { a: 1, y: 11 }, { a: 2, y: 20 }, { a: 3, y: 29 },
    ]) {
      const v = grad[cfg.a];
      const barW = (Math.abs(v) / maxAbs) * (CELL / 2 - 6);
      const posColor = showScaled ? "var(--pg-advantage)" : "var(--pg-softmax)";
      const negColor = "var(--pg-high-variance)";
      g.append("rect")
        .attr("x", ox + CELL / 2 + (v < 0 ? -barW : 0))
        .attr("y", oy + cfg.y)
        .attr("width", barW || 1).attr("height", barH)
        .attr("rx", 1)
        .attr("fill", v >= 0 ? posColor : negColor)
        .attr("opacity", 0.8);
      // Center line
      g.append("line")
        .attr("x1", ox + CELL / 2).attr("x2", ox + CELL / 2)
        .attr("y1", oy + cfg.y).attr("y2", oy + cfg.y + barH)
        .attr("stroke", "#94a3b8").attr("stroke-width", 0.5);
      g.append("text")
        .attr("x", ox + 4).attr("y", oy + cfg.y + barH - 1)
        .attr("font-size", "7px").attr("fill", "#64748b")
        .text(ACTION_ARROWS[cfg.a]);
    }

    // Step return label
    const Gs = computeReturns(trajectory.rewards, mdp.gamma);
    const t = trajectory.states.indexOf(s);
    if (t >= 0) {
      const label = showScaled ? `G_${t}=${Gs[t].toFixed(2)}` : `score`;
      g.append("text")
        .attr("x", ox + CELL / 2).attr("y", oy + CELL - 4)
        .attr("text-anchor", "middle").attr("font-size", "8px").attr("fill", "#475569")
        .text(label);
    }
  }

  // Dim unvisited non-terminal states
  const visited = new Set(trajectory.states);
  for (let s = 0; s < 9; s++) {
    if (!mdp.terminals[s] && !visited.has(s)) {
      const r0 = Math.floor(s / 3), c0 = s % 3;
      g.append("rect")
        .attr("x", PAD + c0 * (CELL + GAP)).attr("y", PAD + r0 * (CELL + GAP))
        .attr("width", CELL).attr("height", CELL).attr("rx", 4)
        .attr("fill", "#f8fafc").attr("opacity", 0.6);
    }
  }
}

class PolicyGradientTheorem extends HTMLElement {
  connectedCallback() {
    const { panel, body, setStatus } = createPanel({ id: "policy-gradient-theorem" });
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.alignItems = "center";
    body.style.gap = "10px";

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);

    const svg = d3.select(svgEl);
    const gLeft = svg.append("g").attr("transform", "translate(10, 28)");
    const gRight = svg.append("g").attr("transform", `translate(${10 + G3W + 36}, 28)`);

    // Divider
    const divX = 10 + G3W + 18;
    svg.append("line")
      .attr("x1", divX).attr("y1", 10).attr("x2", divX).attr("y2", H - 10)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

    // Legend area at bottom
    const legY = H - 36;
    svg.append("text").attr("x", 10 + G3W / 2).attr("y", legY)
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#64748b")
      .text("Bars = score ∇log π(a|s)  ·  blue=positive, red=negative");
    svg.append("text").attr("x", 10 + G3W + 36 + G3W / 2).attr("y", legY)
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#64748b")
      .text("Bars = score × G_t  ·  positive G_t amplifies, negative flips");

    // Controls
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;align-items:center;gap:14px;font-size:13px;flex-wrap:wrap;";

    const resampleBtn = document.createElement("button");
    resampleBtn.className = "rl-btn"; resampleBtn.textContent = "Resample trajectory";
    resampleBtn.style.fontSize = "12px";

    const nLabel = document.createElement("label");
    nLabel.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px;";
    nLabel.textContent = "Avg over N trajectories: ";
    const nSlider = document.createElement("input");
    nSlider.type = "range"; nSlider.min = "1"; nSlider.max = "20"; nSlider.value = "1";
    nSlider.style.width = "100px";
    const nVal = document.createElement("span");
    nVal.style.cssText = "font-family:var(--font-mono);min-width:24px;font-weight:600;";
    nVal.textContent = "1";
    nLabel.append(nSlider, nVal);

    controls.append(resampleBtn, nLabel);
    body.appendChild(controls);
    this.appendChild(panel);

    const policy = new SoftmaxPolicy(mdp.nS, mdp.nA); // uniform
    let allTrajectories = Array.from({ length: 20 }, () => sampleTrajectory(policy));
    let mainTraj = allTrajectories[0];
    let nAvg = 1;

    const render = () => {
      drawGrid(gLeft, "Score function  ∇log π(aₜ|sₜ)", mainTraj, policy, false, nAvg, allTrajectories);
      drawGrid(gRight, "Score × Gₜ  (gradient contribution)", mainTraj, policy, true, nAvg, allTrajectories);
      setStatus(`T=${mainTraj.states.length} N=${nAvg}`);
    };

    resampleBtn.addEventListener("click", () => {
      allTrajectories = Array.from({ length: 20 }, () => sampleTrajectory(policy));
      mainTraj = allTrajectories[0];
      render();
    });

    nSlider.addEventListener("input", () => {
      nAvg = parseInt(nSlider.value);
      nVal.textContent = String(nAvg);
      render();
    });

    render();
  }
}

customElements.define("pg-policy-gradient-theorem", PolicyGradientTheorem);
