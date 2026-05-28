/**
 * V4 — REINFORCE Training Trace.
 * Top: 3×3 gridworld showing current softmax policy bars (all states).
 * Bottom: per-episode return (raw) + running mean, with reference lines.
 * Timeline scrubber replays training history.
 * Runs REINFORCE in-browser at load (2000 episodes, ~150ms).
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld } from "../mdp/gridworld";
import { sampleStep } from "../td/helpers";
import { mulberry32 } from "../importance-sampling/gaussian";
import { SoftmaxPolicy } from "../pg/softmax-policy";
import { computeReturns } from "../monte-carlo/returns";

const W = 880;
const H_GRID = 220;
const H_CHART = 200;
const H = H_GRID + H_CHART + 30;

const CELL = 44;
const GAP = 4;
const PAD = 10;
const G3W = 3 * CELL + 2 * GAP + 2 * PAD;
const GRID_OX = (W - G3W) / 2;

const CHART_MARGIN = { top: 10, right: 20, bottom: 36, left: 52 };
const CHART_W = W - CHART_MARGIN.left - CHART_MARGIN.right;
const CHART_H = H_CHART - CHART_MARGIN.top - CHART_MARGIN.bottom;

const N_EPISODES = 2000;
const ALPHA = 0.1;
const RECORD_EVERY = 10;
const V_STAR = 0.729;
const V_CAP = 0.722;

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

interface Snapshot {
  ep: number;
  probsByState: number[][];  // [nS][nA]
  v: number;
  gradNorm: number;
}

function runReinforce(): { snapshots: Snapshot[]; allReturns: number[] } {
  const rng = mulberry32(0);
  const policy = new SoftmaxPolicy(mdp.nS, mdp.nA);
  const snapshots: Snapshot[] = [];
  const allReturns: number[] = [];

  for (let ep = 0; ep < N_EPISODES; ep++) {
    const states: number[] = [], actions: number[] = [], rewards: number[] = [];
    let s = 0;
    for (let t = 0; t < 300; t++) {
      if (mdp.terminals[s]) break;
      const a = policy.sample(s, rng);
      const { sp, r, done } = sampleStep(mdp, s, a, rng);
      states.push(s); actions.push(a); rewards.push(r);
      s = sp;
      if (done) break;
    }

    const T = states.length;
    let epReturn = 0;
    let gradNormSq = 0;

    if (T > 0) {
      const Gs = computeReturns(rewards, mdp.gamma);
      epReturn = Gs[0];
      for (let t = 0; t < T; t++) {
        const st = states[t], at = actions[t];
        const gammaT = Math.pow(mdp.gamma, t);
        const score = policy.scoreFunction(st, at);
        const base = st * mdp.nA;
        for (let ap = 0; ap < mdp.nA; ap++) {
          const delta = ALPHA * gammaT * Gs[t] * score[ap];
          policy.theta[base + ap] += delta;
          gradNormSq += delta * delta;
        }
      }
    }
    allReturns.push(epReturn);

    if (ep % RECORD_EVERY === 0 || ep === N_EPISODES - 1) {
      // Estimate V(s0) via 8 rollouts
      const evalRng = mulberry32(0xDEAD0000 + (ep & 0xFFFF));
      let vSum = 0;
      for (let i = 0; i < 8; i++) {
        let sE = 0, G = 0, tE = 0;
        while (!mdp.terminals[sE] && tE < 100) {
          const aE = policy.sample(sE, evalRng);
          const { sp: spE, r: rE } = sampleStep(mdp, sE, aE, evalRng);
          G += Math.pow(mdp.gamma, tE) * rE;
          sE = spE; tE++;
        }
        vSum += G;
      }
      snapshots.push({
        ep,
        probsByState: Array.from({ length: mdp.nS }, (_, s2) => Array.from(policy.probs(s2))),
        v: vSum / 8,
        gradNorm: Math.sqrt(gradNormSq),
      });
    }
  }
  return { snapshots, allReturns };
}

function drawGridPolicy(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  probsByState: number[][],
) {
  g.selectAll("*").remove();
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
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 5)
        .attr("text-anchor", "middle").attr("font-size", "16px").text("⚑");
    } else if (isPit) {
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 5)
        .attr("text-anchor", "middle").attr("font-size", "16px").text("☠");
    } else {
      const probs = probsByState[s];
      const maxP = Math.max(...probs);
      // Draw 4 directional bars
      const configs = [
        { a: 0, bx: 6, by: 2, bw: CELL - 12, bh: 6 },    // Up
        { a: 2, bx: 6, by: CELL - 8, bw: CELL - 12, bh: 6 }, // Down
        { a: 1, bx: CELL - 8, by: 6, bw: 6, bh: CELL - 12 }, // Right
        { a: 3, bx: 2, by: 6, bw: 6, bh: CELL - 12 },    // Left
      ];
      for (const cfg of configs) {
        const p = probs[cfg.a];
        const isMax = p === maxP;
        g.append("rect").attr("x", x + cfg.bx).attr("y", y + cfg.by)
          .attr("width", cfg.bw).attr("height", cfg.bh).attr("rx", 1)
          .attr("fill", "#e2e8f0");
        const fill2 = isMax ? "var(--pg-softmax)" : "#a78bfa";
        const fw = cfg.bw > cfg.bh ? p * cfg.bw : cfg.bw;
        const fh = cfg.bh > cfg.bw ? p * cfg.bh : cfg.bh;
        g.append("rect")
          .attr("x", x + cfg.bx + (cfg.bw > cfg.bh ? 0 : 0))
          .attr("y", y + cfg.by + (cfg.bh > cfg.bw ? cfg.bh - fh : 0))
          .attr("width", Math.max(fw, 1)).attr("height", Math.max(fh, 1))
          .attr("rx", 1).attr("fill", fill2).attr("opacity", isMax ? 0.9 : 0.7);
      }
      // Max prob label
      g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 4)
        .attr("text-anchor", "middle").attr("font-size", "9px")
        .attr("fill", "#334155").attr("font-weight", "600")
        .text(maxP.toFixed(2));
    }
  }
}

class REINFORCETrainingTrace extends HTMLElement {
  connectedCallback() {
    const { panel, body, setStatus } = createPanel({
      id: "reinforce-training-trace",
      heavy: true,
      mobileNotice: "Scroll right for the full training trace.",
    });
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.alignItems = "center";
    body.style.gap = "8px";

    const loadingEl = document.createElement("div");
    loadingEl.style.cssText = "font-size:13px;color:#64748b;padding:16px;";
    loadingEl.textContent = "Training REINFORCE (2 000 episodes)…";
    body.appendChild(loadingEl);
    this.appendChild(panel);

    setTimeout(() => {
      const { snapshots, allReturns } = runReinforce();

      loadingEl.remove();

      const wrap = document.createElement("div");
      wrap.style.overflowX = "auto";
      const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
      svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svgEl.style.width = "100%";
      svgEl.style.maxWidth = `${W}px`;
      wrap.appendChild(svgEl);
      body.insertBefore(wrap, body.firstChild);

      const svg = d3.select(svgEl);

      // Grid title
      svg.append("text").attr("x", W / 2).attr("y", 14)
        .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
        .attr("fill", "var(--rl-ink)").text("Softmax policy  π_θ(a|s)  at all states");

      const gGrid = svg.append("g").attr("transform", `translate(${GRID_OX}, 20)`);

      // Chart group
      const gChart = svg.append("g").attr("transform",
        `translate(${CHART_MARGIN.left},${H_GRID + CHART_MARGIN.top + 10})`);

      // Build return chart
      const xScale = d3.scaleLinear().domain([0, N_EPISODES - 1]).range([0, CHART_W]);
      const yScale = d3.scaleLinear().domain([-1.1, 0.85]).range([CHART_H, 0]);

      gChart.append("g").attr("transform", `translate(0,${CHART_H})`)
        .call(d3.axisBottom(xScale).ticks(8).tickSize(-CHART_H))
        .call(ax => ax.select(".domain").remove())
        .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));
      gChart.append("g")
        .call(d3.axisLeft(yScale).ticks(5).tickSize(-CHART_W))
        .call(ax => ax.select(".domain").remove())
        .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));

      gChart.append("text").attr("x", CHART_W / 2).attr("y", CHART_H + 28)
        .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b")
        .text("Episode");
      gChart.append("text").attr("transform", "rotate(-90)")
        .attr("x", -CHART_H / 2).attr("y", -38)
        .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b")
        .text("Return G₀");

      // Reference lines
      for (const [val, label, color] of [
        [V_STAR, `V*(0,0) = ${V_STAR}`, "#15803d"],
        [V_CAP, `softmax cap ≈ ${V_CAP}`, "#7c3aed"],
      ] as [number, string, string][]) {
        gChart.append("line")
          .attr("x1", 0).attr("x2", CHART_W)
          .attr("y1", yScale(val)).attr("y2", yScale(val))
          .attr("stroke", color).attr("stroke-width", 1).attr("stroke-dasharray", "5,3").attr("opacity", 0.7);
        gChart.append("text").attr("x", CHART_W - 4).attr("y", yScale(val) - 3)
          .attr("text-anchor", "end").attr("font-size", "9px").attr("fill", color).text(label);
      }

      // Raw returns as translucent bars (downsampled)
      const stride = Math.ceil(N_EPISODES / 400);
      const barW = Math.max(1, xScale(stride) - xScale(0) - 0.5);
      for (let i = 0; i < allReturns.length; i += stride) {
        const rv = allReturns[i];
        const y0 = yScale(Math.max(rv, -1.1));
        const y1 = yScale(0);
        gChart.append("rect")
          .attr("x", xScale(i)).attr("y", Math.min(y0, y1))
          .attr("width", barW).attr("height", Math.abs(y0 - y1))
          .attr("fill", rv >= 0 ? "var(--pg-advantage)" : "var(--pg-high-variance)")
          .attr("opacity", 0.25);
      }

      // Running mean line
      const WINDOW = 100;
      const runMean: [number, number][] = [];
      for (let i = 0; i < allReturns.length; i++) {
        const start = Math.max(0, i - WINDOW + 1);
        const avg = allReturns.slice(start, i + 1).reduce((a, b) => a + b, 0) / (i - start + 1);
        runMean.push([i, avg]);
      }
      const lineGen = d3.line<[number,number]>().x(d => xScale(d[0])).y(d => yScale(d[1]));
      gChart.append("path").datum(runMean)
        .attr("fill", "none").attr("stroke", "var(--pg-baseline)").attr("stroke-width", 2)
        .attr("d", lineGen);

      // Current episode marker
      const marker = gChart.append("line")
        .attr("y1", 0).attr("y2", CHART_H)
        .attr("stroke", "var(--pg-theta)").attr("stroke-width", 1.5).attr("opacity", 0.8);

      // V estimate dot
      const vDot = gChart.append("circle").attr("r", 4)
        .attr("fill", "var(--pg-theta)").attr("stroke", "white").attr("stroke-width", 1.5);

      // Scrubber
      const scrubWrap = document.createElement("div");
      scrubWrap.style.cssText = "display:flex;align-items:center;gap:10px;font-size:12px;width:90%;max-width:800px;";
      const scrubLabel = document.createElement("label");
      scrubLabel.style.cssText = "display:flex;align-items:center;gap:8px;flex:1;";
      scrubLabel.textContent = "Episode: ";
      const scrubVal = document.createElement("span");
      scrubVal.style.cssText = "font-family:var(--font-mono);font-weight:600;color:var(--pg-theta);min-width:48px;";
      const scrub = document.createElement("input");
      scrub.type = "range"; scrub.min = "0"; scrub.max = String(snapshots.length - 1);
      scrub.value = "0"; scrub.style.flex = "1";
      scrubLabel.append(scrub, scrubVal);
      scrubWrap.appendChild(scrubLabel);
      body.appendChild(scrubWrap);

      const showSnapshot = (idx: number) => {
        const snap = snapshots[idx];
        drawGridPolicy(gGrid, snap.probsByState);
        const epX = xScale(snap.ep);
        marker.attr("x1", epX).attr("x2", epX);
        vDot.attr("cx", epX).attr("cy", yScale(snap.v));
        scrubVal.textContent = String(snap.ep);
        setStatus(`ep=${snap.ep}  V̂(0,0)=${snap.v.toFixed(3)}`);
      };

      showSnapshot(0);

      scrub.addEventListener("input", () => showSnapshot(parseInt(scrub.value)));
    }, 0);
  }
}

customElements.define("pg-reinforce-training-trace", REINFORCETrainingTrace);
