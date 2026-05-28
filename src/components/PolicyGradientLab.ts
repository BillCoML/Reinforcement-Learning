/**
 * V6 — Policy Gradient Lab (Centerpiece).
 * Trains vanilla REINFORCE, REINFORCE+baseline, and actor-critic in parallel.
 * Six synchronized panels: policy display, learning curves, gradient norms,
 * critic value (AC only), episode returns, settings strip.
 * Width 960 (arena breakout), Height 900.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld } from "../mdp/gridworld";
import { reinforce } from "../pg/reinforce";
import { actorCritic } from "../pg/actor-critic";
import { mulberry32 } from "../importance-sampling/gaussian";
import { makeValueColorScale } from "./value-scale";
import { policyEvaluationExact } from "../mdp/policy-evaluation";

const W = 960;
const CELL = 36;
const GAP = 3;
const PAD = 8;
const G3W = 3 * CELL + 2 * GAP + 2 * PAD;
const G3H = 3 * CELL + 2 * GAP + 2 * PAD;

const VANILLA_COLOR = "var(--pg-vanilla)";
const BASELINE_COLOR = "var(--pg-baseline)";
const AC_COLOR = "var(--pg-actor-critic)";
const V_STAR = 0.729;
const V_CAP = 0.722;

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const TRUE_V = policyEvaluationExact(mdp, { pi: Array.from({ length: mdp.nS }, () => Array(mdp.nA).fill(1/mdp.nA)) });

interface AlgoResult {
  label: string;
  color: string;
  history: number[];
  gradNorms: number[];
  episodeReturns: number[];
  probsByEp?: number[][][];    // snapshots[epIdx][stateIdx][actionIdx]
  criticByEp?: number[][];     // snapshots[epIdx][stateIdx] — AC only
  snapStride: number;
}

function runAll(nEpisodes: number, alphaActor: number, alphaCritic: number, gamma: number): AlgoResult[] {
  const mdp2 = buildGridworld({ slippery: false, gamma });
  const SNAP_STRIDE = Math.max(1, Math.floor(nEpisodes / 100));

  const runReinforceWithSnaps = (useBaseline: boolean) => {
    const { history, gradNorms, episodeReturns } =
      reinforce(mdp2, nEpisodes, alphaActor, {
        rng: mulberry32(0),
        useBaseline,
        alphaCritic: useBaseline ? alphaCritic : 0,
        maxSteps: 300,
      });

    // Can't snapshot internals easily; just record final state
    // Collect snapshots by re-running (cheap for small N)
    const probsByEp: number[][][] = [];
    {
      const rng2 = mulberry32(0);
      const policy2 = new (class {
        theta = new Float64Array(mdp2.nS * mdp2.nA);
        probs(s: number) {
          const base = s * mdp2.nA;
          let max = -Infinity;
          for (let a = 0; a < mdp2.nA; a++) if (this.theta[base + a] > max) max = this.theta[base + a];
          let sum = 0; const p = new Float64Array(mdp2.nA);
          for (let a = 0; a < mdp2.nA; a++) { p[a] = Math.exp(this.theta[base + a] - max); sum += p[a]; }
          for (let a = 0; a < mdp2.nA; a++) p[a] /= sum;
          return p;
        }
        scoreFunction(s: number, a: number) {
          const p = this.probs(s);
          const sc = new Float64Array(mdp2.nA);
          for (let ap = 0; ap < mdp2.nA; ap++) sc[ap] = (ap === a ? 1 : 0) - p[ap];
          return sc;
        }
        sample(s: number, rng: () => number) {
          const p = this.probs(s); let u = rng(), acc = 0;
          for (let a = 0; a < mdp2.nA; a++) { acc += p[a]; if (u <= acc) return a; }
          return mdp2.nA - 1;
        }
      })();
      const V2 = new Float64Array(mdp2.nS);
      let runMean = 0, seenCount = 0;

      for (let ep = 0; ep < nEpisodes; ep++) {
        const states: number[] = [], actions: number[] = [], rewards: number[] = [];
        let s = 0;
        for (let t = 0; t < 300; t++) {
          if (mdp2.terminals[s]) break;
          const a = policy2.sample(s, rng2);
          let sp = 0, r2 = 0, done2 = false;
          const u = rng2();
          let acc = 0;
          for (let sp2 = 0; sp2 < mdp2.nS; sp2++) {
            acc += mdp2.P[s][a][sp2];
            if (u <= acc) { sp = sp2; break; }
          }
          r2 = mdp2.r[s][a]; done2 = mdp2.terminals[sp];
          states.push(s); actions.push(a); rewards.push(r2);
          if (useBaseline && alphaCritic > 0) {
            const vNext = done2 ? 0 : V2[sp];
            V2[s] += alphaCritic * (r2 + mdp2.gamma * vNext - V2[s]);
          }
          s = sp; if (done2) break;
        }
        const T = states.length;
        if (T > 0) {
          const Gs: number[] = new Array(T);
          let g = 0;
          for (let t = T - 1; t >= 0; t--) { g = rewards[t] + mdp2.gamma * g; Gs[t] = g; }
          if (useBaseline && alphaCritic === 0) { seenCount++; runMean += (Gs[0] - runMean) / seenCount; }
          for (let t = 0; t < T; t++) {
            const st = states[t], at = actions[t];
            const b = useBaseline ? (alphaCritic > 0 ? V2[st] : runMean) : 0;
            const adv = Gs[t] - b;
            const gammaT = Math.pow(mdp2.gamma, t);
            const sc = policy2.scoreFunction(st, at);
            const base = st * mdp2.nA;
            for (let ap = 0; ap < mdp2.nA; ap++) policy2.theta[base + ap] += alphaActor * gammaT * adv * sc[ap];
          }
        }
        if (ep % SNAP_STRIDE === 0 || ep === nEpisodes - 1) {
          probsByEp.push(
            Array.from({ length: mdp2.nS }, (_, s2) => Array.from(policy2.probs(s2)))
          );
        }
      }
    }

    return { history, gradNorms, episodeReturns, probsByEp };
  };

  const vanillaRun = runReinforceWithSnaps(false);
  const baselineRun = runReinforceWithSnaps(true);

  // Actor-Critic with critic snapshots
  const acResult = actorCritic(mdp2, nEpisodes, alphaActor, alphaCritic, { rng: mulberry32(0), maxSteps: 300 });
  const acProbsByEp: number[][][] = [];
  const acCriticByEp: number[][] = [];
  {
    const rng2 = mulberry32(0);
    const acPol = new (class {
      theta = new Float64Array(mdp2.nS * mdp2.nA);
      probs(s: number) {
        const base = s * mdp2.nA;
        let max = -Infinity;
        for (let a = 0; a < mdp2.nA; a++) if (this.theta[base + a] > max) max = this.theta[base + a];
        let sum = 0; const p = new Float64Array(mdp2.nA);
        for (let a = 0; a < mdp2.nA; a++) { p[a] = Math.exp(this.theta[base + a] - max); sum += p[a]; }
        for (let a = 0; a < mdp2.nA; a++) p[a] /= sum;
        return p;
      }
      scoreFunction(s: number, a: number) {
        const p = this.probs(s);
        const sc = new Float64Array(mdp2.nA);
        for (let ap = 0; ap < mdp2.nA; ap++) sc[ap] = (ap === a ? 1 : 0) - p[ap];
        return sc;
      }
      sample(s: number, rng: () => number) {
        const p = this.probs(s); let u = rng(), acc = 0;
        for (let a = 0; a < mdp2.nA; a++) { acc += p[a]; if (u <= acc) return a; }
        return mdp2.nA - 1;
      }
    })();
    const acV = new Float64Array(mdp2.nS);
    for (let ep = 0; ep < nEpisodes; ep++) {
      let s = 0; let t = 0;
      while (!mdp2.terminals[s] && t < 300) {
        const a = acPol.sample(s, rng2);
        let sp = 0; const u = rng2(); let acc = 0;
        for (let sp2 = 0; sp2 < mdp2.nS; sp2++) { acc += mdp2.P[s][a][sp2]; if (u <= acc) { sp = sp2; break; } }
        const r2 = mdp2.r[s][a]; const done2 = mdp2.terminals[sp];
        const vNext = done2 ? 0 : acV[sp];
        const delta = r2 + mdp2.gamma * vNext - acV[sp];
        acV[s] += alphaCritic * delta;
        const sc = acPol.scoreFunction(s, a);
        const base2 = s * mdp2.nA;
        for (let ap = 0; ap < mdp2.nA; ap++) acPol.theta[base2 + ap] += alphaActor * delta * sc[ap];
        s = sp; t++; if (done2) break;
      }
      if (ep % SNAP_STRIDE === 0 || ep === nEpisodes - 1) {
        acProbsByEp.push(Array.from({ length: mdp2.nS }, (_, s2) => Array.from(acPol.probs(s2))));
        acCriticByEp.push(Array.from(acV));
      }
    }
  }

  return [
    { label: "Vanilla REINFORCE", color: VANILLA_COLOR, snapStride: SNAP_STRIDE, ...vanillaRun },
    { label: "REINFORCE + baseline", color: BASELINE_COLOR, snapStride: SNAP_STRIDE, ...baselineRun },
    { label: "Actor-Critic", color: AC_COLOR, snapStride: SNAP_STRIDE,
      history: acResult.history, gradNorms: acResult.gradNorms, episodeReturns: acResult.episodeReturns,
      probsByEp: acProbsByEp, criticByEp: acCriticByEp },
  ];
}

function drawGridPolicy(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  probs: number[][] | null,
) {
  g.selectAll("*").remove();
  for (let s = 0; s < 9; s++) {
    const r0 = Math.floor(s / 3), c0 = s % 3;
    const x = PAD + c0 * (CELL + GAP), y = PAD + r0 * (CELL + GAP);
    const isPit = mdp.terminals[s] && s === 4;
    const isGoal = mdp.terminals[s] && s === 8;
    const fill = isPit ? "#fecaca" : isGoal ? "#bbf7d0" : "#f8fafc";
    g.append("rect").attr("x", x).attr("y", y)
      .attr("width", CELL).attr("height", CELL).attr("rx", 3)
      .attr("fill", fill).attr("stroke", "#cbd5e1").attr("stroke-width", 1);
    if (isGoal) g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 5)
      .attr("text-anchor", "middle").attr("font-size", "14px").text("⚑");
    else if (isPit) g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 5)
      .attr("text-anchor", "middle").attr("font-size", "14px").text("☠");
    else if (probs) {
      const p = probs[s];
      const maxP = Math.max(...p);
      const configs = [
        { a: 0, bx: 5, by: 1, bw: CELL-10, bh: 5 },
        { a: 2, bx: 5, by: CELL-6, bw: CELL-10, bh: 5 },
        { a: 1, bx: CELL-6, by: 5, bw: 5, bh: CELL-10 },
        { a: 3, bx: 1, by: 5, bw: 5, bh: CELL-10 },
      ];
      for (const cfg of configs) {
        const pv = p[cfg.a];
        g.append("rect").attr("x", x+cfg.bx).attr("y", y+cfg.by)
          .attr("width", cfg.bw).attr("height", cfg.bh).attr("rx", 1).attr("fill", "#e2e8f0");
        const fw = cfg.bw > cfg.bh ? pv * cfg.bw : cfg.bw;
        const fh = cfg.bh > cfg.bw ? pv * cfg.bh : cfg.bh;
        g.append("rect")
          .attr("x", x+cfg.bx).attr("y", y+cfg.by+(cfg.bh > cfg.bw ? cfg.bh-fh : 0))
          .attr("width", Math.max(fw,1)).attr("height", Math.max(fh,1))
          .attr("rx", 1).attr("fill", pv === maxP ? "var(--pg-softmax)" : "#a78bfa").attr("opacity", pv===maxP?0.9:0.65);
      }
      g.append("text").attr("x", x+CELL/2).attr("y", y+CELL/2+4)
        .attr("text-anchor", "middle").attr("font-size", "8px").attr("fill", "#334155").attr("font-weight", "600")
        .text(maxP.toFixed(2));
    }
  }
}

function drawCriticGrid(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  V: number[] | null,
) {
  g.selectAll("*").remove();
  const scale = makeValueColorScale([-1, 0.8]);
  for (let s = 0; s < 9; s++) {
    const r0 = Math.floor(s / 3), c0 = s % 3;
    const x = PAD + c0 * (CELL + GAP), y = PAD + r0 * (CELL + GAP);
    const isPit = mdp.terminals[s] && s === 4;
    const isGoal = mdp.terminals[s] && s === 8;
    const v = V ? V[s] : 0;
    const fill = isPit ? "#fecaca" : isGoal ? "#bbf7d0" : (V ? scale(v) : "#f8fafc");
    g.append("rect").attr("x", x).attr("y", y)
      .attr("width", CELL).attr("height", CELL).attr("rx", 3)
      .attr("fill", fill).attr("stroke", "#cbd5e1").attr("stroke-width", 1);
    if (isGoal) g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 5)
      .attr("text-anchor", "middle").attr("font-size", "14px").text("⚑");
    else if (isPit) g.append("text").attr("x", x + CELL/2).attr("y", y + CELL/2 + 5)
      .attr("text-anchor", "middle").attr("font-size", "14px").text("☠");
    else if (V) {
      const bg = scale(v);
      const lum = (() => { const c = d3.color(bg)?.rgb(); if (!c) return 0.5; return (0.299*c.r + 0.587*c.g + 0.114*c.b)/255; })();
      g.append("text").attr("x", x+CELL/2).attr("y", y+CELL/2+4)
        .attr("text-anchor", "middle").attr("font-size", "9px")
        .attr("fill", lum < 0.5 ? "white" : "#1c1e22").attr("font-weight", "600")
        .text(v.toFixed(2));
    }
  }
}

class PolicyGradientLab extends HTMLElement {
  connectedCallback() {
    const { panel, body, setStatus } = createPanel({
      id: "policy-gradient-lab",
      arena: true,
      heavy: true,
      mobileNotice: "View on a wider screen for the full Policy Gradient Lab.",
    });
    body.style.display = "flex"; body.style.flexDirection = "column"; body.style.gap = "12px";

    // Settings strip
    const settings = document.createElement("div");
    settings.style.cssText = "display:flex;align-items:center;gap:16px;flex-wrap:wrap;font-size:12px;padding:4px 0;";

    const makeSlider = (label: string, min: number, max: number, step: number, val: number, unit = "") => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;align-items:center;gap:6px;";
      const lbl = document.createElement("span");
      lbl.style.color = "#64748b"; lbl.textContent = label + " =";
      const valEl = document.createElement("span");
      valEl.style.cssText = "font-family:var(--font-mono);font-weight:600;color:var(--pg-theta);min-width:36px;";
      valEl.textContent = val.toFixed(step < 1 ? 2 : 0) + unit;
      const sl = document.createElement("input");
      sl.type = "range"; sl.min = String(min); sl.max = String(max);
      sl.step = String(step); sl.value = String(val); sl.style.width = "90px";
      sl.addEventListener("input", () => { valEl.textContent = parseFloat(sl.value).toFixed(step < 1 ? 2 : 0) + unit; });
      wrap.append(lbl, sl, valEl);
      return { wrap, sl };
    };

    const { wrap: alphaActorWrap, sl: alphaActorSl } = makeSlider("α_actor", 0.01, 0.5, 0.01, 0.1);
    const { wrap: alphaCriticWrap, sl: alphaCriticSl } = makeSlider("α_critic", 0.01, 0.5, 0.01, 0.2);
    const { wrap: nWrap, sl: nSl } = makeSlider("N", 200, 4000, 100, 2000, " eps");
    const resetBtn = document.createElement("button");
    resetBtn.className = "rl-btn"; resetBtn.textContent = "Train"; resetBtn.style.fontSize = "12px";
    settings.append(alphaActorWrap, alphaCriticWrap, nWrap, resetBtn);
    body.appendChild(settings);

    const loadingEl = document.createElement("div");
    loadingEl.style.cssText = "font-size:13px;color:#64748b;padding:8px 0;";
    loadingEl.textContent = "Click Train to run all three algorithms…";
    body.appendChild(loadingEl);

    // --- Main SVG canvas ---
    const PANEL_H = 200;
    const CURVE_W = W - 40;
    const CM = { top: 24, right: 16, bottom: 36, left: 52 };
    const SVG_H = G3H + 16 + PANEL_H + 12 + PANEL_H + 8;

    const wrap2 = document.createElement("div");
    wrap2.style.overflowX = "auto"; wrap2.style.display = "none";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${SVG_H}`);
    svgEl.style.width = "100%"; svgEl.style.maxWidth = `${W}px`;
    wrap2.appendChild(svgEl); body.appendChild(wrap2);

    const svg = d3.select(svgEl);

    // Policy grids row: 3 grids equally spaced
    const gridOX = [0, 1, 2].map(i => 40 + i * (W - 40) / 3 + (W - 40) / 6 - G3W / 2);
    const gridOY = 8;
    const gGrids = [0, 1, 2].map(i => {
      const g = svg.append("g").attr("transform", `translate(${gridOX[i]},${gridOY})`);
      return g;
    });
    // Grid titles
    const algoLabels = ["Vanilla REINFORCE", "REINFORCE + baseline", "Actor-Critic"];
    const algoColors = [VANILLA_COLOR, BASELINE_COLOR, AC_COLOR];
    for (let i = 0; i < 3; i++) {
      svg.append("text")
        .attr("x", gridOX[i] + G3W / 2).attr("y", gridOY - 4)
        .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
        .attr("fill", algoColors[i]).text(algoLabels[i]);
    }

    // Panel B+C (learning curve + grad norms) below grids
    const ROW2_Y = gridOY + G3H + 20;
    const HALF_W = CURVE_W / 2 - 8;
    const CM_W = HALF_W - CM.left - CM.right;
    const CM_H = PANEL_H - CM.top - CM.bottom;

    const buildCurvePanel = (ox: number, oy: number, title: string, yDomain: [number,number], yLabel: string) => {
      const g = svg.append("g").attr("transform", `translate(${ox + CM.left},${oy + CM.top})`);
      g.append("text").attr("x", CM_W / 2).attr("y", -14)
        .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
        .attr("fill", "var(--rl-ink)").text(title);
      const xSc = d3.scaleLinear().domain([0, 1]).range([0, CM_W]);
      const ySc = d3.scaleLinear().domain(yDomain).range([CM_H, 0]);
      g.append("g").attr("transform", `translate(0,${CM_H})`)
        .call(d3.axisBottom(xSc).ticks(5).tickSize(-CM_H))
        .call(ax => { ax.select(".domain").remove(); ax.selectAll(".tick line").attr("stroke", "#e2e8f0"); });
      g.append("g").call(d3.axisLeft(ySc).ticks(4).tickSize(-CM_W))
        .call(ax => { ax.select(".domain").remove(); ax.selectAll(".tick line").attr("stroke", "#e2e8f0"); });
      g.append("text").attr("x", CM_W / 2).attr("y", CM_H + 28)
        .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#64748b").text("Episode");
      g.append("text").attr("transform", "rotate(-90)").attr("x", -CM_H/2).attr("y", -38)
        .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#64748b").text(yLabel);
      return { g, xSc, ySc };
    };

    const { g: gCurve, xSc: xCurve, ySc: yCurve } = buildCurvePanel(
      20, ROW2_Y, "Learning Curves  V̂(0,0)", [0.4, 0.80], "V̂(s₀=0)"
    );
    const { g: gGrad, xSc: xGrad, ySc: yGrad } = buildCurvePanel(
      20 + HALF_W + 16, ROW2_Y, "Gradient Norms  ‖∇J‖", [0, 0.5], "‖∇J‖"
    );

    // V* lines
    for (const [val, label, color] of [
      [V_STAR, `V*=${V_STAR}`, "#15803d"],
      [V_CAP, `cap≈${V_CAP}`, "#7c3aed"],
    ] as [number, string, string][]) {
      gCurve.append("line").attr("x1", 0).attr("x2", CM_W)
        .attr("y1", yCurve(val)).attr("y2", yCurve(val))
        .attr("stroke", color).attr("stroke-width", 1).attr("stroke-dasharray", "4,3").attr("opacity", 0.6);
      gCurve.append("text").attr("x", CM_W - 2).attr("y", yCurve(val) - 2)
        .attr("text-anchor", "end").attr("font-size", "8px").attr("fill", color).text(label);
    }

    // Panel row 3: episode returns + critic value
    const ROW3_Y = ROW2_Y + PANEL_H + 16;
    const { g: gRet, xSc: xRet, ySc: yRet } = buildCurvePanel(
      20, ROW3_Y, "Episode Returns  G₀", [-1.1, 0.85], "Return G₀"
    );
    for (const [val, color] of [[V_STAR, "#15803d"], [V_CAP, "#7c3aed"]] as [number,string][]) {
      gRet.append("line").attr("x1", 0).attr("x2", CM_W)
        .attr("y1", yRet(val)).attr("y2", yRet(val))
        .attr("stroke", color).attr("stroke-width", 1).attr("stroke-dasharray", "4,3").attr("opacity", 0.4);
    }

    // Critic value grid panel
    svg.append("text").attr("x", 20 + HALF_W + 16 + HALF_W / 2).attr("y", ROW3_Y + 12)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
      .attr("fill", AC_COLOR).text("Actor-Critic Critic  V_φ(s)");
    const gCriticGrid = svg.append("g").attr("transform",
      `translate(${20 + HALF_W + 16 + (HALF_W - G3W) / 2},${ROW3_Y + 18})`);

    // True V reference grid
    svg.append("text").attr("x", 20 + HALF_W + 16 + HALF_W / 2 + G3W + 8).attr("y", ROW3_Y + 12)
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#64748b")
      .text("True V^π_uniform");
    const gTrueGrid = svg.append("g").attr("transform",
      `translate(${20 + HALF_W + 16 + (HALF_W - G3W) / 2 + G3W + 8},${ROW3_Y + 18})`);
    drawCriticGrid(gTrueGrid, TRUE_V as number[]);

    // Scrubber
    const scrubWrap = document.createElement("div");
    scrubWrap.style.cssText = "display:flex;align-items:center;gap:10px;font-size:12px;width:90%;max-width:880px;";
    const scrubLabel = document.createElement("label");
    scrubLabel.style.cssText = "display:flex;align-items:center;gap:8px;flex:1;";
    scrubLabel.textContent = "Episode: ";
    const scrubVal = document.createElement("span");
    scrubVal.style.cssText = "font-family:var(--font-mono);font-weight:600;color:var(--pg-theta);min-width:48px;";
    const scrub = document.createElement("input");
    scrub.type = "range"; scrub.min = "0"; scrub.max = "99"; scrub.value = "99";
    scrub.style.flex = "1";
    scrubLabel.append(scrub, scrubVal);
    scrubWrap.appendChild(scrubLabel);
    scrubWrap.style.display = "none";
    body.appendChild(scrubWrap);

    this.appendChild(panel);

    let results: AlgoResult[] = [];

    const renderSnapshot = (idx: number) => {
      if (results.length === 0) return;
      const nEpisodes = parseInt(nSl.value);
      const ep = Math.round((idx / 99) * (nEpisodes - 1));
      scrubVal.textContent = String(ep);
      setStatus(`ep=${ep}`);

      for (let i = 0; i < 3; i++) {
        const r = results[i];
        const snapIdx = Math.min(idx, (r.probsByEp?.length ?? 1) - 1);
        drawGridPolicy(gGrids[i], r.probsByEp?.[snapIdx] ?? null);
      }
      // Critic grid
      const acResult = results[2];
      const snapIdx = Math.min(idx, (acResult.criticByEp?.length ?? 1) - 1);
      drawCriticGrid(gCriticGrid, acResult.criticByEp?.[snapIdx] ?? null);
    };

    const drawCurves = () => {
      if (results.length === 0) return;
      const nEpisodes = results[0].history.length;

      // Update x scales
      const xUpdater = (xSc: d3.ScaleLinear<number, number>) =>
        xSc.domain([0, nEpisodes - 1]);
      xUpdater(xCurve); xUpdater(xGrad); xUpdater(xRet);

      // Redraw axes
      [gCurve, gGrad, gRet].forEach((g, gi) => {
        const xSc = [xCurve, xGrad, xRet][gi];
        const ySc = [yCurve, yGrad, yRet][gi];
        const w = CM_W, h = CM_H;
        g.select(".x-axis").remove(); g.select(".y-axis").remove();
        g.append("g").attr("class", "x-axis").attr("transform", `translate(0,${h})`)
          .call(d3.axisBottom(xSc).ticks(5).tickSize(-h))
          .call(ax => { ax.select(".domain").remove(); ax.selectAll(".tick line").attr("stroke", "#e2e8f0"); });
        g.append("g").attr("class", "y-axis").call(d3.axisLeft(ySc).ticks(4).tickSize(-w))
          .call(ax => { ax.select(".domain").remove(); ax.selectAll(".tick line").attr("stroke", "#e2e8f0"); });
      });

      const stride = Math.max(1, Math.floor(nEpisodes / 400));
      const eps = d3.range(0, nEpisodes, stride);

      for (const r of results) {
        // Smooth history (running window of 50)
        const W_SMOOTH = 50;
        const smoothed = r.history.map((_, i) => {
          const s0 = Math.max(0, i - W_SMOOTH + 1);
          return r.history.slice(s0, i + 1).reduce((a, b) => a + b, 0) / (i - s0 + 1);
        });

        gCurve.append("path").datum(eps)
          .attr("fill", "none").attr("stroke", r.color).attr("stroke-width", 2)
          .attr("d", d3.line<number>().x(i => xCurve(i)).y(i => yCurve(smoothed[i])));

        // Grad norms smoothed
        const smoothGrad = r.gradNorms.map((_, i) => {
          const s0 = Math.max(0, i - W_SMOOTH + 1);
          return r.gradNorms.slice(s0, i + 1).reduce((a, b) => a + b, 0) / (i - s0 + 1);
        });
        gGrad.append("path").datum(eps)
          .attr("fill", "none").attr("stroke", r.color).attr("stroke-width", 2)
          .attr("d", d3.line<number>().x(i => xGrad(i)).y(i => yGrad(Math.min(smoothGrad[i], 0.5))));

        // Episode returns (sparse bars)
        for (let i = 0; i < r.episodeReturns.length; i += stride * 3) {
          const rv = r.episodeReturns[i];
          const y0 = yRet(Math.max(Math.min(rv, 0.85), -1.1));
          const y1 = yRet(0);
          gRet.append("rect")
            .attr("x", xRet(i)).attr("y", Math.min(y0, y1))
            .attr("width", Math.max(1, xRet(stride * 3) - xRet(0) - 0.5))
            .attr("height", Math.abs(y0 - y1))
            .attr("fill", r.color).attr("opacity", 0.2);
        }
      }

      // Legend
      const leg = gCurve.append("g").attr("transform", `translate(${CM_W - 140}, 4)`);
      for (const [i, r] of results.entries()) {
        leg.append("line").attr("x1", 0).attr("x2", 16).attr("y1", i * 16).attr("y2", i * 16)
          .attr("stroke", r.color).attr("stroke-width", 2);
        leg.append("text").attr("x", 20).attr("y", i * 16 + 4)
          .attr("font-size", "9px").attr("fill", "var(--rl-ink)").text(r.label);
      }
    };

    const train = () => {
      loadingEl.textContent = "Training all three algorithms…";
      loadingEl.style.display = "block";
      wrap2.style.display = "none";
      scrubWrap.style.display = "none";

      // Clear old curves
      gCurve.selectAll("path").remove(); gCurve.selectAll("g.x-axis,g.y-axis").remove();
      gGrad.selectAll("path").remove(); gGrad.selectAll("g.x-axis,g.y-axis").remove();
      gRet.selectAll("path,rect").remove(); gRet.selectAll("g.x-axis,g.y-axis").remove();

      setTimeout(() => {
        const alphaA = parseFloat(alphaActorSl.value);
        const alphaC = parseFloat(alphaCriticSl.value);
        const N = parseInt(nSl.value);
        results = runAll(N, alphaA, alphaC, 0.9);
        drawCurves();
        wrap2.style.display = "block";
        scrubWrap.style.display = "flex";
        loadingEl.style.display = "none";
        scrub.max = String(99);
        scrub.value = "99";
        renderSnapshot(99);
        setStatus(`N=${N} α_a=${alphaA.toFixed(2)} α_c=${alphaC.toFixed(2)}`);
      }, 0);
    };

    resetBtn.addEventListener("click", train);
    scrub.addEventListener("input", () => renderSnapshot(parseInt(scrub.value)));
  }
}

customElements.define("pg-policy-gradient-lab", PolicyGradientLab);
