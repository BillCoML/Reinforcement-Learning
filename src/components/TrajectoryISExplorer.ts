/**
 * V3 — Trajectory IS Explorer (centerpiece, 960px wide).
 *
 * Four synchronized panels:
 *   A (top-left)  — two policy gridworld renderings + matching-prob readout
 *   B (top-right) — animated trajectory with running IS weight
 *   C (bot-left)  — histogram of contributions + running IS averages
 *   D (bot-right) — 50-trial boxplot (precomputed JSON or live recomputation)
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy, deterministicPolicy } from "../mdp/gridworld";
import { optimalPolicy, epsilonSoftOptimal } from "../mdp/policies";
import { rollout } from "../mdp/rollout";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { trajectoryISWeight, effectiveSampleSize } from "../importance-sampling/estimators";
import { mulberry32 } from "../importance-sampling/gaussian";
import { GridworldRenderer } from "./GridworldRenderer";
import { RIGHT, DOWN, type MDP, type Policy } from "../mdp/types";

// ---------------------------------------------------------------------------
// Types & constants
// ---------------------------------------------------------------------------

type PolicyKind = "uniform" | "det-optimal" | "epsilon-soft" | "all-down" | "all-right";

interface Sample { weight: number; G0: number; }

const DEFAULT_MDP = buildGridworld({ slippery: false, gamma: 0.9 });
const TRUE_VALUE = policyEvaluationExact(DEFAULT_MDP, optimalPolicy(DEFAULT_MDP))[0];
const CHARACTERISTIC_VALUE = 256 * Math.pow(0.9, 3); // 186.624

// ---------------------------------------------------------------------------
// Policy helpers
// ---------------------------------------------------------------------------

function makePolicy(kind: PolicyKind, mdp: MDP, epsilon: number): Policy {
  switch (kind) {
    case "uniform":      return uniformPolicy(mdp);
    case "det-optimal":  return optimalPolicy(mdp);
    case "epsilon-soft": return epsilonSoftOptimal(mdp, epsilon);
    case "all-down":     return deterministicPolicy(mdp, Array(9).fill(DOWN));
    case "all-right":    return deterministicPolicy(mdp, Array(9).fill(RIGHT));
  }
}

function matchProbability(targetKind: PolicyKind, behaviorKind: PolicyKind, epsilon: number): number {
  // Analytical formula for default case; empirical otherwise
  if (targetKind === "det-optimal" && behaviorKind === "uniform") return 1 / 256;
  if (targetKind === "uniform" && behaviorKind === "uniform") return 1;
  // For epsilon-soft target with uniform behavior on 4-step path:
  if (targetKind === "epsilon-soft" && behaviorKind === "uniform") {
    const p = (1 - epsilon + epsilon / 4) * 0.25; // per step prob
    return Math.pow(p, 4);
  }
  return NaN; // fallback
}

// ---------------------------------------------------------------------------
// Boxplot helpers
// ---------------------------------------------------------------------------

function boxStats(arr: number[]) {
  const sorted = arr.slice().sort(d3.ascending);
  return {
    min: sorted[0],
    q1: d3.quantile(sorted, 0.25) ?? 0,
    median: d3.quantile(sorted, 0.5) ?? 0,
    q3: d3.quantile(sorted, 0.75) ?? 0,
    max: sorted[sorted.length - 1],
    mean: d3.mean(sorted) ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export class TrajectoryISExplorer extends HTMLElement {
  // Config
  private behaviorKind: PolicyKind = "uniform";
  private targetKind: PolicyKind = "det-optimal";
  private epsilon = 0.1;
  private gamma = 0.9;
  private N: 100 | 1000 | 10000 = 1000;

  // MDPs & policies
  private mdp!: MDP;
  private behaviorPolicy!: Policy;
  private targetPolicy!: Policy;

  // Accumulated samples for Panel C
  private samples: Sample[] = [];
  private rng = mulberry32(42);

  // Panel B animation
  private animSteps: ReturnType<typeof rollout> = [];
  private animIdx = 0;
  private animRho = 1;
  private animTimer = 0;
  private animSpeedMs = 300;

  // Panel D
  private boxplotData: { ordinary: number[]; weighted: number[] } | null = null;
  private boxplotComputing = false;

  // DOM refs
  private panelAEl!: HTMLElement;
  private panelBEl!: HTMLElement;
  private panelCEl!: HTMLElement;
  private panelDEl!: HTMLElement;
  private rendererB!: GridworldRenderer;
  private panelCsvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private panelDsvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private panelBInfo!: HTMLElement;
  private panelCReadout!: HTMLElement;
  private matchProbEl!: HTMLElement;
  private policyARenderers: [GridworldRenderer | null, GridworldRenderer | null] = [null, null];
  private setStatus!: (t: string) => void;

  connectedCallback() { this.build(); }

  private rebuild() {
    this.mdp = buildGridworld({ slippery: false, gamma: this.gamma });
    this.behaviorPolicy = makePolicy(this.behaviorKind, this.mdp, this.epsilon);
    this.targetPolicy = makePolicy(this.targetKind, this.mdp, this.epsilon);
    this.samples = [];
    this.rng = mulberry32(42);
    this.animSteps = [];
    this.animIdx = 0;
    this.animRho = 1;
    this.boxplotData = null;
  }

  // ---- Panel A ----
  private buildPanelA(container: HTMLElement) {
    container.innerHTML = "";
    const title = document.createElement("div");
    title.className = "is-panel-title";
    title.textContent = "A — Policy Pair";
    container.appendChild(title);

    const grids = document.createElement("div");
    grids.style.cssText = "display:flex;gap:12px;justify-content:center;";
    container.appendChild(grids);

    for (let i = 0; i < 2; i++) {
      const wrap = document.createElement("div");
      const lbl = document.createElement("div");
      lbl.style.cssText = `text-align:center;font-size:10px;margin-bottom:4px;color:var(--${i===0?"is-proposal":"is-target"})`;
      lbl.textContent = i === 0 ? "Behavior π_b" : "Target π_t";
      wrap.appendChild(lbl);
      grids.appendChild(wrap);
      const r = new GridworldRenderer(wrap, {
        mdp: this.mdp,
        policy: i === 0 ? this.behaviorPolicy : this.targetPolicy,
        showArrows: true,
        cellPx: 60,
      });
      this.policyARenderers[i] = r;
    }

    this.matchProbEl = document.createElement("div");
    this.matchProbEl.style.cssText = "text-align:center;font-family:var(--rl-font-mono);font-size:12px;margin-top:8px;";
    container.appendChild(this.matchProbEl);
    this.updateMatchProb();
  }

  private updateMatchProb() {
    const p = matchProbability(this.targetKind, this.behaviorKind, this.epsilon);
    if (isNaN(p)) {
      this.matchProbEl.textContent = "match prob: (varies)";
    } else {
      this.matchProbEl.innerHTML =
        `match prob: <strong>1/${Math.round(1/p)}</strong> ≈ ${p.toExponential(2)}`;
    }
  }

  // ---- Panel B ----
  private buildPanelB(container: HTMLElement) {
    container.innerHTML = "";
    const title = document.createElement("div");
    title.className = "is-panel-title";
    title.textContent = "B — Trajectory Animation";
    container.appendChild(title);

    // Gridworld for trajectory animation
    this.rendererB = new GridworldRenderer(container, {
      mdp: this.mdp,
      cellPx: 72,
      showArrows: false,
    });

    // Info area
    this.panelBInfo = document.createElement("div");
    this.panelBInfo.className = "is-panel-info rl-mono";
    this.panelBInfo.style.cssText = "font-size:11px;min-height:80px;padding:8px;background:var(--rl-surface-2);border-radius:4px;margin-top:8px;";
    container.appendChild(this.panelBInfo);

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;margin-top:8px;align-items:center;";
    const sampleBtn = document.createElement("button");
    sampleBtn.className = "rl-btn";
    sampleBtn.textContent = "Sample trajectory";
    sampleBtn.addEventListener("click", () => this.sampleAndAnimate());
    const speedLabel = document.createElement("label");
    speedLabel.style.cssText = "font-size:11px;display:flex;align-items:center;gap:6px;";
    speedLabel.innerHTML = `Speed <input type="range" min="50" max="600" step="50" value="${600 - this.animSpeedMs + 50}" style="width:80px">`;
    const speedInput = speedLabel.querySelector("input")!;
    speedInput.addEventListener("input", () => {
      this.animSpeedMs = 600 - parseInt(speedInput.value) + 50;
    });
    btnRow.appendChild(sampleBtn);
    btnRow.appendChild(speedLabel);
    container.appendChild(btnRow);
  }

  private sampleAndAnimate() {
    clearTimeout(this.animTimer);
    this.animSteps = rollout(this.mdp, this.behaviorPolicy, 0, 20, this.rng);
    this.animIdx = 0;
    this.animRho = 1;
    this.rendererB.update({
      mdp: this.mdp, policy: this.behaviorPolicy, showArrows: false,
      startState: 0,
    });
    this.stepAnimation();
  }

  private stepAnimation() {
    const steps = this.animSteps;
    if (this.animIdx >= steps.length) {
      // Trajectory done — add to Panel C samples
      const traj = steps.map(({ s, a }) => ({ s, a }));
      const piT = (s: number, a: number) => this.targetPolicy.pi[s][a];
      const piB = (s: number, a: number) => this.behaviorPolicy.pi[s][a];
      const w = trajectoryISWeight(traj, piT, piB);
      const G0 = steps.reduce((acc, { r }, t) => acc + Math.pow(this.gamma, t) * r, 0);
      this.samples.push({ weight: w, G0 });
      this.renderPanelC();
      this.panelBInfo.innerHTML =
        `<span style="color:var(--rl-ink-faint)">Complete: </span>` +
        `ρ = <b>${this.animRho.toFixed(0)}</b>  G₀ = <b>${G0.toFixed(4)}</b>  ` +
        `contribution = <b style="color:${w>0?"var(--is-ordinary)":"var(--rl-ink-muted)"}">${(w * G0).toFixed(4)}</b>`;
      return;
    }

    const step = steps[this.animIdx];
    const { s, a, r } = step;
    const piT = (ss: number, aa: number) => this.targetPolicy.pi[ss][aa];
    const piB = (ss: number, aa: number) => this.behaviorPolicy.pi[ss][aa];
    const pb = piB(s, a);
    const ratio = pb === 0 ? 0 : piT(s, a) / pb;
    const prevRho = this.animRho;
    this.animRho *= ratio;

    const accepted = ratio > 0;
    const actionNames = ["↑", "→", "↓", "←"];

    this.rendererB.update({
      mdp: this.mdp, showArrows: false,
      highlightState: s,
    });

    this.panelBInfo.innerHTML =
      `step <b>${this.animIdx}</b>: s=(${Math.floor(s/3)},${s%3}) a=${actionNames[a]} r=${r.toFixed(2)}<br>` +
      `ratio: π_t/π_b = ${piT(s,a).toFixed(2)}/${pb.toFixed(2)} = ` +
      `<b style="color:${accepted?"var(--is-weight)":"var(--is-explosion)"}">${ratio.toFixed(2)}</b><br>` +
      `ρ: ${prevRho.toFixed(2)} × ${ratio.toFixed(2)} = ` +
      `<b style="color:${accepted?"var(--is-ordinary)":"var(--is-explosion)"}">${this.animRho.toFixed(2)}</b>`;

    if (!accepted) {
      // Trajectory killed — highlight rejection
      this.rendererB.update({ highlightState: s });
      setTimeout(() => {
        // Still add to samples with weight 0
        const G0 = steps.slice(0, this.animIdx + 1)
          .reduce((acc, { r: rr }, t) => acc + Math.pow(this.gamma, t) * rr, 0);
        this.samples.push({ weight: 0, G0 });
        this.renderPanelC();
        this.panelBInfo.innerHTML +=
          `<br><span style="color:var(--is-explosion)">✕ REJECTED — ρ = 0</span>`;
      }, this.animSpeedMs);
      return;
    }

    this.animIdx++;
    this.animTimer = window.setTimeout(() => this.stepAnimation(), this.animSpeedMs);
  }

  // ---- Panel C ----
  private buildPanelC(container: HTMLElement) {
    container.innerHTML = "";
    const title = document.createElement("div");
    title.className = "is-panel-title";
    title.textContent = "C — Estimator Distribution";
    container.appendChild(title);

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", "0 0 440 220");
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);
    this.panelCsvg = d3.select(svgEl as SVGSVGElement);

    this.panelCReadout = document.createElement("div");
    this.panelCReadout.className = "rl-mono";
    this.panelCReadout.style.cssText = "font-size:10px;padding:4px 0;";
    container.appendChild(this.panelCReadout);

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;";
    for (const n of [100, 1000, 10000] as const) {
      const btn = document.createElement("button");
      btn.className = "rl-btn";
      btn.textContent = `+${n.toLocaleString()} trajs`;
      btn.addEventListener("click", () => this.runBatch(n));
      btnRow.appendChild(btn);
    }
    const resetBtn = document.createElement("button");
    resetBtn.className = "rl-btn rl-btn--ghost";
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => { this.samples = []; this.renderPanelC(); });
    btnRow.appendChild(resetBtn);
    container.appendChild(btnRow);

    this.renderPanelC();
  }

  private runBatch(n: number) {
    const piT = (s: number, a: number) => this.targetPolicy.pi[s][a];
    const piB = (s: number, a: number) => this.behaviorPolicy.pi[s][a];
    for (let i = 0; i < n; i++) {
      const steps = rollout(this.mdp, this.behaviorPolicy, 0, 20, this.rng);
      const w = trajectoryISWeight(steps.map(({ s, a }) => ({ s, a })), piT, piB);
      const G0 = steps.reduce((acc, { r }, t) => acc + Math.pow(this.gamma, t) * r, 0);
      this.samples.push({ weight: w, G0 });
    }
    this.renderPanelC();
    if (n >= 1000) this.recomputeBoxplot();
  }

  private renderPanelC() {
    const svg = this.panelCsvg;
    svg.selectAll("*").remove();

    const CW = 440, CH = 220;
    const MG = { top: 24, right: 16, bottom: 36, left: 52 };
    const IW = CW - MG.left - MG.right;
    const IH = CH - MG.top - MG.bottom;

    const g = svg.append("g").attr("transform", `translate(${MG.left},${MG.top})`);

    const n = this.samples.length;
    const weights = this.samples.map((s) => s.weight);
    const contributions = this.samples.map((s) => s.weight * s.G0);

    if (n === 0) {
      g.append("text").attr("x", IW / 2).attr("y", IH / 2)
        .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-muted)")
        .attr("font-size", "12px")
        .text("Sample trajectories using the buttons below");
      return;
    }

    // Histogram of contributions
    const nonZero = contributions.filter((c) => c > 0);
    const domainMax = nonZero.length > 0
      ? Math.max(...nonZero) * 1.05
      : CHARACTERISTIC_VALUE * 1.2;

    const xScale = d3.scaleLinear().domain([0, domainMax]).range([0, IW]);
    const bins = d3.bin<number, number>()
      .value((d) => d)
      .domain([0, domainMax])
      .thresholds(30)(contributions.map((c) => Math.min(c, domainMax)));

    const yMax = Math.max(d3.max(bins, (b) => b.length) ?? 1, 1);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([IH, 0]);

    bins.forEach((b) => {
      const isNonZero = (b.x0 ?? 0) > 0;
      g.append("rect")
        .attr("x", xScale(b.x0 ?? 0))
        .attr("y", yScale(b.length))
        .attr("width", Math.max(0, xScale(b.x1 ?? 0) - xScale(b.x0 ?? 0) - 1))
        .attr("height", IH - yScale(b.length))
        .attr("fill", isNonZero ? "var(--is-ordinary)" : "var(--rl-border)")
        .attr("opacity", isNonZero ? 0.7 : 0.5);
    });

    // Running estimates
    const stride = Math.max(1, Math.floor(n / 200));
    let cumOrd = 0, cumW = 0, cumWG = 0;
    const ordPts: [number, number][] = [];
    const wtPts: [number, number][] = [];

    for (let i = 0; i < n; i++) {
      const { weight: w, G0 } = this.samples[i];
      cumOrd += w * G0;
      cumW += w;
      cumWG += w * G0;
      if (i % stride === 0 || i === n - 1) {
        const ord = cumOrd / (i + 1);
        const wt = cumW > 0 ? cumWG / cumW : 0;
        ordPts.push([i + 1, ord]);
        wtPts.push([i + 1, wt]);
      }
    }

    const xRun = d3.scaleLinear().domain([1, n]).range([0, IW]);
    const estMax = Math.max(TRUE_VALUE * 2, ...ordPts.map(([, v]) => v), ...wtPts.map(([, v]) => v));
    const estMin = Math.min(0, ...ordPts.map(([, v]) => v), ...wtPts.map(([, v]) => v));
    const yRun = d3.scaleLinear().domain([estMin, estMax]).range([IH, 0]);

    // True value reference
    if (yRun.domain()[0] <= TRUE_VALUE && TRUE_VALUE <= yRun.domain()[1]) {
      g.append("line")
        .attr("x1", 0).attr("x2", IW)
        .attr("y1", yRun(TRUE_VALUE)).attr("y2", yRun(TRUE_VALUE))
        .attr("stroke", "var(--is-weight)").attr("stroke-dasharray", "5 3")
        .attr("stroke-width", 1.5);
      g.append("text").attr("x", IW - 2).attr("y", yRun(TRUE_VALUE) - 3)
        .attr("text-anchor", "end").attr("font-size", "9px")
        .attr("fill", "var(--is-weight)").text(`V*=${TRUE_VALUE.toFixed(4)}`);
    }

    const mkLine = (pts: [number, number][]) =>
      d3.line<[number, number]>().x(([i]) => xRun(i)).y(([, v]) => {
        const [dMin, dMax] = yRun.domain() as number[];
        const clamped = Math.max(dMin, Math.min(dMax, v));
        return yRun(clamped);
      })(pts);

    if (ordPts.length > 1) {
      g.append("path").attr("d", mkLine(ordPts))
        .attr("fill", "none").attr("stroke", "var(--is-ordinary)")
        .attr("stroke-width", 1.5).attr("opacity", 0.9);
    }
    if (wtPts.length > 1) {
      g.append("path").attr("d", mkLine(wtPts))
        .attr("fill", "none").attr("stroke", "var(--is-weighted)")
        .attr("stroke-width", 1.5).attr("opacity", 0.9);
    }

    // Axes
    g.append("g").attr("transform", `translate(0,${IH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0)
        .tickFormat((d) => +d > 0 ? d3.format(".0f")(+d) : "0"))
      .selectAll("text").style("font-size", "9px");
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(4).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    // Legend
    const lx = IW - 120;
    g.append("line").attr("x1", lx).attr("x2", lx+18).attr("y1", 8).attr("y2", 8)
      .attr("stroke", "var(--is-ordinary)").attr("stroke-width", 2);
    g.append("text").attr("x", lx+22).attr("y", 11).attr("font-size", "9px")
      .attr("fill", "var(--is-ordinary)").text("ordinary IS");
    g.append("line").attr("x1", lx).attr("x2", lx+18).attr("y1", 22).attr("y2", 22)
      .attr("stroke", "var(--is-weighted)").attr("stroke-width", 2);
    g.append("text").attr("x", lx+22).attr("y", 25).attr("font-size", "9px")
      .attr("fill", "var(--is-weighted)").text("weighted IS");

    // Readout
    const lastOrd = ordPts.length > 0 ? ordPts[ordPts.length - 1][1] : 0;
    const lastWt = wtPts.length > 0 ? wtPts[wtPts.length - 1][1] : 0;
    const ess = effectiveSampleSize(weights);
    const nz = weights.filter((w) => w > 0).length;
    this.panelCReadout.innerHTML =
      `N=${n}  nonZero=${nz}  ord=${lastOrd.toFixed(4)}  ` +
      `wt=${lastWt.toFixed(4)}  ESS/N=${(ess/n*100).toFixed(1)}%`;

    this.setStatus?.(`N=${n}  nonZero=${nz}  ord=${lastOrd.toFixed(4)}`);
  }

  // ---- Panel D ----
  private buildPanelD(container: HTMLElement) {
    container.innerHTML = "";
    const title = document.createElement("div");
    title.className = "is-panel-title";
    title.textContent = "D — 50-Trial Distribution (N = " + this.N.toLocaleString() + ")";
    container.appendChild(title);

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", "0 0 440 200");
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);
    this.panelDsvg = d3.select(svgEl as SVGSVGElement);

    // N selector
    const nRow = document.createElement("div");
    nRow.style.cssText = "display:flex;gap:8px;margin-top:8px;align-items:center;font-size:11px;";
    nRow.innerHTML = `<span style="color:var(--rl-ink-faint)">N per trial:</span>`;
    for (const n of [100, 1000, 10000] as const) {
      const btn = document.createElement("button");
      btn.className = "rl-btn" + (n === this.N ? "" : " rl-btn--ghost");
      btn.textContent = n.toLocaleString();
      btn.addEventListener("click", () => {
        this.N = n;
        container.querySelector(".is-panel-title")!.textContent =
          "D — 50-Trial Distribution (N = " + this.N.toLocaleString() + ")";
        nRow.querySelectorAll("button").forEach((b, i) => {
          b.className = "rl-btn" + ([100, 1000, 10000][i] === n ? "" : " rl-btn--ghost");
        });
        this.recomputeBoxplot();
      });
      nRow.appendChild(btn);
    }
    container.appendChild(nRow);

    this.loadPrecomputedBoxplot();
  }

  private async loadPrecomputedBoxplot() {
    // Load the default-config precomputed stats from JSON
    try {
      const resp = await fetch("/data/is/gridworld_is_stats.json");
      const data = await resp.json() as {
        table: Array<{
          N: number;
          boxplotOrdinary: number[];
          boxplotWeighted: number[];
          trueVal: number;
        }>;
      };
      const row = data.table.find((r) => r.N === this.N);
      if (row) {
        this.boxplotData = { ordinary: row.boxplotOrdinary, weighted: row.boxplotWeighted };
        this.renderPanelD();
      }
    } catch (_) {
      this.recomputeBoxplot();
    }
  }

  private recomputeBoxplot() {
    if (this.boxplotComputing) return;
    this.boxplotComputing = true;
    this.renderPanelD(); // show spinner
    setTimeout(() => {
      const piT = (s: number, a: number) => this.targetPolicy.pi[s][a];
      const piB = (s: number, a: number) => this.behaviorPolicy.pi[s][a];
      const rng = mulberry32(99);
      const ordinary: number[] = [];
      const weighted: number[] = [];
      for (let trial = 0; trial < 50; trial++) {
        const ws: number[] = [], g0s: number[] = [];
        for (let i = 0; i < this.N; i++) {
          const steps = rollout(this.mdp, this.behaviorPolicy, 0, 20, rng);
          const w = trajectoryISWeight(steps.map(({ s, a }) => ({ s, a })), piT, piB);
          const G0 = steps.reduce((acc, { r }, t) => acc + Math.pow(this.gamma, t) * r, 0);
          ws.push(w);
          g0s.push(G0);
        }
        const ordEst = ws.reduce((s, w, i) => s + w * g0s[i], 0) / this.N;
        const wSum = ws.reduce((s, w) => s + w, 0);
        const wtEst = wSum > 0 ? ws.reduce((s, w, i) => s + w * g0s[i], 0) / wSum : 0;
        ordinary.push(ordEst);
        weighted.push(wtEst);
      }
      this.boxplotData = { ordinary, weighted };
      this.boxplotComputing = false;
      this.renderPanelD();
    }, 0);
  }

  private renderPanelD() {
    const svg = this.panelDsvg;
    svg.selectAll("*").remove();

    const CW = 440, CH = 200;
    const MG = { top: 20, right: 16, bottom: 36, left: 52 };
    const IW = CW - MG.left - MG.right;
    const IH = CH - MG.top - MG.bottom;

    const g = svg.append("g").attr("transform", `translate(${MG.left},${MG.top})`);

    if (this.boxplotComputing) {
      g.append("text").attr("x", IW / 2).attr("y", IH / 2)
        .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-muted)")
        .attr("font-size", "12px").text("Computing 50 trials…");
      return;
    }
    if (!this.boxplotData) {
      g.append("text").attr("x", IW / 2).attr("y", IH / 2)
        .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-muted)")
        .attr("font-size", "12px").text("Loading…");
      return;
    }

    const { ordinary, weighted } = this.boxplotData;
    const allVals = [...ordinary, ...weighted];
    const vMin = Math.min(...allVals, 0);
    const vMax = Math.max(...allVals, TRUE_VALUE * 1.5);
    const yScale = d3.scaleLinear().domain([vMin, vMax]).range([IH, 0]);

    // True value ref
    g.append("line").attr("x1", 0).attr("x2", IW)
      .attr("y1", yScale(TRUE_VALUE)).attr("y2", yScale(TRUE_VALUE))
      .attr("stroke", "var(--is-weight)").attr("stroke-dasharray", "5 3")
      .attr("stroke-width", 1.5);
    g.append("text").attr("x", IW - 2).attr("y", yScale(TRUE_VALUE) - 3)
      .attr("text-anchor", "end").attr("font-size", "9px")
      .attr("fill", "var(--is-weight)").text(`V*=${TRUE_VALUE.toFixed(4)}`);

    // Draw boxplots
    const drawBox = (data: number[], cx: number, color: string) => {
      const st = boxStats(data);
      const bw = 40;
      // whiskers
      g.append("line").attr("x1", cx).attr("x2", cx)
        .attr("y1", yScale(st.min)).attr("y2", yScale(st.max))
        .attr("stroke", color).attr("stroke-width", 1.5).attr("opacity", 0.6);
      // box
      g.append("rect")
        .attr("x", cx - bw / 2).attr("y", yScale(st.q3))
        .attr("width", bw).attr("height", yScale(st.q1) - yScale(st.q3))
        .attr("fill", color).attr("fill-opacity", 0.2)
        .attr("stroke", color).attr("stroke-width", 1.5);
      // median
      g.append("line").attr("x1", cx - bw / 2).attr("x2", cx + bw / 2)
        .attr("y1", yScale(st.median)).attr("y2", yScale(st.median))
        .attr("stroke", color).attr("stroke-width", 2.5);
      // mean dot
      g.append("circle").attr("cx", cx).attr("cy", yScale(st.mean))
        .attr("r", 3).attr("fill", color);
    };

    const cx1 = IW * 0.3;
    const cx2 = IW * 0.7;
    drawBox(ordinary, cx1, "var(--is-ordinary)");
    drawBox(weighted, cx2, "var(--is-weighted)");

    // Labels
    g.append("text").attr("x", cx1).attr("y", IH + 16)
      .attr("text-anchor", "middle").attr("font-size", "10px")
      .attr("fill", "var(--is-ordinary)").text("ordinary IS");
    g.append("text").attr("x", cx2).attr("y", IH + 16)
      .attr("text-anchor", "middle").attr("font-size", "10px")
      .attr("fill", "var(--is-weighted)").text("weighted IS");

    // Stats annotation
    const ordSt = boxStats(ordinary);
    const wtSt = boxStats(weighted);
    g.append("text").attr("x", cx1).attr("y", IH + 28)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-faint)")
      .text(`SD=${d3.deviation(ordinary)?.toFixed(3)}`);
    g.append("text").attr("x", cx2).attr("y", IH + 28)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-faint)")
      .text(`SD=${d3.deviation(weighted)?.toFixed(3)}`);

    // Y axis
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0)
        .tickFormat((d) => d3.format(".3f")(+d)))
      .selectAll("text").style("font-size", "9px");

    void ordSt; void wtSt;
  }

  // ---- Main build ----
  private build() {
    this.innerHTML = "";
    this.rebuild();

    const { panel, body, setStatus } = createPanel({
      id: "trajectory-is-explorer",
      arena: true,
      heavy: false,
    });
    this.setStatus = setStatus;

    // CSS for the 4-panel grid
    body.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:16px;";

    // Panel A
    this.panelAEl = document.createElement("div");
    this.panelAEl.className = "is-sub-panel";
    this.buildPanelA(this.panelAEl);
    body.appendChild(this.panelAEl);

    // Panel B
    this.panelBEl = document.createElement("div");
    this.panelBEl.className = "is-sub-panel";
    this.buildPanelB(this.panelBEl);
    body.appendChild(this.panelBEl);

    // Panel C
    this.panelCEl = document.createElement("div");
    this.panelCEl.className = "is-sub-panel";
    this.buildPanelC(this.panelCEl);
    body.appendChild(this.panelCEl);

    // Panel D
    this.panelDEl = document.createElement("div");
    this.panelDEl.className = "is-sub-panel";
    this.buildPanelD(this.panelDEl);
    body.appendChild(this.panelDEl);

    // Policy selectors — shared controls above the grid
    this.buildPolicyControls(body);

    this.appendChild(panel);
  }

  private buildPolicyControls(body: HTMLElement) {
    // Insert controls before the grid panels
    const controlBar = document.createElement("div");
    controlBar.style.cssText =
      "grid-column:1/-1;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;" +
      "padding:12px;background:var(--rl-surface-2);border-radius:6px;font-size:11px;";

    const mkSelect = (label: string, defaultVal: PolicyKind, onChange: (v: PolicyKind) => void) => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:2px;";
      wrap.textContent = label;
      const sel = document.createElement("select");
      sel.className = "rl-select";
      for (const [v, t] of [
        ["uniform", "Uniform random"],
        ["det-optimal", "Deterministic optimal"],
        ["epsilon-soft", "ε-soft optimal"],
        ["all-down", "All-down"],
        ["all-right", "All-right"],
      ] as [PolicyKind, string][]) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = t;
        if (v === defaultVal) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.addEventListener("change", () => onChange(sel.value as PolicyKind));
      wrap.appendChild(sel);
      return wrap;
    };

    const epsilonWrap = document.createElement("label");
    epsilonWrap.style.cssText = "display:flex;flex-direction:column;gap:2px;";
    epsilonWrap.innerHTML = `ε <input type="range" min="0.01" max="0.5" step="0.01"
      value="${this.epsilon}" style="width:80px">
      <span class="rl-mono" id="v3-eps-val">${this.epsilon.toFixed(2)}</span>`;
    const epsSlider = epsilonWrap.querySelector("input")!;
    const epsVal = epsilonWrap.querySelector<HTMLElement>("#v3-eps-val")!;

    const gammaWrap = document.createElement("label");
    gammaWrap.style.cssText = "display:flex;flex-direction:column;gap:2px;";
    gammaWrap.innerHTML = `γ <input type="range" min="0.5" max="0.99" step="0.01"
      value="${this.gamma}" style="width:80px">
      <span class="rl-mono" id="v3-gam-val">${this.gamma.toFixed(2)}</span>`;
    const gammaSlider = gammaWrap.querySelector("input")!;
    const gammaVal = gammaWrap.querySelector<HTMLElement>("#v3-gam-val")!;

    const reloadAll = () => {
      this.rebuild();
      this.buildPanelA(this.panelAEl);
      this.buildPanelC(this.panelCEl);
      this.buildPanelD(this.panelDEl);
      this.buildPanelB(this.panelBEl);
      this.loadPrecomputedBoxplot();
    };

    controlBar.appendChild(mkSelect("Behavior π_b", this.behaviorKind, (v) => {
      this.behaviorKind = v;
      reloadAll();
    }));
    controlBar.appendChild(mkSelect("Target π_t", this.targetKind, (v) => {
      this.targetKind = v;
      reloadAll();
    }));
    controlBar.appendChild(epsilonWrap);
    controlBar.appendChild(gammaWrap);

    epsSlider.addEventListener("input", () => {
      this.epsilon = parseFloat(epsSlider.value);
      epsVal.textContent = this.epsilon.toFixed(2);
      reloadAll();
    });
    gammaSlider.addEventListener("input", () => {
      this.gamma = parseFloat(gammaSlider.value);
      gammaVal.textContent = this.gamma.toFixed(2);
      reloadAll();
    });

    // Insert before first child
    body.insertBefore(controlBar, body.firstChild);
  }
}

customElements.define("trajectory-is-explorer", TrajectoryISExplorer);
