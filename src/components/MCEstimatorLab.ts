/**
 * V3 — MC Estimator Lab. CENTERPIECE (arena, 960px).
 * Panels: A (mode selector + policy info), B (V̂ heatmap), C (visit counts),
 *         D (V̂(0,0) convergence trace), E (50-trial statistics), F (last trajectory).
 * The on-policy / off-policy toggle is the central pedagogical feature.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { optimalPolicy } from "../mdp/policies";
import { rollout, type Step } from "../mdp/rollout";
import { computeReturns } from "../monte-carlo/returns";
import { makeValueColorScale } from "./value-scale";
import { prefersReducedMotion } from "./base";
import { rc, idx, GRID_SIZE, type Policy } from "../mdp/types";

const _mdp = buildGridworld();
const _uniform = uniformPolicy(_mdp);
const _optimal = optimalPolicy(_mdp);

type Mode = "uniform" | "optimal" | "off-weighted" | "off-ordinary";

const TRUE_V: Record<Mode, number> = {
  uniform: -0.4205,
  optimal: 0.7290,
  "off-weighted": 0.7290,
  "off-ordinary": 0.7290,
};

const BEHAVIOR_POLICY: Policy = _uniform;
const TARGET_POLICY: Policy = _optimal;

const NS_SVG = "http://www.w3.org/2000/svg";
const NS = _mdp.nS;
const CELL_B = 78;   // px per cell for Panel B
const CELL_C = 58;   // px per cell for Panel C
const MAX_HIST = 600;
const CHUNK = 8;
const TICK_MS = 90;

/** Run one rollout under `policy` from state 0. */
function oneEpisode(policy: Policy, rng: () => number): Step[] {
  return rollout(_mdp, policy, 0, 200, rng);
}

// ── Lab state ──────────────────────────────────────────────────────────────────

interface LabState {
  V: number[];
  visits: number[];
  episode: number;
  history: number[];   // V̂(0,0) per episode
  lastTraj: Step[];
  // Off-policy accumulators (weighted IS)
  numArr: number[];
  denArr: number[];     // weighted: Σρ ; ordinary: count
  sumW: number[];
  sumW2: number[];
}

function freshState(): LabState {
  return {
    V: new Array(NS).fill(0),
    visits: new Array(NS).fill(0),
    episode: 0,
    history: [],
    lastTraj: [],
    numArr: new Array(NS).fill(0),
    denArr: new Array(NS).fill(0),
    sumW: new Array(NS).fill(0),
    sumW2: new Array(NS).fill(0),
  };
}

/** Update state with one on-policy episode (first-visit). */
function applyOnPolicy(st: LabState, traj: Step[], firstVisit: boolean): void {
  if (traj.length === 0) return;
  const Gs = computeReturns(traj.map(s => s.r), _mdp.gamma);
  const seen = new Set<number>();
  for (let t = 0; t < traj.length; t++) {
    const s = traj[t].s;
    if (firstVisit && seen.has(s)) continue;
    seen.add(s);
    st.visits[s] += 1;
    st.V[s] += (Gs[t] - st.V[s]) / st.visits[s];
  }
  st.episode += 1;
  st.history.push(st.V[0]);
  if (st.history.length > MAX_HIST) st.history.shift();
  st.lastTraj = traj;
}

/** Update state with one off-policy episode (IS). */
function applyOffPolicy(st: LabState, traj: Step[], weighted: boolean): void {
  const T = traj.length;
  if (T === 0) { st.episode += 1; return; }
  const Gs = computeReturns(traj.map(s => s.r), _mdp.gamma);

  // backward suffix weights rhoSuffix[t] = ρ_{t:T-1}
  const rho = new Float64Array(T + 1);
  rho[T] = 1;
  for (let k = T - 1; k >= 0; k--) {
    const { s, a } = traj[k];
    const pb = BEHAVIOR_POLICY.pi[s][a];
    const pt = TARGET_POLICY.pi[s][a];
    rho[k] = pb === 0 ? 0 : rho[k + 1] * (pt / pb);
  }

  const seen = new Set<number>();
  for (let t = 0; t < T; t++) {
    const s = traj[t].s;
    if (seen.has(s)) continue;
    seen.add(s);
    const w = rho[t];
    st.numArr[s] += w * Gs[t];
    if (weighted) {
      st.denArr[s] += w;
    } else {
      st.denArr[s] += 1;
    }
    st.sumW[s] += w;
    st.sumW2[s] += w * w;
  }
  // recompute V from accumulators
  for (let s = 0; s < NS; s++) {
    st.V[s] = st.denArr[s] === 0 ? 0 : st.numArr[s] / st.denArr[s];
  }

  st.episode += 1;
  st.history.push(st.V[0]);
  if (st.history.length > MAX_HIST) st.history.shift();
  st.lastTraj = traj;
}

// ── Component ──────────────────────────────────────────────────────────────────

class MCEstimatorLab extends HTMLElement {
  private mode: Mode = "uniform";
  private firstVisit = true;
  private state = freshState();
  private running = false;
  private timerId: ReturnType<typeof setTimeout> | null = null;

  // Panel B renderer
  private rendererB: GridworldRenderer | null = null;
  // Panel C SVG
  private cellsC: SVGRectElement[] = [];
  private labelsC: SVGTextElement[] = [];
  // Panel D SVG
  private svgD: SVGSVGElement | null = null;
  // Panel E container
  private panelE: HTMLElement | null = null;
  // Panel F container
  private panelF: HTMLElement | null = null;

  private setStatus: (t: string) => void = () => {};
  private btnRun: HTMLButtonElement | null = null;

  connectedCallback() { this.build(); }
  disconnectedCallback() { this.stop(); }

  // ── Build ───────────────────────────────────────────────────────────────────

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({
      id: "mc-estimator-lab",
      arena: true,
      heavy: true,
      mobileNotice: "Best viewed on a wider screen — panels wrap on small displays.",
    });
    this.setStatus = setStatus;

    // ── Controls row ─────────────────────────────────────────────────────────
    const ctrlRow = document.createElement("div");
    ctrlRow.className = "rl-controls-row";
    ctrlRow.style.cssText = "gap:10px 20px;padding:8px 0 12px;border-bottom:1px solid var(--rl-border);";

    const modeSelect = document.createElement("select");
    modeSelect.className = "rl-select";
    const modes: [Mode, string][] = [
      ["uniform",      "On-policy · uniform random"],
      ["optimal",      "On-policy · optimal π*"],
      ["off-weighted", "Off-policy · weighted IS"],
      ["off-ordinary", "Off-policy · ordinary IS"],
    ];
    for (const [v, label] of modes) {
      const opt = document.createElement("option");
      opt.value = v; opt.textContent = label;
      modeSelect.appendChild(opt);
    }

    const fvLabel = document.createElement("label");
    fvLabel.className = "rl-label";
    fvLabel.innerHTML = `<input type="checkbox" id="mc-fv-check" checked> First-visit`;
    const fvCheck = fvLabel.querySelector<HTMLInputElement>("#mc-fv-check")!;

    const btnRun = document.createElement("button");
    btnRun.className = "rl-btn";
    btnRun.textContent = "▶ Run";
    this.btnRun = btnRun;

    const btnStep = document.createElement("button");
    btnStep.className = "rl-btn rl-btn--ghost";
    btnStep.textContent = "Step ×50";

    const btnReset = document.createElement("button");
    btnReset.className = "rl-btn rl-btn--ghost";
    btnReset.textContent = "↺ Reset";

    const epLabel = document.createElement("span");
    epLabel.className = "rl-mono";
    epLabel.style.cssText = "font-size:11px;color:var(--rl-ink-muted);";
    epLabel.textContent = "ep: 0";

    const modeLabel = document.createElement("span");
    modeLabel.className = "rl-label";
    modeLabel.style.cssText = "font-size:11px;color:var(--rl-ink-muted);";
    modeLabel.textContent = "Mode:";

    ctrlRow.append(modeLabel, modeSelect, fvLabel, btnRun, btnStep, btnReset, epLabel);
    body.appendChild(ctrlRow);

    // ── Main grid row (A, B, C) ───────────────────────────────────────────────
    const gridRow = document.createElement("div");
    gridRow.style.cssText = "display:flex;gap:16px;margin-top:14px;flex-wrap:wrap;align-items:flex-start;";
    body.appendChild(gridRow);

    // Panel A — mode info
    const panelA = document.createElement("div");
    panelA.className = "is-sub-panel";
    panelA.style.cssText = "flex:0 0 200px;min-width:180px;";
    panelA.innerHTML = `<div class="is-panel-title">A · Mode</div>`;
    const modeInfo = document.createElement("div");
    modeInfo.className = "is-panel-info";
    modeInfo.style.cssText = "font-size:10.5px;line-height:1.7;";
    panelA.appendChild(modeInfo);
    gridRow.appendChild(panelA);

    // Panel B — V̂ heatmap
    const panelB = document.createElement("div");
    panelB.className = "is-sub-panel";
    panelB.style.cssText = "flex:0 0 auto;min-width:0;";
    panelB.innerHTML = `<div class="is-panel-title">B · V̂(s) estimate</div>`;
    const bHost = document.createElement("div");
    panelB.appendChild(bHost);
    gridRow.appendChild(panelB);

    // Panel C — visit counts
    const panelC = document.createElement("div");
    panelC.className = "is-sub-panel";
    panelC.style.cssText = "flex:0 0 auto;min-width:0;";
    panelC.innerHTML = `<div class="is-panel-title">C · Visit counts N(s)</div>`;
    const cHost = document.createElement("div");
    panelC.appendChild(cHost);
    gridRow.appendChild(panelC);

    // ── Trace row (D) ─────────────────────────────────────────────────────────
    const traceRow = document.createElement("div");
    traceRow.className = "is-sub-panel";
    traceRow.style.cssText = "margin-top:14px;";
    traceRow.innerHTML = `<div class="is-panel-title">D · V̂(0,0) convergence  <span style="color:var(--rl-ink-muted);font-size:10px;font-weight:400;">true value shown as dashed line</span></div>`;
    const dHost = document.createElement("div");
    traceRow.appendChild(dHost);
    body.appendChild(traceRow);

    // ── Bottom row (E + F) ────────────────────────────────────────────────────
    const bottomRow = document.createElement("div");
    bottomRow.style.cssText = "display:flex;gap:16px;margin-top:14px;flex-wrap:wrap;align-items:flex-start;";
    body.appendChild(bottomRow);

    const panelE = document.createElement("div");
    panelE.className = "is-sub-panel";
    panelE.style.cssText = "flex:1 1 380px;min-width:280px;";
    panelE.innerHTML = `<div class="is-panel-title">E · 50-trial statistics (pre-computed)</div>`;
    this.panelE = panelE;
    bottomRow.appendChild(panelE);

    const panelF = document.createElement("div");
    panelF.className = "is-sub-panel";
    panelF.style.cssText = "flex:1 1 260px;min-width:220px;";
    panelF.innerHTML = `<div class="is-panel-title">F · Last trajectory</div>`;
    this.panelF = panelF;
    bottomRow.appendChild(panelF);

    this.appendChild(panel);

    // ── Initialize renderers ──────────────────────────────────────────────────
    const colorScale = makeValueColorScale([-1, 1]);

    this.rendererB = new GridworldRenderer(bHost, {
      mdp: _mdp,
      valueFn: [...this.state.V],
      policy: _uniform,
      showValues: true,
      colorScale,
      cellPx: CELL_B,
      startState: 0,
    });

    this.buildCountGrid(cHost);

    const dW = 560, dH = 140;
    const svgDEl = document.createElementNS(NS_SVG, "svg") as SVGSVGElement;
    svgDEl.setAttribute("viewBox", `0 0 ${dW} ${dH}`);
    svgDEl.setAttribute("width", "100%");
    svgDEl.style.height = "auto";
    svgDEl.classList.add("rl-svg");
    dHost.appendChild(svgDEl);
    this.svgD = svgDEl;

    this.updateModeInfo(modeInfo);
    this.loadStats();

    // ── Wire controls ─────────────────────────────────────────────────────────
    modeSelect.addEventListener("change", () => {
      this.mode = modeSelect.value as Mode;
      this.resetState();
      this.updateModeInfo(modeInfo);
      this.rendererB?.update({ policy: this.displayPolicy() });
      this.loadStats();
    });

    fvCheck.addEventListener("change", () => {
      this.firstVisit = fvCheck.checked;
    });

    btnRun.addEventListener("click", () => {
      if (this.running) { this.stop(); } else { this.start(); }
    });

    btnStep.addEventListener("click", () => {
      this.stop();
      for (let i = 0; i < 50; i++) this.runOne();
      this.repaint(epLabel);
    });

    btnReset.addEventListener("click", () => {
      this.stop();
      this.resetState();
      epLabel.textContent = "ep: 0";
      this.repaint(epLabel);
    });
  }

  // ── Build count grid (Panel C) ────────────────────────────────────────────

  private buildCountGrid(container: HTMLElement): SVGSVGElement {
    const C = CELL_C, G = 5, P = 10;
    const SIZE = GRID_SIZE * C + (GRID_SIZE - 1) * G + 2 * P;
    const svgEl = document.createElementNS(NS_SVG, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
    svgEl.setAttribute("width", `${SIZE}`);
    svgEl.style.maxWidth = "100%";
    svgEl.style.height = "auto";
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);

    this.cellsC = [];
    this.labelsC = [];
    const svg = d3.select(svgEl);

    for (let s = 0; s < NS; s++) {
      const { r, c } = rc(s);
      const x = P + c * (C + G);
      const y = P + r * (C + G);
      const cx = x + C / 2, cy = y + C / 2;

      const rect = svg.append("rect")
        .attr("x", x).attr("y", y)
        .attr("width", C).attr("height", C)
        .attr("rx", 4)
        .attr("fill", "var(--rl-surface)")
        .attr("stroke", "var(--rl-border)")
        .attr("stroke-width", 1)
        .node()!;
      this.cellsC.push(rect);

      if (_mdp.terminals[s]) {
        svg.append("text").attr("x", cx).attr("y", cy + 3)
          .attr("text-anchor", "middle").attr("dominant-baseline", "central")
          .attr("font-size", 16).text(s === idx(1, 1) ? "☠" : "⚑");
      }

      const label = svg.append("text")
        .attr("x", cx).attr("y", cy + 3)
        .attr("text-anchor", "middle").attr("dominant-baseline", "central")
        .attr("font-family", "var(--rl-font-mono)")
        .attr("font-size", 11)
        .attr("fill", "var(--rl-ink)")
        .text("")
        .node()!;
      this.labelsC.push(label);
    }

    return svgEl;
  }

  // ── Mode info ──────────────────────────────────────────────────────────────

  private displayPolicy(): Policy {
    return (this.mode === "uniform" || this.mode === "off-ordinary" || this.mode === "off-weighted")
      ? _uniform
      : _optimal;
  }

  private updateModeInfo(el: HTMLElement) {
    const isOff = this.mode.startsWith("off");
    const trueV = TRUE_V[this.mode];
    const evalPi = this.mode === "uniform" ? "uniform random π" : "optimal π*";
    const behPi = isOff ? "uniform random (behavior)" : "—";

    el.innerHTML = [
      isOff
        ? `<b style="color:var(--mc-off-policy)">Off-policy IS</b>`
        : `<b style="color:var(--mc-on-policy)">On-policy MC</b>`,
      `<br>Evaluate: <b>${evalPi}</b>`,
      isOff ? `<br>Behavior: <b>${behPi}</b>` : "",
      isOff
        ? (this.mode === "off-weighted"
          ? `<br>Estimator: <b>weighted IS</b> (self-normalised)`
          : `<br>Estimator: <b>ordinary IS</b>`)
        : `<br>Estimator: <b>${this.firstVisit ? "first-visit" : "every-visit"} MC</b>`,
      `<br><br>True V(0,0): <b style="font-family:var(--rl-mono)">${trueV.toFixed(4)}</b>`,
      isOff ? `<br><br><span style="color:var(--rl-ink-muted);font-size:9.5px;">Weighted IS is exact at large N; ordinary IS has high variance from rare trajectories.</span>` : "",
    ].join("");
  }

  // ── Statistics panel (E) ──────────────────────────────────────────────────

  private async loadStats() {
    const el = this.panelE;
    if (!el) return;
    const isOff = this.mode.startsWith("off");
    const url = isOff
      ? "/data/mc/off_policy_gridworld.json"
      : "/data/mc/on_policy_convergence.json";

    try {
      const resp = await fetch(url);
      const data = await resp.json();
      this.renderStats(el, data, isOff);
    } catch {
      el.innerHTML += `<div class="is-panel-info" style="color:var(--rl-ink-muted)">Statistics unavailable — run dev server.</div>`;
    }
  }

  private renderStats(container: HTMLElement, data: Record<string, unknown>, isOff: boolean) {
    // Remove old table if present
    const old = container.querySelector("table");
    if (old) old.remove();
    const old2 = container.querySelector(".mc-stat-info");
    if (old2) old2.remove();

    const rows = data.rows as Record<string, number>[];
    const trueVal = data.trueValue as number;

    const tbl = document.createElement("table");
    tbl.style.cssText = "width:100%;border-collapse:collapse;font-family:var(--rl-font-mono);font-size:11px;";

    const makeCell = (text: string, bold = false, color = "") => {
      const td = document.createElement("td");
      td.style.cssText = `padding:3px 8px;border:1px solid var(--rl-border);${color ? `color:${color};` : ""}${bold ? "font-weight:600;" : ""}`;
      td.textContent = text;
      return td;
    };

    const thead = tbl.createTHead();
    const hRow = thead.insertRow();
    if (isOff) {
      for (const h of ["N", "Avg non-zero", "Weighted mean", "±SD", "Ordinary mean", "±SD"]) {
        const th = document.createElement("th");
        th.textContent = h;
        th.style.cssText = "padding:3px 8px;border:1px solid var(--rl-border);text-align:left;color:var(--rl-ink-muted);font-weight:600;";
        hRow.appendChild(th);
      }
    } else {
      for (const h of ["N", "Mean V̂(0,0)", "±SD", "RMSE"]) {
        const th = document.createElement("th");
        th.textContent = h;
        th.style.cssText = "padding:3px 8px;border:1px solid var(--rl-border);text-align:left;color:var(--rl-ink-muted);font-weight:600;";
        hRow.appendChild(th);
      }
    }

    const tbody = tbl.createTBody();
    for (const row of rows) {
      const tr = tbody.insertRow();
      if (isOff) {
        const wt = (row.weightedMean as number).toFixed(4);
        const wtClr = Math.abs((row.weightedMean as number) - trueVal) < 0.001 ? "var(--mc-on-policy)" : "";
        tr.append(
          makeCell(String(row.N), true),
          makeCell((row.avgNonZero as number).toFixed(1)),
          makeCell(wt, false, wtClr),
          makeCell((row.weightedSD as number).toFixed(4)),
          makeCell((row.ordinaryMean as number).toFixed(4)),
          makeCell((row.ordinarySD as number).toFixed(4)),
        );
      } else {
        tr.append(
          makeCell(String(row.N), true),
          makeCell((row.mean as number).toFixed(4)),
          makeCell((row.std as number).toFixed(4)),
          makeCell((row.rmse as number).toFixed(4)),
        );
      }
    }

    container.appendChild(tbl);

    const info = document.createElement("div");
    info.className = "mc-stat-info is-panel-info";
    info.style.cssText = "margin-top:6px;font-size:10px;color:var(--rl-ink-muted);";
    info.textContent = `50 independent trials · true V(0,0) = ${trueVal}`;
    container.appendChild(info);
  }

  // ── Run loop ───────────────────────────────────────────────────────────────

  private start() {
    if (this.running) return;
    this.running = true;
    if (this.btnRun) this.btnRun.textContent = "■ Stop";
    this.tick();
  }

  private stop() {
    this.running = false;
    if (this.timerId !== null) { clearTimeout(this.timerId); this.timerId = null; }
    if (this.btnRun) this.btnRun.textContent = "▶ Run";
  }

  private tick() {
    const epEl = this.querySelector<HTMLElement>(".rl-mono");
    for (let i = 0; i < CHUNK; i++) this.runOne();
    this.repaint(epEl);
    if (this.running) {
      const delay = prefersReducedMotion() ? 200 : TICK_MS;
      this.timerId = setTimeout(() => this.tick(), delay);
    }
  }

  private runOne() {
    const st = this.state;
    const isOff = this.mode.startsWith("off");
    const policy = isOff ? BEHAVIOR_POLICY : (this.mode === "optimal" ? _optimal : _uniform);
    const traj = oneEpisode(policy, Math.random);
    if (isOff) {
      applyOffPolicy(st, traj, this.mode === "off-weighted");
    } else {
      applyOnPolicy(st, traj, this.firstVisit);
    }
  }

  private resetState() {
    this.state = freshState();
    this.repaint(null);
  }

  // ── Repaint all panels ─────────────────────────────────────────────────────

  private repaint(epEl: HTMLElement | null) {
    const st = this.state;

    // Panel B
    const colorScale = makeValueColorScale([-1, 1]);
    const dispPolicy = this.displayPolicy();
    this.rendererB?.update({ valueFn: [...st.V], policy: dispPolicy, colorScale }, { animate: true });

    // Panel C
    this.repaintCounts();

    // Panel D
    this.repaintTrace();

    // Panel F
    this.repaintTraj();

    // Status + ep label
    const trueV = TRUE_V[this.mode];
    const v00 = st.V[0];
    const err = (v00 - trueV).toFixed(4);
    const sign = v00 >= trueV ? "+" : "";
    this.setStatus(`ep ${st.episode}  V̂(0,0) = ${v00.toFixed(4)}  err ${sign}${err}`);

    if (epEl) epEl.textContent = `ep: ${st.episode}`;
  }

  private repaintCounts() {
    const st = this.state;
    if (!this.cellsC || !this.labelsC) return;
    const maxVisit = Math.max(...st.visits, 1);
    for (let s = 0; s < NS; s++) {
      const n = st.visits[s];
      const frac = n / maxVisit;
      const bg = `rgba(14,116,144,${0.08 + 0.72 * frac})`; // --mc-first-visit tint
      const textClr = frac > 0.55 ? "#fff" : "var(--rl-ink)";
      this.cellsC[s].setAttribute("fill", _mdp.terminals[s] ? "var(--rl-surface)" : bg);
      this.labelsC[s].setAttribute("fill", textClr);
      this.labelsC[s].textContent = _mdp.terminals[s] ? "" : (n === 0 ? "" : String(n));
    }
  }

  private repaintTrace() {
    if (!this.svgD) return;
    const hist = this.state.history;
    const trueV = TRUE_V[this.mode];
    const svg = d3.select(this.svgD);
    svg.selectAll("*").remove();

    const W = 560, H = 140;
    const M = { t: 14, r: 14, b: 28, l: 46 };
    const PW = W - M.l - M.r;
    const PH = H - M.t - M.b;

    if (hist.length < 2) {
      svg.append("text").attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle")
        .attr("font-family", "var(--rl-font-mono)")
        .attr("font-size", 11)
        .attr("fill", "var(--rl-ink-muted)")
        .text("Run episodes to see convergence trace");
      return;
    }

    const g = svg.append("g").attr("transform", `translate(${M.l},${M.t})`);
    const n = hist.length;
    const xScale = d3.scaleLinear().domain([0, n - 1]).range([0, PW]);
    const allVals = [...hist, trueV];
    const yMin = Math.min(...allVals) - 0.05;
    const yMax = Math.max(...allVals) + 0.05;
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([PH, 0]).nice();

    // Axes
    g.append("g").attr("transform", `translate(0,${PH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(4).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    // True value reference line
    g.append("line")
      .attr("x1", 0).attr("x2", PW)
      .attr("y1", yScale(trueV)).attr("y2", yScale(trueV))
      .attr("stroke", "var(--mc-truth)").attr("stroke-width", 1.2)
      .attr("stroke-dasharray", "5,3").attr("opacity", 0.5);
    g.append("text")
      .attr("x", PW - 2).attr("y", yScale(trueV) - 4)
      .attr("text-anchor", "end").attr("font-family", "var(--rl-font-mono)")
      .attr("font-size", 9).attr("fill", "var(--mc-truth)").attr("opacity", 0.6)
      .text(`true ${trueV}`);

    // Running estimate line
    const lineColor = this.mode.startsWith("off") ? "var(--mc-off-policy)" : "var(--mc-on-policy)";
    const linePath = d3.line<number>().x((_, i) => xScale(i)).y((d) => yScale(d)).curve(d3.curveMonotoneX);
    g.append("path").datum(hist)
      .attr("d", linePath)
      .attr("fill", "none").attr("stroke", lineColor).attr("stroke-width", 1.4);

    // x axis label
    g.append("text").attr("x", PW / 2).attr("y", PH + 22)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "var(--rl-ink-muted)")
      .text("episode");
  }

  private repaintTraj() {
    if (!this.panelF) return;
    const traj = this.state.lastTraj;
    const titleEl = this.panelF.querySelector(".is-panel-title");
    // Remove old content except title
    while (this.panelF.lastChild !== titleEl) this.panelF.removeChild(this.panelF.lastChild!);

    if (traj.length === 0) {
      const p = document.createElement("div");
      p.className = "is-panel-info";
      p.style.color = "var(--rl-ink-muted)";
      p.textContent = "No trajectory yet.";
      this.panelF.appendChild(p);
      return;
    }

    // Path line
    const actionSym = ["↑", "→", "↓", "←"];
    const pathParts: string[] = [];
    for (const { s, a } of traj) {
      const { r: row, c: col } = rc(s);
      pathParts.push(`(${row},${col})${actionSym[a]}`);
    }
    const last = traj[traj.length - 1];
    const { r: lr, c: lc } = rc(last.sp);
    pathParts.push(`(${lr},${lc})`);

    const pathDiv = document.createElement("div");
    pathDiv.className = "is-panel-info";
    pathDiv.style.cssText = "font-size:10.5px;word-break:break-all;line-height:1.8;";
    pathDiv.textContent = pathParts.join(" ");
    this.panelF.appendChild(pathDiv);

    // Returns
    const G0 = computeReturns(traj.map(s => s.r), _mdp.gamma)[0];
    const rDiv = document.createElement("div");
    rDiv.className = "is-panel-info";
    rDiv.style.cssText = "margin-top:6px;font-size:10px;color:var(--rl-ink-muted);";
    const outcome = last.sp === idx(2, 2) ? "⚑ goal" : (last.sp === idx(1, 1) ? "☠ pit" : "max-steps");
    rDiv.innerHTML = `len: <b>${traj.length}</b>  G₀ = <b>${G0.toFixed(3)}</b>  outcome: <b>${outcome}</b>`;
    this.panelF.appendChild(rDiv);
  }
}

customElements.define("mc-estimator-lab", MCEstimatorLab);
