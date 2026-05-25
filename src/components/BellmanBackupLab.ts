/**
 * V6 — Bellman Backup Lab. THE CENTERPIECE. Four synchronized panels over one
 * shared Store, animating how value propagates outward from terminal states as
 * the Bellman operator is iterated:
 *
 *   A (top-left)     one state's backup, expanded term-by-term
 *   B (top-right)    full grid colored by V_k + step/play/reset + mode toggle
 *   C (bottom-left)  V_k(s) traces + log-scale ‖V_k − V∞‖∞ (geometric at rate γ)
 *   D (bottom-right) the policy in use — fixed π, or the greedy policy emerging
 *
 * Flip "mode" to watch V_k converge to a *different* fixed point: V^π under the
 * expectation operator T^π, V^* under the optimality operator T^*. That switch
 * is the setup for value iteration in Lesson 3.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { MDPEditor } from "./MDPEditor";
import { StepLoop } from "./loop";
import { Store } from "./store";
import { buildGridworld } from "../mdp/gridworld";
import { policyEvaluationExact, bellmanExpectationBackup } from "../mdp/policy-evaluation";
import { bellmanOptimalityBackup, optimalValue, qOfAction, greedyPolicy } from "../mdp/value-iteration";
import { resolvePolicy, buildPolicySelect, savedPolicy, type PolicyKey } from "./mdp-shared";
import { fmtV } from "./value-scale";
import { cssVar } from "./base";
import { ACTION_NAMES, idx, rc, type MDP, type Policy } from "../mdp/types";

type Mode = "expectation" | "optimality";
const NS = "http://www.w3.org/2000/svg";
const TRACE = [idx(0, 0), idx(1, 0), idx(0, 1), idx(1, 2)];
const TRACE_VARS = ["--mc-state-1", "--mc-state-2", "--mc-state-3", "--mc-state-4"];

interface LabState {
  mdp: MDP;
  mode: Mode;
  policyKey: PolicyKey;
  policy: Policy;
  V: number[];
  k: number;
  targetV: number[];
  traces: { perState: Record<number, number[]>; err: number[] };
  selected: number;
}

export class BellmanBackupLab extends HTMLElement {
  private store!: Store<LabState>;
  private loop!: StepLoop;
  private gamma = 0.9;
  private slippery = false;

  // panel handles
  private panelA!: HTMLElement;
  private gridB!: GridworldRenderer;
  private gridD!: GridworldRenderer;
  private svgC!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private statusB!: (t: string) => void;
  private playBtn!: HTMLButtonElement;

  connectedCallback(): void {
    this.innerHTML = "";
    const mdp = buildGridworld({ slippery: this.slippery, gamma: this.gamma });
    const policy = resolvePolicy(mdp, "uniform");
    const targetV = policyEvaluationExact(mdp, policy);
    this.store = new Store<LabState>({
      mdp,
      mode: "expectation",
      policyKey: "uniform",
      policy,
      V: new Array(mdp.nS).fill(0),
      k: 0,
      targetV,
      traces: this.initTraces(new Array(mdp.nS).fill(0), targetV),
      selected: idx(1, 0),
    });

    const { panel, body, setStatus } = createPanel({ id: "bellman-backup-lab", arena: true });
    void setStatus;
    body.appendChild(this.buildControls());

    const grid = document.createElement("div");
    grid.className = "mdp-lab-grid";
    grid.append(this.buildPanelA(), this.buildPanelB(), this.buildPanelC(), this.buildPanelD());
    body.appendChild(grid);
    this.appendChild(panel);

    this.loop = new StepLoop({
      baseIntervalMs: 600,
      onStep: () => this.step(),
      onFrame: () => {},
    });

    this.store.subscribe((s) => this.render(s));
    savedPolicy.subscribe(() => {
      if (this.store.get().policyKey === "saved") this.reconfigure();
    });
    this.render(this.store.get());
  }

  disconnectedCallback(): void {
    this.loop?.destroy();
  }

  // ---- state transitions ----

  private initTraces(V: number[], target: number[]): LabState["traces"] {
    const perState: Record<number, number[]> = {};
    for (const s of TRACE) perState[s] = [V[s]];
    return { perState, err: [this.errNorm(V, target)] };
  }

  private errNorm(V: number[], target: number[]): number {
    let m = 0;
    for (let s = 0; s < V.length; s++) m = Math.max(m, Math.abs(V[s] - target[s]));
    return m;
  }

  private backup(s: LabState, V: number[]): number[] {
    return s.mode === "optimality"
      ? bellmanOptimalityBackup(s.mdp, V)
      : bellmanExpectationBackup(s.mdp, s.policy, V);
  }

  private step(): boolean {
    const s = this.store.get();
    const V = this.backup(s, s.V);
    const k = s.k + 1;
    const traces = {
      perState: { ...s.traces.perState },
      err: [...s.traces.err, this.errNorm(V, s.targetV)],
    };
    for (const st of TRACE) traces.perState[st] = [...s.traces.perState[st], V[st]];
    this.store.set({ V, k, traces });
    return k < 80 && this.errNorm(V, s.targetV) > 1e-7;
  }

  private reset(): void {
    this.loop.pause();
    this.syncPlayBtn();
    const s = this.store.get();
    const V0 = new Array(s.mdp.nS).fill(0);
    this.store.set({ V: V0, k: 0, traces: this.initTraces(V0, s.targetV) });
  }

  /** Rebuild MDP / policy / target from the current controls and reset. */
  private reconfigure(patch: Partial<Pick<LabState, "mode" | "policyKey">> = {}): void {
    this.loop.pause();
    this.syncPlayBtn();
    const cur = this.store.get();
    const mode = patch.mode ?? cur.mode;
    const policyKey = patch.policyKey ?? cur.policyKey;
    const mdp = buildGridworld({ slippery: this.slippery, gamma: this.gamma });
    const policy = resolvePolicy(mdp, policyKey);
    const targetV = mode === "optimality" ? optimalValue(mdp, 300) : policyEvaluationExact(mdp, policy);
    const V0 = new Array(mdp.nS).fill(0);
    this.store.set({
      mdp,
      mode,
      policyKey,
      policy,
      V: V0,
      k: 0,
      targetV,
      traces: this.initTraces(V0, targetV),
    });
  }

  // ---- controls ----

  private buildControls(): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "mdp-controls";

    // mode toggle
    const modeSeg = document.createElement("div");
    modeSeg.className = "mdp-seg";
    const mkMode = (text: string, mode: Mode) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.setAttribute("aria-label", `${text} operator`);
      if (this.store.get().mode === mode) b.classList.add("is-active");
      b.addEventListener("click", () => {
        modeSeg.querySelectorAll("button").forEach((x) => x.classList.remove("is-active"));
        b.classList.add("is-active");
        this.reconfigure({ mode });
      });
      return b;
    };
    modeSeg.append(mkMode("expectation T^π", "expectation"), mkMode("optimality T*", "optimality"));
    const modeLabel = document.createElement("label");
    modeLabel.append("operator ");
    modeLabel.appendChild(modeSeg);
    wrap.appendChild(modeLabel);

    // policy dropdown (drives expectation mode + Panel D)
    const polLabel = document.createElement("label");
    polLabel.append("π ");
    polLabel.appendChild(buildPolicySelect((k) => this.reconfigure({ policyKey: k })));
    wrap.appendChild(polLabel);

    // γ + stochasticity via MDPEditor
    new MDPEditor(wrap, {
      opts: { slippery: this.slippery, gamma: this.gamma },
      gammaRange: [0.5, 0.99],
      onChange: (opts) => {
        this.gamma = opts.gamma ?? 0.9;
        this.slippery = !!opts.slippery;
        this.reconfigure();
      },
    });

    // play / step / reset
    this.playBtn = document.createElement("button");
    this.playBtn.type = "button";
    this.playBtn.textContent = "Play";
    this.playBtn.addEventListener("click", () => this.togglePlay());
    const stepBtn = document.createElement("button");
    stepBtn.type = "button";
    stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => {
      this.loop.pause();
      this.syncPlayBtn();
      this.step();
    });
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => this.reset());
    wrap.append(this.playBtn, stepBtn, resetBtn);

    // speed
    const spLabel = document.createElement("label");
    spLabel.append("speed ");
    const sp = document.createElement("select");
    sp.setAttribute("aria-label", "playback speed");
    for (const m of [0.5, 1, 2, 4, 8, 16]) {
      const o = document.createElement("option");
      o.value = String(m);
      o.textContent = `${m}×`;
      if (m === 1) o.selected = true;
      sp.appendChild(o);
    }
    sp.addEventListener("change", () => this.loop.setSpeed(Number(sp.value)));
    spLabel.appendChild(sp);
    wrap.appendChild(spLabel);

    return wrap;
  }

  private togglePlay(): void {
    if (this.loop.isPlaying()) this.loop.pause();
    else {
      if (this.errNorm(this.store.get().V, this.store.get().targetV) <= 1e-7) this.reset();
      this.loop.play();
    }
    this.syncPlayBtn();
  }

  private syncPlayBtn(): void {
    this.playBtn.textContent = this.loop.isPlaying() ? "Pause" : "Play";
  }

  // ---- panel builders ----

  private labPanel(title: string, sub: string): { panel: HTMLElement; body: HTMLElement } {
    const panel = document.createElement("div");
    panel.className = "mdp-lab-panel";
    const h = document.createElement("p");
    h.className = "mdp-lab-panel__title";
    h.innerHTML = `${title} <small>${sub}</small>`;
    const body = document.createElement("div");
    panel.append(h, body);
    return { panel, body };
  }

  private buildPanelA(): HTMLElement {
    const { panel, body } = this.labPanel("A · Backup at one state", "click a cell in B");
    this.panelA = body;
    return panel;
  }

  private buildPanelB(): HTMLElement {
    const { panel, body } = this.labPanel("B · Value grid V_k", "");
    this.statusB = (t) => {
      (panel.querySelector(".mdp-lab-panel__title small") as HTMLElement).textContent = t;
    };
    this.gridB = new GridworldRenderer(body, {
      mdp: this.store.get().mdp,
      valueFn: this.store.get().V,
      cellPx: 74,
      highlightState: this.store.get().selected,
      onCellClick: (s) => {
        if (!this.store.get().mdp.terminals[s]) this.store.set({ selected: s });
      },
    });
    return panel;
  }

  private buildPanelC(): HTMLElement {
    const { panel, body } = this.labPanel("C · Convergence", "V_k(s) and ‖V_k−V∞‖∞ (log)");
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", "0 0 360 300");
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.svgC = d3.select(svgEl as SVGSVGElement);
    return panel;
  }

  private buildPanelD(): HTMLElement {
    const { panel, body } = this.labPanel("D · Policy", "");
    this.gridD = new GridworldRenderer(body, {
      mdp: this.store.get().mdp,
      policy: this.store.get().policy,
      cellPx: 74,
    });
    return panel;
  }

  // ---- rendering ----

  private render(s: LabState): void {
    this.statusB?.(`k = ${s.k}`);
    this.gridB.update({ mdp: s.mdp, valueFn: s.V, highlightState: s.selected }, { animate: true });

    // Panel D: fixed π in expectation; greedy(V_k) emerging in optimality.
    const shownPolicy = s.mode === "optimality" ? greedyPolicy(s.mdp, s.V) : s.policy;
    this.gridD.update({ mdp: s.mdp, policy: shownPolicy, showArrows: true });

    this.renderPanelA(s);
    this.renderPanelC(s);
  }

  private renderPanelA(s: LabState): void {
    const sel = s.selected;
    const { r, c } = rc(sel);
    if (s.mdp.terminals[sel]) {
      this.panelA.innerHTML = `<p class="mdp-readout">(${r},${c}) is terminal — V = 0 always.</p>`;
      return;
    }
    const q = Array.from({ length: s.mdp.nA }, (_, a) => qOfAction(s.mdp, s.V, sel, a));
    const isOpt = s.mode === "optimality";
    const best = q.indexOf(Math.max(...q));
    let vNew = 0;
    let html = `<p class="mdp-readout" style="margin:0 0 6px">backing up <span class="mdp-stat">(${r},${c})</span> · ${isOpt ? "max over a" : "Σ π(a|s)·q"}</p>`;
    html += `<div class="mdp-backup-row">`;
    for (let a = 0; a < s.mdp.nA; a++) {
      const pa = s.policy.pi[sel][a];
      const contrib = isOpt ? q[a] : pa * q[a];
      if (!isOpt) vNew += contrib;
      const mark = isOpt && a === best ? ' style="color:var(--mdp-backup-source);font-weight:600"' : "";
      const lead = isOpt
        ? `q(${ACTION_NAMES[a][0]}) = ${q[a].toFixed(3)}`
        : `π=${pa.toFixed(2)} · q(${ACTION_NAMES[a][0]})=${q[a].toFixed(3)} = ${contrib.toFixed(3)}`;
      html += `<span${mark}>${ACTION_NAMES[a]}</span><span${mark}>${lead}</span>`;
    }
    html += `</div>`;
    if (isOpt) vNew = q[best];
    html += `<p class="mdp-readout" style="margin:6px 0 0">⇒ <span class="mdp-stat">V_{k+1}(${r},${c}) = ${fmtV(vNew)}</span> &nbsp;<span style="color:var(--rl-ink-faint)">(currently V_k = ${fmtV(s.V[sel])})</span></p>`;
    this.panelA.innerHTML = html;

    // highlight successors of the selected state in grid B
    const inputs = new Set<number>();
    for (let a = 0; a < s.mdp.nA; a++)
      s.mdp.P[sel][a].forEach((p, sp) => p > 0 && inputs.add(sp));
    inputs.delete(sel);
    this.gridB.update({ highlightState: sel, highlightInputs: [...inputs] });
  }

  private renderPanelC(s: LabState): void {
    const svg = this.svgC;
    svg.selectAll("*").remove();
    const W = 360;
    const topH = 150;
    const botH = 150;
    const m = { l: 38, r: 10, t: 14, b: 22 };
    const kMax = Math.max(10, s.traces.err.length - 1);

    // --- top: V_k(s) traces ---
    const allV = TRACE.flatMap((st) => s.traces.perState[st]).concat(s.targetV.filter((_, i) => TRACE.includes(i)));
    const yMin = Math.min(0, ...allV) - 0.05;
    const yMax = Math.max(0, ...allV) + 0.05;
    const x = d3.scaleLinear().domain([0, kMax]).range([m.l, W - m.r]);
    const y1 = d3.scaleLinear().domain([yMin, yMax]).range([topH - m.b, m.t]);
    // zero line
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y1(0)).attr("y2", y1(0)).attr("stroke", "var(--rl-border)");
    svg.append("text").attr("x", 4).attr("y", m.t + 4).attr("class", "annot").attr("fill", "var(--rl-ink-faint)").style("font-size", "9px").text("V_k(s)");
    TRACE.forEach((st, i) => {
      const series = s.traces.perState[st];
      const line = d3.line<number>().x((_, k) => x(k)).y((v) => y1(v));
      svg.append("path").datum(series).attr("d", line).attr("fill", "none").attr("stroke", cssVar(TRACE_VARS[i])).attr("stroke-width", 1.6);
      // target asymptote
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y1(s.targetV[st])).attr("y2", y1(s.targetV[st]))
        .attr("stroke", cssVar(TRACE_VARS[i])).attr("stroke-width", 0.8).attr("stroke-dasharray", "2 3").attr("opacity", 0.6);
      const { r, c } = rc(st);
      svg.append("text").attr("x", W - m.r).attr("y", y1(series[series.length - 1]) - 2).attr("text-anchor", "end")
        .attr("class", "annot").attr("fill", cssVar(TRACE_VARS[i])).style("font-size", "8px").text(`(${r},${c})`);
    });

    // --- bottom: log-scale error ---
    const off = topH;
    const errs = s.traces.err.map((e) => Math.max(e, 1e-9));
    const eMax = Math.max(...errs, 1e-3);
    const yLog = d3.scaleLog().domain([1e-7, eMax]).range([off + botH - m.b, off + m.t]).clamp(true);
    svg.append("text").attr("x", 4).attr("y", off + m.t + 4).attr("class", "annot").attr("fill", "var(--rl-ink-faint)").style("font-size", "9px").text("‖V_k−V∞‖∞");
    // axis ticks (decades)
    for (const d of [1e-6, 1e-4, 1e-2, 1]) {
      if (d > eMax) continue;
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", yLog(d)).attr("y2", yLog(d)).attr("stroke", "var(--rl-border)").attr("opacity", 0.5);
      svg.append("text").attr("x", m.l - 3).attr("y", yLog(d) + 3).attr("text-anchor", "end").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").style("font-size", "8px").text(`1e${Math.round(Math.log10(d))}`);
    }
    const errLine = d3.line<number>().x((_, k) => x(k)).y((e) => yLog(e));
    svg.append("path").datum(errs).attr("d", errLine).attr("fill", "none").attr("stroke", "var(--mdp-backup-source)").attr("stroke-width", 1.8);
    // expected geometric slope guide (rate γ) for expectation mode
    if (s.mode === "expectation" && errs.length > 1) {
      const e0 = errs[0];
      const guide = d3.range(0, kMax + 1).map((k) => e0 * Math.pow(this.gamma, k));
      svg.append("path").datum(guide).attr("d", errLine).attr("fill", "none").attr("stroke", "var(--rl-ink-faint)").attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
    }
    svg.append("text").attr("x", W - m.r).attr("y", off + botH - 4).attr("text-anchor", "end").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").style("font-size", "8px").text(`k → ${kMax}`);
  }
}

customElements.define("bellman-backup-lab", BellmanBackupLab);
