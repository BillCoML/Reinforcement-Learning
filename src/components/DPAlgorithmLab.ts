/**
 * V4 — DP Algorithm Lab. CENTERPIECE.
 * Three modes (PI, VI, MPI) in one interface, with optional side-by-side comparison.
 * VI shows the value-ripple animation. PI shows phase (PE/PI step) in real time.
 */
import * as d3 from 'd3';
import { createPanel } from './PanelChrome';
import { GridworldRenderer } from './GridworldRenderer';
import { buildGridworld, uniformPolicy } from '../mdp/gridworld';
import { bellmanExpectationBackup } from '../mdp/policy-evaluation';
import { bellmanOptimalityBackup } from '../mdp/value-iteration';
import { policyEvaluationExact } from '../mdp/policy-evaluation';
import { policyImprovement, supDist, policiesEqual } from '../dp/algorithms';
import { idx, type MDP, type Policy } from '../mdp/types';
import { prefersReducedMotion } from './base';

type Mode = 'PI' | 'VI' | 'MPI';
type Phase = 'idle' | 'pe' | 'pi-step' | 'backup';

interface AlgoState {
  mode: Mode;
  V: number[];
  policy: Policy;
  outerIter: number;
  peSubIter: number;
  phase: Phase;
  converged: boolean;
  totalBackups: number;
  history: Array<{ v00: number; outerIter: number }>;
  m: number;
  gamma: number;
  slippery: boolean;
}

function makeInitState(mode: Mode, mdp: MDP, m = -1): AlgoState {
  return {
    mode,
    V: new Array(mdp.nS).fill(0),
    policy: uniformPolicy(mdp),
    outerIter: 0,
    peSubIter: 0,
    phase: 'idle',
    converged: false,
    totalBackups: 0,
    history: [{ v00: 0, outerIter: 0 }],
    m,
    gamma: mdp.gamma,
    slippery: false,
  };
}

const STOP_EPS = 1e-8;

function stepVI(mdp: MDP, st: AlgoState): AlgoState {
  if (st.converged) return st;
  const Vn = bellmanOptimalityBackup(mdp, st.V);
  const delta = supDist(Vn, st.V);
  const thresh = STOP_EPS * (1 - mdp.gamma) / mdp.gamma;
  const outerIter = st.outerIter + 1;
  const converged = delta < thresh;
  const policy = converged ? policyImprovement(mdp, Vn) : policyImprovement(mdp, Vn);
  return {
    ...st, V: Vn, policy, outerIter,
    totalBackups: st.totalBackups + mdp.nS,
    phase: 'backup', converged,
    history: [...st.history, { v00: Vn[idx(0, 0)], outerIter }],
  };
}

function stepPI(mdp: MDP, st: AlgoState): AlgoState {
  if (st.converged) return st;
  const V = policyEvaluationExact(mdp, st.policy);
  const newPolicy = policyImprovement(mdp, V);
  const converged = policiesEqual(st.policy, newPolicy);
  const outerIter = st.outerIter + 1;
  return {
    ...st, V, policy: newPolicy, outerIter,
    totalBackups: st.totalBackups + mdp.nS * 20, // approx cost
    phase: 'pi-step', converged,
    history: [...st.history, { v00: V[idx(0, 0)], outerIter }],
  };
}

function stepMPI(mdp: MDP, st: AlgoState): AlgoState {
  if (st.converged) return st;
  const m = st.m <= 0 || !isFinite(st.m) ? 999 : st.m;
  let V = st.V.slice();
  for (let i = 0; i < m; i++) V = bellmanExpectationBackup(mdp, st.policy, V);
  const newPolicy = policyImprovement(mdp, V);
  const converged = policiesEqual(st.policy, newPolicy);
  const outerIter = st.outerIter + 1;
  return {
    ...st, V, policy: newPolicy, outerIter,
    totalBackups: st.totalBackups + mdp.nS * Math.min(m, 30),
    phase: 'pe', converged,
    history: [...st.history, { v00: V[idx(0, 0)], outerIter }],
  };
}

function doStep(mdp: MDP, st: AlgoState): AlgoState {
  if (st.mode === 'VI') return stepVI(mdp, st);
  if (st.mode === 'PI') return stepPI(mdp, st);
  return stepMPI(mdp, st);
}

const PLOT_W = 280, PLOT_H = 140;

class AlgoColumn {
  state!: AlgoState;
  mdp!: MDP;
  renderer!: GridworldRenderer;
  plotSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  phaseEl!: HTMLElement;
  iterEl!: HTMLElement;
  backupEl!: HTMLElement;
  convEl!: HTMLElement;
  equationEl!: HTMLElement;
  root: HTMLElement;

  constructor(container: HTMLElement, mode: Mode, gamma: number, slippery: boolean, m = -1) {
    this.root = document.createElement('div');
    this.root.style.cssText = 'display:flex;flex-direction:column;gap:10px;flex:1;min-width:280px';
    container.appendChild(this.root);
    this.mdp = buildGridworld({ slippery, gamma });
    this.state = makeInitState(mode, this.mdp, m);
    this.buildUI();
  }

  private phaseColor(phase: Phase): string {
    if (phase === 'backup') return 'var(--dp-phase-optimality)';
    if (phase === 'pe') return 'var(--dp-phase-evaluation)';
    if (phase === 'pi-step') return 'var(--dp-phase-improvement)';
    return 'var(--rl-ink-muted)';
  }

  private phaseLabel(phase: Phase, mode: Mode): string {
    if (mode === 'VI') return 'T* backup';
    if (phase === 'pe') return 'Policy Evaluation';
    if (phase === 'pi-step') return 'Policy Improvement';
    return 'idle';
  }

  buildUI() {
    this.root.innerHTML = '';

    // Mode badge
    const modeColors: Record<Mode, string> = {
      PI: 'var(--dp-phase-improvement)', VI: 'var(--dp-phase-optimality)', MPI: 'var(--dp-phase-evaluation)',
    };
    const modeHeader = document.createElement('div');
    modeHeader.style.cssText = 'display:flex;align-items:center;gap:8px';
    const modeBadge = document.createElement('span');
    modeBadge.style.cssText = `background:${modeColors[this.state.mode]};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;font-family:var(--rl-font-ui);font-weight:600`;
    modeBadge.textContent = this.state.mode === 'MPI' ? `MPI (m=${this.state.m <= 0 ? '∞' : this.state.m})` : this.state.mode;
    this.phaseEl = document.createElement('span');
    this.phaseEl.style.cssText = 'font-size:11px;font-family:var(--rl-font-mono);color:var(--rl-ink-muted)';
    modeHeader.append(modeBadge, this.phaseEl);
    this.root.appendChild(modeHeader);

    // Equation display
    this.equationEl = document.createElement('div');
    this.equationEl.style.cssText = 'font-size:11px;font-family:var(--rl-font-mono);color:var(--rl-ink-faint);background:var(--rl-surface-2);padding:6px 8px;border-radius:4px;min-height:28px';
    this.root.appendChild(this.equationEl);

    // Gridworld
    this.renderer = new GridworldRenderer(this.root, {
      mdp: this.mdp,
      valueFn: this.state.V,
      policy: this.state.policy,
      showValues: true,
      showArrows: true,
      cellPx: 78,
    });

    // Stats
    const stats = document.createElement('div');
    stats.style.cssText = 'display:flex;gap:12px;font-size:11px;font-family:var(--rl-font-mono);flex-wrap:wrap';
    this.iterEl = document.createElement('span');
    this.iterEl.style.color = 'var(--dp-iter-text)';
    this.backupEl = document.createElement('span');
    this.backupEl.style.color = 'var(--rl-ink-faint)';
    this.convEl = document.createElement('span');
    this.convEl.style.cssText = 'background:var(--dp-improving);color:#fff;padding:1px 5px;border-radius:3px;display:none';
    this.convEl.textContent = 'converged';
    stats.append(this.iterEl, this.backupEl, this.convEl);
    this.root.appendChild(stats);

    // Plot
    const plotTitle = document.createElement('div');
    plotTitle.style.cssText = 'font-size:10px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui)';
    plotTitle.textContent = 'V_k(0,0) — outer iterations';
    this.root.appendChild(plotTitle);

    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', String(PLOT_W)); svgEl.setAttribute('height', String(PLOT_H));
    svgEl.classList.add('rl-svg');
    this.root.appendChild(svgEl);
    this.plotSvg = d3.select(svgEl as SVGSVGElement);

    this.refreshUI();
  }

  step() {
    if (this.state.converged) return;
    this.state = doStep(this.mdp, this.state);
    const animate = !prefersReducedMotion();
    this.renderer.update({ valueFn: this.state.V, policy: this.state.policy }, { animate });
    this.refreshUI();
  }

  reset(gamma?: number, slippery?: boolean, m?: number) {
    if (gamma !== undefined) this.mdp = buildGridworld({ slippery: slippery ?? false, gamma });
    if (m !== undefined) this.state = makeInitState(this.state.mode, this.mdp, m);
    else this.state = makeInitState(this.state.mode, this.mdp, this.state.m);
    this.renderer.update({ mdp: this.mdp, valueFn: this.state.V, policy: this.state.policy });
    this.refreshUI();
  }

  refreshUI() {
    const { outerIter, totalBackups, converged, phase, mode } = this.state;
    this.phaseEl.textContent = converged ? '✓ done' : this.phaseLabel(phase, mode);
    this.phaseEl.style.color = this.phaseColor(phase);
    this.iterEl.textContent = `iter: ${outerIter}`;
    this.backupEl.textContent = `backups: ${totalBackups}`;
    this.convEl.style.display = converged ? '' : 'none';

    // Equation
    const k = outerIter;
    if (mode === 'VI') {
      this.equationEl.textContent = `V_${k+1}(s) = max_a [ r(s,a) + ${this.mdp.gamma} · Σ P V_${k}(s') ]`;
    } else if (mode === 'PI') {
      this.equationEl.textContent = phase === 'pe'
        ? `V_{π_${k}} = (I − γP^π)⁻¹ R^π  →  improve`
        : `π_${k+1}(s) = argmax_a Q^{π_${k}}(s,a)`;
    } else {
      this.equationEl.textContent = `MPI m=${this.state.m}: apply T^π ${this.state.m} times, then improve`;
    }

    this.updatePlot();
  }

  private updatePlot() {
    const svg = this.plotSvg;
    svg.selectAll('.lab-plot').remove();
    const hist = this.state.history;
    if (hist.length < 2) return;

    const margin = { top: 8, right: 8, bottom: 24, left: 38 };
    const W = PLOT_W - margin.left - margin.right;
    const H = PLOT_H - margin.top - margin.bottom;
    const g = svg.append('g').attr('class', 'lab-plot')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xs = d3.scaleLinear().domain([0, Math.max(1, hist[hist.length - 1].outerIter)]).range([0, W]);
    const yMin = Math.min(...hist.map(h => h.v00)) - 0.05;
    const yMax = Math.max(...hist.map(h => h.v00)) + 0.05;
    const ys = d3.scaleLinear().domain([yMin, yMax]).range([H, 0]);

    const modeColor = this.state.mode === 'VI' ? 'var(--dp-phase-optimality)'
      : this.state.mode === 'PI' ? 'var(--dp-phase-improvement)'
      : 'var(--dp-phase-evaluation)';

    g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(4)).attr('class', 'annot');
    g.append('g').call(d3.axisLeft(ys).ticks(3)).attr('class', 'annot');

    const line = d3.line<{ v00: number; outerIter: number }>().x(d => xs(d.outerIter)).y(d => ys(d.v00));
    g.append('path').datum(hist).attr('fill', 'none').attr('stroke', modeColor)
      .attr('stroke-width', 2).attr('d', line);

    hist.forEach(d => {
      g.append('circle').attr('cx', xs(d.outerIter)).attr('cy', ys(d.v00)).attr('r', 3)
        .attr('fill', modeColor).attr('stroke', 'var(--rl-surface)').attr('stroke-width', 1.5);
    });

    g.append('text').attr('x', W / 2).attr('y', H + 20).attr('text-anchor', 'middle')
      .attr('class', 'annot').attr('fill', 'var(--rl-ink-faint)').style('font-size', '9px')
      .text('outer iteration k');
  }
}

export class DPAlgorithmLab extends HTMLElement {
  private gamma = 0.9;
  private slippery = false;
  private sideBy = false;
  private mpiM = 5;
  private cols: AlgoColumn[] = [];
  private colContainer!: HTMLElement;
  private playing = false;
  private playTimer = 0;
  private setStatus!: (t: string) => void;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = '';
    const { panel, body, setStatus } = createPanel({ id: 'v4-dp-algorithm-lab', arena: true, heavy: true });
    this.setStatus = setStatus;
    this.appendChild(panel);

    // ── Top controls ──
    const topBar = document.createElement('div');
    topBar.style.cssText = 'display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px';
    body.appendChild(topBar);

    const modeColors: Record<Mode, string> = {
      PI: 'var(--dp-phase-improvement)', VI: 'var(--dp-phase-optimality)', MPI: 'var(--dp-phase-evaluation)',
    };

    // Mode selector tabs
    (['VI', 'PI', 'MPI'] as Mode[]).forEach(mode => {
      const b = document.createElement('button');
      b.className = 'rl-btn';
      b.style.cssText = `border-left:3px solid ${modeColors[mode]}`;
      b.textContent = mode === 'MPI' ? 'Modified PI' : mode;
      b.addEventListener('click', () => this.setMode(mode));
      topBar.appendChild(b);
    });

    // Side-by-side toggle
    const sbBtn = document.createElement('button');
    sbBtn.className = 'rl-btn';
    sbBtn.textContent = 'Side-by-side PI vs VI';
    sbBtn.addEventListener('click', () => {
      this.sideBy = !this.sideBy;
      sbBtn.style.background = this.sideBy ? 'var(--rl-ucb-tint)' : '';
      this.rebuildCols();
    });
    topBar.appendChild(sbBtn);

    // Sliders
    const mkSlider = (label: string, min: number, max: number, step: number, val: number, cb: (v: number) => void) => {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;font-family:var(--rl-font-ui);color:var(--rl-ink-faint)';
      const lbl = document.createElement('span'); lbl.textContent = label;
      const sl = document.createElement('input');
      sl.type = 'range'; sl.min = String(min); sl.max = String(max); sl.step = String(step); sl.value = String(val);
      sl.className = 'rl-slider';
      const out = document.createElement('span'); out.className = 'rl-mono'; out.textContent = String(val);
      sl.addEventListener('input', () => { out.textContent = sl.value; cb(parseFloat(sl.value)); });
      wrap.append(lbl, sl, out);
      topBar.appendChild(wrap);
    };
    mkSlider('γ', 0.5, 0.99, 0.01, this.gamma, v => {
      this.gamma = v;
      this.cols.forEach(c => c.reset(v, this.slippery));
    });

    // Stochastic toggle
    const slipBtn = document.createElement('button');
    slipBtn.className = 'rl-btn';
    slipBtn.textContent = 'Deterministic';
    slipBtn.addEventListener('click', () => {
      this.slippery = !this.slippery;
      slipBtn.textContent = this.slippery ? 'Slippery 80-10-10' : 'Deterministic';
      this.cols.forEach(c => c.reset(this.gamma, this.slippery));
    });
    topBar.appendChild(slipBtn);

    // MPI m control
    const mWrap = document.createElement('div');
    mWrap.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;font-family:var(--rl-font-ui);color:var(--rl-ink-faint)';
    mWrap.innerHTML = '<span>MPI m:</span>';
    const mSel = document.createElement('select');
    mSel.className = 'rl-select';
    [['∞ (full PI)', -1], ['10', 10], ['5', 5], ['3', 3], ['1 (≈VI)', 1]].forEach(([label, val]) => {
      const o = document.createElement('option'); o.value = String(val); o.textContent = String(label);
      mSel.appendChild(o);
    });
    mSel.value = String(this.mpiM);
    mSel.addEventListener('change', () => {
      this.mpiM = Number(mSel.value);
      this.cols.filter(c => c.state.mode === 'MPI').forEach(c => c.reset(this.gamma, this.slippery, this.mpiM));
    });
    mWrap.appendChild(mSel);
    topBar.appendChild(mWrap);

    // ── Column container ──
    this.colContainer = document.createElement('div');
    this.colContainer.style.cssText = 'display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start';
    body.appendChild(this.colContainer);

    // ── Bottom controls ──
    const botBar = document.createElement('div');
    botBar.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px';
    body.appendChild(botBar);

    const mkBtn = (label: string, cb: () => void) => {
      const b = document.createElement('button');
      b.className = 'rl-btn'; b.textContent = label;
      b.addEventListener('click', cb); botBar.appendChild(b);
    };
    mkBtn('Step', () => { this.stopPlay(); this.doStep(); });
    mkBtn('Play', () => this.startPlay());
    mkBtn('Pause', () => this.stopPlay());
    mkBtn('Reset All', () => {
      this.stopPlay();
      this.cols.forEach(c => c.reset(this.gamma, this.slippery));
    });

    this.setMode('VI');
  }

  private setMode(mode: Mode) {
    this.sideBy = false;
    this.stopPlay();
    this.colContainer.innerHTML = '';
    this.cols = [];
    const col = new AlgoColumn(this.colContainer, mode, this.gamma, this.slippery,
      mode === 'MPI' ? this.mpiM : -1);
    this.cols.push(col);
  }

  private rebuildCols() {
    this.stopPlay();
    this.colContainer.innerHTML = '';
    this.cols = [];
    if (this.sideBy) {
      this.cols.push(new AlgoColumn(this.colContainer, 'PI', this.gamma, this.slippery));
      this.cols.push(new AlgoColumn(this.colContainer, 'VI', this.gamma, this.slippery));
    } else {
      const mode = this.cols[0]?.state.mode ?? 'VI';
      this.cols.push(new AlgoColumn(this.colContainer, mode, this.gamma, this.slippery));
    }
  }

  private doStep() {
    this.cols.forEach(c => c.step());
    const allDone = this.cols.every(c => c.state.converged);
    this.setStatus(this.cols.map(c =>
      `${c.state.mode} iter=${c.state.outerIter}${c.state.converged ? '✓' : ''}`
    ).join(' | '));
    if (allDone) this.stopPlay();
  }

  private startPlay() {
    if (this.playing) return;
    this.playing = true;
    const tick = () => {
      if (!this.playing || this.cols.every(c => c.state.converged)) {
        this.playing = false; return;
      }
      this.doStep();
      this.playTimer = window.setTimeout(tick, 200);
    };
    tick();
  }

  private stopPlay() {
    this.playing = false;
    clearTimeout(this.playTimer);
  }
}

customElements.define('dp-algorithm-lab', DPAlgorithmLab);
