/**
 * V1 — Iterative PE Watcher.
 * The 3×3 gridworld colored by V_k as k increases, with a convergence line plot.
 * Policy selector: uniform, all-down, all-right.
 */
import * as d3 from 'd3';
import { createPanel } from './PanelChrome';
import { GridworldRenderer } from './GridworldRenderer';
import { buildGridworld, uniformPolicy, deterministicPolicy } from '../mdp/gridworld';
import { bellmanExpectationBackup } from '../mdp/policy-evaluation';
import { supDist } from '../dp/utils';
import { DOWN, RIGHT, idx, type MDP, type Policy } from '../mdp/types';
import { prefersReducedMotion } from './base';

const PLOT_W = 260, PLOT_H = 200;

export class IterativePEWatcher extends HTMLElement {
  private gamma = 0.9;
  private epsilon = 0.01;
  private policyName: 'uniform' | 'all-down' | 'all-right' = 'uniform';
  private mdp!: MDP;
  private policy!: Policy;
  private V: number[] = [];
  private k = 0;
  private history: Array<{ k: number; V: number[]; delta: number }> = [];
  private converged = false;
  private playing = false;
  private playTimer = 0;
  private renderer!: GridworldRenderer;
  private plotSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private kEl!: HTMLElement;
  private deltaEl!: HTMLElement;
  private threshEl!: HTMLElement;
  private badgeEl!: HTMLElement;
  private setStatus!: (t: string) => void;

  connectedCallback() { this.build(); }

  private getPolicy(mdp: MDP): Policy {
    if (this.policyName === 'all-down') return deterministicPolicy(mdp, Array(9).fill(DOWN));
    if (this.policyName === 'all-right') return deterministicPolicy(mdp, Array(9).fill(RIGHT));
    return uniformPolicy(mdp);
  }

  private reset() {
    this.mdp = buildGridworld({ slippery: false, gamma: this.gamma });
    this.policy = this.getPolicy(this.mdp);
    this.V = new Array(9).fill(0);
    this.k = 0;
    this.history = [{ k: 0, V: this.V.slice(), delta: 0 }];
    this.converged = false;
    this.stopPlay();
    this.renderer?.update({ mdp: this.mdp, valueFn: this.V, policy: this.policy });
    this.refreshUI();
  }

  private step() {
    if (this.converged) return;
    const Vn = bellmanExpectationBackup(this.mdp, this.policy, this.V);
    const delta = supDist(Vn, this.V);
    this.k++;
    this.V = Vn;
    this.history.push({ k: this.k, V: this.V.slice(), delta });
    const thresh = this.epsilon * (1 - this.gamma) / this.gamma;
    if (delta < thresh) this.converged = true;
    this.renderer.update({ valueFn: this.V }, { animate: !prefersReducedMotion() });
    this.refreshUI();
  }

  private startPlay() {
    if (this.playing) return;
    this.playing = true;
    const tick = () => {
      if (!this.playing || this.converged) { this.playing = false; return; }
      this.step();
      this.playTimer = window.setTimeout(tick, 80);
    };
    tick();
  }

  private stopPlay() {
    this.playing = false;
    clearTimeout(this.playTimer);
  }

  private refreshUI() {
    const thresh = this.epsilon * (1 - this.gamma) / this.gamma;
    const last = this.history[this.history.length - 1];
    this.kEl.textContent = `k = ${this.k}`;
    this.deltaEl.textContent = `‖ΔV‖∞ = ${last.delta === 0 ? '—' : last.delta.toFixed(5)}`;
    this.threshEl.textContent = `stop at < ${thresh.toFixed(5)}`;
    this.badgeEl.style.display = this.converged ? '' : 'none';
    this.badgeEl.textContent = `converged at k=${this.k}`;
    this.setStatus(`k=${this.k}${this.converged ? ' ✓' : ''}`);
    this.updatePlot();
  }

  private updatePlot() {
    const svg = this.plotSvg;
    svg.selectAll('.pe-line,.pe-dot,.pe-axis').remove();
    const hist = this.history;
    if (hist.length < 2) return;
    const margin = { top: 14, right: 14, bottom: 30, left: 44 };
    const W = PLOT_W - margin.left - margin.right;
    const H = PLOT_H - margin.top - margin.bottom;
    const g = svg.append('g').attr('class', 'pe-axis')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xs = d3.scaleLinear().domain([0, hist[hist.length - 1].k]).range([0, W]);
    const allV = hist.map(h => h.V[idx(0, 0)]);
    const yMin = Math.min(...allV) - 0.05;
    const yMax = Math.max(...allV) + 0.05;
    const ys = d3.scaleLinear().domain([yMin, yMax]).range([H, 0]);

    g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(5))
      .attr('class', 'annot');
    g.append('g').call(d3.axisLeft(ys).ticks(4))
      .attr('class', 'annot');

    // V(0,0) line
    const line = d3.line<{ k: number; V: number[] }>()
      .x(d => xs(d.k)).y(d => ys(d.V[idx(0, 0)]));
    g.append('path').datum(hist).attr('class', 'pe-line')
      .attr('fill', 'none')
      .attr('stroke', 'var(--dp-phase-evaluation)')
      .attr('stroke-width', 2)
      .attr('d', line);

    // convergence line marker
    if (this.converged) {
      g.append('line').attr('class', 'pe-line')
        .attr('x1', xs(this.k)).attr('x2', xs(this.k))
        .attr('y1', 0).attr('y2', H)
        .attr('stroke', 'var(--dp-improving)').attr('stroke-dasharray', '4 3').attr('stroke-width', 1.5);
    }

    // x-axis label
    g.append('text').attr('class', 'pe-line annot')
      .attr('x', W / 2).attr('y', H + 26).attr('text-anchor', 'middle')
      .attr('fill', 'var(--rl-ink-faint)').style('font-size', '10px')
      .text('iteration k');
    g.append('text').attr('class', 'pe-line annot')
      .attr('x', -H / 2).attr('y', -32).attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('fill', 'var(--rl-ink-faint)').style('font-size', '10px')
      .text('V(0,0)');
  }

  private build() {
    this.innerHTML = '';
    const { panel, body, setStatus } = createPanel({ id: 'v1-iterative-pe-watcher' });
    this.setStatus = setStatus;
    this.appendChild(panel);

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start';
    body.appendChild(row);

    // Gridworld
    const gridCol = document.createElement('div');
    gridCol.style.cssText = 'display:flex;flex-direction:column;gap:8px';
    row.appendChild(gridCol);

    this.mdp = buildGridworld({ slippery: false, gamma: this.gamma });
    this.policy = this.getPolicy(this.mdp);
    this.V = new Array(9).fill(0);
    this.k = 0;
    this.history = [{ k: 0, V: this.V.slice(), delta: 0 }];

    this.renderer = new GridworldRenderer(gridCol, {
      mdp: this.mdp, valueFn: this.V, policy: this.policy,
      showValues: true, showArrows: true, startState: 0, cellPx: 88,
    });

    // status bar
    const statBar = document.createElement('div');
    statBar.style.cssText = 'display:flex;gap:12px;font-size:12px;font-family:var(--rl-font-mono);flex-wrap:wrap';
    gridCol.appendChild(statBar);

    this.kEl = Object.assign(document.createElement('span'), {
      style: 'color:var(--dp-phase-evaluation)',
    });
    this.deltaEl = Object.assign(document.createElement('span'), {
      style: 'color:var(--rl-ink-faint)',
    });
    this.threshEl = Object.assign(document.createElement('span'), {
      style: 'color:var(--rl-ink-muted)',
    });
    this.badgeEl = Object.assign(document.createElement('span'), {
      style: 'background:var(--dp-improving);color:#fff;padding:1px 6px;border-radius:4px;display:none',
    });
    statBar.append(this.kEl, this.deltaEl, this.threshEl, this.badgeEl);

    // controls
    const ctrl = document.createElement('div');
    ctrl.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px';
    gridCol.appendChild(ctrl);

    const mkBtn = (label: string, cb: () => void) => {
      const b = document.createElement('button');
      b.className = 'rl-btn'; b.textContent = label;
      b.addEventListener('click', cb); ctrl.appendChild(b); return b;
    };
    mkBtn('Step', () => { this.stopPlay(); this.step(); });
    mkBtn('Play', () => this.startPlay());
    mkBtn('Pause', () => this.stopPlay());
    mkBtn('Reset', () => this.reset());

    // Policy selector
    const selLabel = document.createElement('label');
    selLabel.style.cssText = 'font-size:12px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui)';
    selLabel.textContent = 'Policy: ';
    const sel = document.createElement('select');
    sel.className = 'rl-select';
    [['uniform', 'Uniform random'], ['all-down', 'All Down'], ['all-right', 'All Right']]
      .forEach(([v, l]) => {
        const o = document.createElement('option'); o.value = v; o.textContent = l;
        sel.appendChild(o);
      });
    sel.addEventListener('change', () => {
      this.policyName = sel.value as typeof this.policyName;
      this.reset();
    });
    selLabel.appendChild(sel);
    ctrl.appendChild(selLabel);

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
      ctrl.appendChild(wrap);
    };
    mkSlider('γ', 0.5, 0.99, 0.01, this.gamma, v => { this.gamma = v; this.reset(); });
    mkSlider('ε', 0.001, 0.1, 0.001, this.epsilon, v => { this.epsilon = v; this.refreshUI(); });

    // Plot
    const plotCol = document.createElement('div');
    plotCol.style.cssText = 'display:flex;flex-direction:column;gap:4px';
    row.appendChild(plotCol);

    const plotTitle = document.createElement('div');
    plotTitle.style.cssText = 'font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui)';
    plotTitle.textContent = 'V_k(0,0) over iterations';
    plotCol.appendChild(plotTitle);

    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', String(PLOT_W)); svgEl.setAttribute('height', String(PLOT_H));
    svgEl.classList.add('rl-svg');
    plotCol.appendChild(svgEl);
    this.plotSvg = d3.select(svgEl as SVGSVGElement);

    // Color scale legend
    const legend = document.createElement('div');
    legend.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:11px;font-family:var(--rl-font-ui);color:var(--rl-ink-faint);margin-top:4px';
    const csEl = document.createElement('div');
    csEl.style.cssText = `width:100px;height:10px;border-radius:3px;background:linear-gradient(to right, #b91c1c, #f1ede4, #15803d)`;
    legend.innerHTML = '<span>−1</span>';
    legend.appendChild(csEl);
    legend.innerHTML += '<span>+1</span>';
    plotCol.appendChild(legend);

    this.refreshUI();
  }
}

customElements.define('iterative-pe-watcher', IterativePEWatcher);
