/**
 * V3 — Policy Iteration Trace.
 * Reproduces the 3-iteration PI trace with the (0,0) sequence −0.4205→0→0.729.
 * Shows gridworld with policy arrows + V shading, a bar chart per state,
 * a line plot of V(0,0) over outer iterations, and a Modified PI toggle.
 */
import * as d3 from 'd3';
import { createPanel } from './PanelChrome';
import { GridworldRenderer } from './GridworldRenderer';
import { buildGridworld } from '../mdp/gridworld';
import { policyIteration, modifiedPolicyIteration, policyImprovement } from '../dp/algorithms';
import { idx, type MDP } from '../mdp/types';
import type { PIStep } from '../dp/algorithms';
import { makeValueColorScale } from './value-scale';

const PLOT_W = 260, PLOT_H = 180;

export class PolicyIterationTrace extends HTMLElement {
  private mdp!: MDP;
  private history: PIStep[] = [];
  private stepIdx = 0;
  private m = -1; // -1 = full PI
  private renderer!: GridworldRenderer;
  private plotSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private barSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private infoEl!: HTMLElement;
  private phaseEl!: HTMLElement;
  private setStatus!: (t: string) => void;

  connectedCallback() { this.build(); }

  private runAlgo() {
    this.mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    try {
      if (this.m <= 0 || !isFinite(this.m)) {
        const res = policyIteration(this.mdp);
        this.history = res.history;
      } else {
        const res = modifiedPolicyIteration(this.mdp, this.m);
        this.history = res.history;
      }
    } catch {
      this.history = [];
    }
    this.stepIdx = 0;
  }

  private showStep(i: number) {
    if (this.history.length === 0) return;
    this.stepIdx = Math.max(0, Math.min(this.history.length - 1, i));
    const step = this.history[this.stepIdx];

    // Compute what the next policy would be
    const nextPol = policyImprovement(this.mdp, step.V);

    this.renderer?.update({
      mdp: this.mdp,
      valueFn: step.V,
      policy: nextPol,
      showArrows: true,
      showValues: true,
    });

    const isFinal = this.stepIdx === this.history.length - 1;
    const v00 = step.V[idx(0, 0)];
    const bounceAt00 = this.stepIdx === 1 && Math.abs(v00) < 0.01;

    this.phaseEl.innerHTML = `
      <span style="color:var(--dp-phase-evaluation)">■ Eval</span> k=${this.stepIdx}
      → V_{π${this.stepIdx}}(0,0) = ${v00.toFixed(4)}
      ${bounceAt00 ? '<span style="background:var(--dp-phase-improvement);color:#fff;padding:1px 5px;border-radius:3px;font-size:10px;margin-left:4px">bounce ← stuck</span>' : ''}
      ${isFinal ? '<span style="background:var(--dp-improving);color:#fff;padding:1px 5px;border-radius:3px;font-size:10px;margin-left:4px">converged</span>' : ''}
    `;

    this.setStatus(`iter ${this.stepIdx + 1}/${this.history.length}${isFinal ? ' ✓' : ''}`);
    this.updateBarChart(step.V);
    this.updateLinePlot();
  }

  private updateBarChart(V: number[]) {
    const svg = this.barSvg;
    svg.selectAll('.bar-layer').remove();
    const g = svg.append('g').attr('class', 'bar-layer');

    const prevV = this.stepIdx > 0 ? this.history[this.stepIdx - 1].V : null;
    const scale = makeValueColorScale();
    const BAR_W = 220, BAR_H = 120;
    const xScale = d3.scaleBand().domain(d3.range(9).map(String)).range([0, BAR_W]).padding(0.15);
    const yMin = Math.min(-1, ...V);
    const yMax = Math.max(1, ...V);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([BAR_H, 0]);

    const margin = { top: 8, left: 28, bottom: 22 };
    const bg = g.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    bg.append('line').attr('x1', 0).attr('x2', BAR_W).attr('y1', yScale(0)).attr('y2', yScale(0))
      .attr('stroke', 'var(--rl-border)').attr('stroke-dasharray', '3 2');

    if (prevV) {
      d3.range(9).forEach(s => {
        const x = xScale(String(s))!;
        const bw = xScale.bandwidth();
        const y0 = yScale(Math.min(0, prevV[s]));
        const h = Math.abs(yScale(prevV[s]) - yScale(0));
        bg.append('rect').attr('x', x).attr('y', prevV[s] >= 0 ? yScale(prevV[s]) : y0)
          .attr('width', bw).attr('height', h)
          .attr('fill', scale(prevV[s])).attr('opacity', 0.2);
      });
    }

    d3.range(9).forEach(s => {
      const x = xScale(String(s))!;
      const bw = xScale.bandwidth();
      if (this.mdp.terminals[s]) {
        bg.append('text').attr('x', x + bw / 2).attr('y', BAR_H / 2)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
          .attr('class', 'annot').attr('fill', 'var(--rl-ink-muted)').style('font-size', '9px')
          .text(s === 4 ? '☠' : '⚑');
        return;
      }
      const v = V[s];
      const y = v >= 0 ? yScale(v) : yScale(0);
      const h = Math.abs(yScale(v) - yScale(0));
      bg.append('rect').attr('x', x).attr('y', y).attr('width', bw).attr('height', h)
        .attr('fill', scale(v)).attr('rx', 2);
    });

    bg.append('g').attr('transform', `translate(0,${BAR_H})`).call(
      d3.axisBottom(xScale).tickFormat(i => {
        const s = Number(i);
        return `(${Math.floor(s/3)},${s%3})`;
      }).tickSize(0)
    ).attr('class', 'annot').select('.domain').remove();
    bg.append('g').call(d3.axisLeft(yScale).ticks(4)).attr('class', 'annot');
  }

  private updateLinePlot() {
    const svg = this.plotSvg;
    svg.selectAll('.pi-plot').remove();
    const margin = { top: 12, right: 14, bottom: 28, left: 44 };
    const W = PLOT_W - margin.left - margin.right;
    const H = PLOT_H - margin.top - margin.bottom;
    const g = svg.append('g').attr('class', 'pi-plot')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const vals00 = this.history.map((h, k) => ({ k, v: h.V[idx(0, 0)] }));
    if (vals00.length === 0) return;

    const xs = d3.scaleLinear().domain([0, Math.max(1, this.history.length - 1)]).range([0, W]);
    const allV = vals00.map(d => d.v);
    const ys = d3.scaleLinear().domain([Math.min(...allV) - 0.1, Math.max(...allV) + 0.1]).range([H, 0]);

    g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(this.history.length)).attr('class', 'annot');
    g.append('g').call(d3.axisLeft(ys).ticks(4)).attr('class', 'annot');

    // All points line
    const line = d3.line<{ k: number; v: number }>().x(d => xs(d.k)).y(d => ys(d.v));
    g.append('path').datum(vals00).attr('fill', 'none').attr('stroke', 'var(--dp-phase-evaluation)')
      .attr('stroke-width', 2).attr('d', line);

    vals00.forEach(d => {
      const isCurrent = d.k === this.stepIdx;
      g.append('circle').attr('cx', xs(d.k)).attr('cy', ys(d.v)).attr('r', isCurrent ? 5 : 3.5)
        .attr('fill', d.k < this.stepIdx ? 'var(--dp-phase-evaluation)' : 'var(--rl-surface-2)')
        .attr('stroke', 'var(--dp-phase-evaluation)').attr('stroke-width', isCurrent ? 2.5 : 1.5);
      g.append('text').attr('x', xs(d.k)).attr('y', ys(d.v) - 8)
        .attr('text-anchor', 'middle').attr('class', 'annot').style('font-size', '9px')
        .attr('fill', 'var(--rl-ink-faint)').text(d.v.toFixed(3));
    });

    g.append('text').attr('x', W / 2).attr('y', H + 24).attr('text-anchor', 'middle')
      .attr('class', 'annot').attr('fill', 'var(--rl-ink-faint)').style('font-size', '10px')
      .text('PI outer iteration k');
    g.append('text').attr('x', -H / 2).attr('y', -34).attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle').attr('class', 'annot').attr('fill', 'var(--rl-ink-faint)')
      .style('font-size', '10px').text('V_{π_k}(0,0)');
  }

  private build() {
    this.innerHTML = '';
    const { panel, body, setStatus } = createPanel({ id: 'v3-policy-iteration-trace' });
    this.setStatus = setStatus;
    this.appendChild(panel);

    this.runAlgo();

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start';
    body.appendChild(row);

    // ── Gridworld ──
    const leftCol = document.createElement('div');
    leftCol.style.cssText = 'display:flex;flex-direction:column;gap:8px';
    row.appendChild(leftCol);

    this.phaseEl = document.createElement('div');
    this.phaseEl.style.cssText = 'font-size:12px;font-family:var(--rl-font-mono);min-height:24px';
    leftCol.appendChild(this.phaseEl);

    this.renderer = new GridworldRenderer(leftCol, {
      mdp: this.mdp,
      valueFn: this.history[0]?.V,
      showValues: true,
      showArrows: true,
      cellPx: 82,
    });

    // Controls
    const ctrl = document.createElement('div');
    ctrl.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:4px';
    leftCol.appendChild(ctrl);

    const mkBtn = (label: string, cb: () => void) => {
      const b = document.createElement('button');
      b.className = 'rl-btn'; b.textContent = label;
      b.addEventListener('click', cb); ctrl.appendChild(b);
    };
    mkBtn('← Prev', () => this.showStep(this.stepIdx - 1));
    mkBtn('Next →', () => this.showStep(this.stepIdx + 1));
    mkBtn('Play all', () => {
      let i = 0;
      const go = () => {
        if (i >= this.history.length) return;
        this.showStep(i++);
        setTimeout(go, 600);
      };
      go();
    });
    mkBtn('Reset', () => { this.runAlgo(); this.showStep(0); });

    // Modified PI toggle
    const mWrap = document.createElement('div');
    mWrap.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;font-family:var(--rl-font-ui);color:var(--rl-ink-faint)';
    mWrap.innerHTML = '<span>Modified PI m:</span>';
    const mSel = document.createElement('select');
    mSel.className = 'rl-select';
    [['∞ (full PI)', -1], ['10', 10], ['5', 5], ['3', 3], ['1 (≈VI)', 1]].forEach(([label, val]) => {
      const o = document.createElement('option'); o.value = String(val); o.textContent = String(label);
      mSel.appendChild(o);
    });
    mSel.addEventListener('change', () => {
      this.m = Number(mSel.value);
      this.runAlgo();
      this.showStep(0);
    });
    mWrap.appendChild(mSel);
    leftCol.appendChild(mWrap);

    // ── Right: bar chart + line plot ──
    const rightCol = document.createElement('div');
    rightCol.style.cssText = 'display:flex;flex-direction:column;gap:12px';
    row.appendChild(rightCol);

    // Bar chart
    const barTitle = document.createElement('div');
    barTitle.style.cssText = 'font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui)';
    barTitle.textContent = 'V_{π_k}(s) per state (faded = previous iteration)';
    rightCol.appendChild(barTitle);

    const barSvgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    barSvgEl.setAttribute('width', '280'); barSvgEl.setAttribute('height', '150');
    barSvgEl.classList.add('rl-svg');
    rightCol.appendChild(barSvgEl);
    this.barSvg = d3.select(barSvgEl as SVGSVGElement);

    // Line plot
    const plotTitle = document.createElement('div');
    plotTitle.style.cssText = 'font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui)';
    plotTitle.textContent = 'V_{π_k}(0,0) over outer iterations';
    rightCol.appendChild(plotTitle);

    const plotSvgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    plotSvgEl.setAttribute('width', String(PLOT_W)); plotSvgEl.setAttribute('height', String(PLOT_H));
    plotSvgEl.classList.add('rl-svg');
    rightCol.appendChild(plotSvgEl);
    this.plotSvg = d3.select(plotSvgEl as SVGSVGElement);

    this.infoEl = document.createElement('div');
    this.infoEl.style.cssText = 'font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui)';
    rightCol.appendChild(this.infoEl);

    this.showStep(0);
  }
}

customElements.define('policy-iteration-trace', PolicyIterationTrace);
