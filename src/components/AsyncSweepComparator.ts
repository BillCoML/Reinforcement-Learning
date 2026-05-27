/**
 * V5 — Async Sweep Comparator.
 * Three gridworld renderers side-by-side running Jacobi, Forward, and Reverse
 * sweep orders. All step together; a "winner" indicator appears when one converges.
 */
import { createPanel } from './PanelChrome';
import { GridworldRenderer } from './GridworldRenderer';
import { buildGridworld } from '../mdp/gridworld';
import { bellmanOptimalityBackup, qOfAction } from '../mdp/value-iteration';
import { supDist } from '../dp/algorithms';
import { type MDP } from '../mdp/types';
import { prefersReducedMotion } from './base';

const STOP_EPS = 1e-8;

interface PanelState {
  name: string;
  sweepOrder: number[];
  V: number[];
  iter: number;
  converged: boolean;
  async: boolean; // true = Gauss-Seidel, false = Jacobi synchronous
  iterEl: HTMLElement;
  convEl: HTMLElement;
  renderer: GridworldRenderer;
}

export class AsyncSweepComparator extends HTMLElement {
  private mdp!: MDP;
  private panels: PanelState[] = [];
  private playing = false;
  private playTimer = 0;
  private setStatus!: (t: string) => void;

  connectedCallback() { this.build(); }

  private makePanel(container: HTMLElement, name: string, order: number[], isAsync: boolean): PanelState {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;flex-direction:column;gap:6px;flex:1;min-width:200px';
    container.appendChild(wrap);

    const colorMap: Record<string, string> = {
      'Jacobi': 'var(--dp-phase-optimality)',
      'Forward': 'var(--dp-phase-evaluation)',
      'Reverse': 'var(--dp-phase-async)',
      'Smart':   'var(--dp-phase-improvement)',
    };
    const color = colorMap[name] ?? 'var(--rl-ink-faint)';

    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:6px';
    const badge = document.createElement('span');
    badge.style.cssText = `font-size:12px;font-family:var(--rl-font-ui);font-weight:600;color:${color};border-bottom:2px solid ${color};padding-bottom:1px`;
    badge.textContent = name;
    header.appendChild(badge);
    const asyncTag = document.createElement('span');
    asyncTag.style.cssText = 'font-size:10px;color:var(--rl-ink-muted);font-family:var(--rl-font-ui)';
    asyncTag.textContent = isAsync ? '(in-place)' : '(sync batch)';
    header.appendChild(asyncTag);
    wrap.appendChild(header);

    const renderer = new GridworldRenderer(wrap, {
      mdp: this.mdp, valueFn: new Array(9).fill(0), showValues: true, cellPx: 72,
    });

    const iterEl = document.createElement('div');
    iterEl.style.cssText = 'font-size:11px;font-family:var(--rl-font-mono);color:var(--rl-ink-faint)';
    iterEl.textContent = 'iter: 0';
    wrap.appendChild(iterEl);

    const convEl = document.createElement('div');
    convEl.style.cssText = 'font-size:11px;font-family:var(--rl-font-ui);background:var(--dp-improving);color:#fff;padding:2px 8px;border-radius:4px;display:none';
    convEl.textContent = '✓ converged';
    wrap.appendChild(convEl);

    return { name, sweepOrder: order, V: new Array(9).fill(0), iter: 0, converged: false, async: isAsync, iterEl, convEl, renderer };
  }

  private stepPanel(p: PanelState) {
    if (p.converged) return;
    const thresh = STOP_EPS * (1 - this.mdp.gamma) / this.mdp.gamma;

    if (!p.async) {
      // Jacobi: synchronous T* backup
      const Vn = bellmanOptimalityBackup(this.mdp, p.V);
      const delta = supDist(Vn, p.V);
      p.V = Vn;
      p.iter++;
      if (delta < thresh) p.converged = true;
    } else {
      // Gauss-Seidel: in-place sweep
      let maxDelta = 0;
      for (const s of p.sweepOrder) {
        if (this.mdp.terminals[s]) continue;
        const oldV = p.V[s];
        let best = -Infinity;
        for (let a = 0; a < this.mdp.nA; a++) best = Math.max(best, qOfAction(this.mdp, p.V, s, a));
        p.V[s] = best;
        maxDelta = Math.max(maxDelta, Math.abs(p.V[s] - oldV));
      }
      p.iter++;
      if (maxDelta < thresh) p.converged = true;
    }

    p.renderer.update({ valueFn: p.V }, { animate: !prefersReducedMotion() });
    p.iterEl.textContent = `iter: ${p.iter}`;
    if (p.converged) p.convEl.style.display = '';
  }

  private stepAll() {
    const wasAllDone = this.panels.every(p => p.converged);
    if (wasAllDone) return;
    this.panels.forEach(p => this.stepPanel(p));
    this.setStatus(this.panels.map(p => `${p.name}:${p.iter}${p.converged ? '✓' : ''}`).join(' '));
    if (this.panels.every(p => p.converged)) this.stopPlay();
  }

  private resetAll() {
    this.stopPlay();
    this.panels.forEach(p => {
      p.V = new Array(9).fill(0);
      p.iter = 0;
      p.converged = false;
      p.convEl.style.display = 'none';
      p.iterEl.textContent = 'iter: 0';
      p.renderer.update({ valueFn: p.V });
    });
    this.setStatus('reset');
  }

  private startPlay() {
    if (this.playing) return;
    this.playing = true;
    const tick = () => {
      if (!this.playing || this.panels.every(p => p.converged)) { this.playing = false; return; }
      this.stepAll();
      this.playTimer = window.setTimeout(tick, 250);
    };
    tick();
  }

  private stopPlay() { this.playing = false; clearTimeout(this.playTimer); }

  private build() {
    this.innerHTML = '';
    const { panel, body, setStatus } = createPanel({ id: 'v5-async-sweep-comparator' });
    this.setStatus = setStatus;
    this.appendChild(panel);

    this.mdp = buildGridworld({ slippery: false, gamma: 0.9 });

    // Summary bar
    const summaryBar = document.createElement('div');
    summaryBar.style.cssText = 'display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;padding:8px;background:var(--rl-surface-2);border-radius:6px';
    summaryBar.innerHTML = `
      <span style="font-size:12px;font-family:var(--rl-font-ui);color:var(--rl-ink-faint)">Pre-computed iterations to convergence:</span>
      <span style="font-size:12px;font-family:var(--rl-font-mono);color:var(--dp-phase-optimality)">Jacobi: 5</span>
      <span style="font-size:12px;font-family:var(--rl-font-mono);color:var(--dp-phase-evaluation)">Forward: 5</span>
      <span style="font-size:12px;font-family:var(--rl-font-mono);color:var(--dp-phase-async)">Reverse: 2</span>
    `;
    body.appendChild(summaryBar);

    // Panels row
    const panelRow = document.createElement('div');
    panelRow.style.cssText = 'display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start';
    body.appendChild(panelRow);

    const fwd = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const rev = [8, 7, 6, 5, 4, 3, 2, 1, 0];

    this.panels = [
      this.makePanel(panelRow, 'Jacobi', fwd, false),
      this.makePanel(panelRow, 'Forward', fwd, true),
      this.makePanel(panelRow, 'Reverse', rev, true),
    ];

    // Smart sweep option (dropdown)
    const smartOrder = [8, 7, 6, 5, 4, 3, 2, 1, 0].sort((a, b) => {
      const distA = Math.abs(Math.floor(a/3) - 2) + Math.abs(a%3 - 2);
      const distB = Math.abs(Math.floor(b/3) - 2) + Math.abs(b%3 - 2);
      return distA - distB;
    });

    // Controls
    const ctrl = document.createElement('div');
    ctrl.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px';
    body.appendChild(ctrl);

    const mkBtn = (label: string, cb: () => void) => {
      const b = document.createElement('button');
      b.className = 'rl-btn'; b.textContent = label;
      b.addEventListener('click', cb); ctrl.appendChild(b);
    };
    mkBtn('Step all', () => { this.stopPlay(); this.stepAll(); });
    mkBtn('Play', () => this.startPlay());
    mkBtn('Pause', () => this.stopPlay());
    mkBtn('Reset', () => this.resetAll());

    // Swap Jacobi panel with Smart sweep
    const swapWrap = document.createElement('div');
    swapWrap.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;font-family:var(--rl-font-ui);color:var(--rl-ink-faint)';
    swapWrap.innerHTML = '<span>Panel 3:</span>';
    const swapSel = document.createElement('select');
    swapSel.className = 'rl-select';
    [['Reverse', 'reverse'], ['Smart (near-goal first)', 'smart']].forEach(([l, v]) => {
      const o = document.createElement('option'); o.value = v; o.textContent = l;
      swapSel.appendChild(o);
    });
    swapSel.addEventListener('change', () => {
      const p = this.panels[2];
      p.sweepOrder = swapSel.value === 'smart' ? smartOrder : rev;
      p.name = swapSel.value === 'smart' ? 'Smart' : 'Reverse';
      this.resetAll();
    });
    swapWrap.appendChild(swapSel);
    ctrl.appendChild(swapWrap);

    setStatus('ready');
  }
}

customElements.define('async-sweep-comparator', AsyncSweepComparator);
