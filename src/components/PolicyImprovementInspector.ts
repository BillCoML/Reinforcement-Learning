/**
 * V2 — Policy Improvement Inspector.
 * Two stacked gridworlds: top shows V^π (color + values),
 * bottom shows the greedy policy π'. Click a cell to see per-action Q-values.
 * Features the "bounce at (0,0)" callout.
 */
import { createPanel } from './PanelChrome';
import { GridworldRenderer } from './GridworldRenderer';
import { buildGridworld, uniformPolicy } from '../mdp/gridworld';
import { policyEvaluationExact } from '../mdp/policy-evaluation';
import { greedyActions } from '../mdp/value-iteration';
import { qFromV } from '../mdp/q-and-advantage';
import { policyImprovement } from '../dp/algorithms';
import { ACTION_NAMES, idx, type MDP, type Policy } from '../mdp/types';

export class PolicyImprovementInspector extends HTMLElement {
  private mdp!: MDP;
  private V!: number[];
  private greedyPol!: Policy;
  private selectedState: number | null = null;
  private topRenderer!: GridworldRenderer;
  private botRenderer!: GridworldRenderer;
  private sidePanel!: HTMLElement;

  connectedCallback() { this.build(); }

  private computeAndUpdate() {
    this.greedyPol = policyImprovement(this.mdp, this.V);
    this.topRenderer?.update({ valueFn: this.V, policy: this.greedyPol });
    // Bottom renderer shows the resulting greedy policy overlaid on the same V
    this.botRenderer?.update({
      valueFn: this.V,
      policy: this.greedyPol,
      showArrows: true,
      showValues: false,
    });
    this.updateBounceCallout();
  }

  private updateBounceCallout() {
    const bounceSt = idx(0, 0);
    const tied = greedyActions(this.mdp, this.V, bounceSt);
    const bounces = tied.every(a => {
      // check if action leads back to same state (wall bounce)
      return this.mdp.P[bounceSt][a][bounceSt] > 0.5;
    });

    const callout = this.querySelector('.dp-bounce-callout') as HTMLElement;
    if (callout) callout.style.display = bounces ? '' : 'none';
  }

  private updateSidePanel(s: number) {
    this.selectedState = s;
    const Q = qFromV(this.mdp, this.V);
    const tied = greedyActions(this.mdp, this.V, s);
    const chosen = policyImprovement(this.mdp, this.V).pi[s].indexOf(1);

    this.sidePanel.innerHTML = `
      <div style="font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui);margin-bottom:8px">
        State s = (${Math.floor(s/3)},${s%3}) — Q-values under current V
      </div>
      ${ACTION_NAMES.map((name, a) => {
        const q = this.mdp.terminals[s] ? 0 : Q[s][a];
        const isChosen = a === chosen;
        const isTied = tied.includes(a);
        const barW = Math.max(0, Math.min(120, (q + 1.5) / 2.5 * 120));
        return `<div style="margin-bottom:6px">
          <div style="display:flex;align-items:center;gap:6px;font-size:12px;font-family:var(--rl-font-mono)">
            <span style="width:36px;color:${isChosen ? 'var(--dp-phase-improvement)' : 'var(--rl-ink-muted)'}">${name}</span>
            <div style="background:var(--rl-surface-2);border-radius:3px;width:120px;height:12px;position:relative">
              <div style="background:${isChosen ? 'var(--dp-phase-improvement)' : isTied ? 'var(--dp-phase-evaluation)' : 'var(--rl-ink-faint)'};width:${barW}px;height:100%;border-radius:3px;opacity:${isTied ? 1 : 0.4}"></div>
            </div>
            <span style="color:${isChosen ? 'var(--dp-phase-improvement)' : 'var(--rl-ink-faint)'}">${q.toFixed(4)}</span>
            ${isChosen ? '<span style="font-size:10px;background:var(--dp-phase-improvement);color:#fff;padding:1px 4px;border-radius:3px">chosen</span>' : ''}
            ${isTied && !isChosen ? '<span style="font-size:10px;background:var(--dp-phase-evaluation);color:#fff;padding:1px 4px;border-radius:3px">tied</span>' : ''}
          </div>
        </div>`;
      }).join('')}
      ${this.mdp.terminals[s] ? '<p style="font-size:11px;color:var(--rl-ink-muted)">Terminal state — V = 0, policy irrelevant.</p>' : ''}
    `;
    this.topRenderer.update({ selectedState: s });
    this.botRenderer.update({ selectedState: s });
  }

  private build() {
    this.innerHTML = '';
    const { panel, body, setStatus: _ss } = createPanel({ id: 'v2-policy-improvement-inspector' });
    this.appendChild(panel);
    const setStatus = _ss;

    this.mdp = buildGridworld({ slippery: false, gamma: 0.9 });
    const uniform = uniformPolicy(this.mdp);
    this.V = policyEvaluationExact(this.mdp, uniform);

    const mainRow = document.createElement('div');
    mainRow.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start';
    body.appendChild(mainRow);

    // ── Left: two stacked grids ──
    const gridCol = document.createElement('div');
    gridCol.style.cssText = 'display:flex;flex-direction:column;gap:12px';
    mainRow.appendChild(gridCol);

    const mkLabel = (text: string, color: string) => {
      const el = document.createElement('div');
      el.style.cssText = `font-size:11px;font-family:var(--rl-font-ui);color:${color};margin-bottom:2px`;
      el.textContent = text;
      return el;
    };

    const topWrap = document.createElement('div');
    topWrap.appendChild(mkLabel('Input: V^π (click cell to inspect Q-values)', 'var(--dp-phase-evaluation)'));
    this.topRenderer = new GridworldRenderer(topWrap, {
      mdp: this.mdp, valueFn: this.V, showValues: true, cellPx: 82,
      onCellClick: s => this.updateSidePanel(s),
    });
    gridCol.appendChild(topWrap);

    const botWrap = document.createElement('div');
    botWrap.appendChild(mkLabel('Result: greedy policy π\' (arrows show argmax action)', 'var(--dp-phase-improvement)'));
    this.botRenderer = new GridworldRenderer(botWrap, {
      mdp: this.mdp, valueFn: this.V, showArrows: true, showValues: false, cellPx: 82,
      onCellClick: s => this.updateSidePanel(s),
    });
    gridCol.appendChild(botWrap);

    // Bounce callout
    const bounceCallout = document.createElement('div');
    bounceCallout.className = 'dp-bounce-callout forward-link';
    bounceCallout.style.cssText = 'max-width:320px;margin-top:4px';
    bounceCallout.innerHTML = `
      <span class="label" style="color:var(--dp-phase-improvement)">Surprised? — (0,0) bounces off wall</span>
      <p style="font-size:13px;margin:0">At (0,0), <strong>greedy improvement picks Up or Left</strong> —
      both bounce off the wall, effectively keeping the agent at (0,0) forever.
      This is better than moving toward more-negative neighbors under the current $V^{\\pi_0}$.
      The "stuck" behaviour is temporary; the next round of improvement will fix it.</p>`;
    gridCol.appendChild(bounceCallout);

    // Controls
    const ctrl = document.createElement('div');
    ctrl.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px';
    gridCol.appendChild(ctrl);

    const mkBtn = (label: string, cb: () => void) => {
      const b = document.createElement('button');
      b.className = 'rl-btn'; b.textContent = label;
      b.addEventListener('click', cb); ctrl.appendChild(b);
    };

    const policies: Array<{ label: string; policy: Policy }> = [];
    let policyIdx = 0;

    const loadPolicy = (pol: Policy, label: string) => {
      this.V = policyEvaluationExact(this.mdp, pol);
      this.computeAndUpdate();
      setStatus(label);
      if (this.selectedState !== null) this.updateSidePanel(this.selectedState);
    };

    mkBtn('← Previous policy', () => {
      policyIdx = Math.max(0, policyIdx - 1);
      loadPolicy(policies[policyIdx].policy, policies[policyIdx].label);
    });
    mkBtn('Improve once →', () => {
      const newPol = policyImprovement(this.mdp, this.V);
      policies.push({ label: `π after improvement`, policy: newPol });
      policyIdx = policies.length - 1;
      loadPolicy(newPol, `π${policyIdx}`);
    });
    mkBtn('Reset (uniform)', () => {
      const uni = uniformPolicy(this.mdp);
      policies.length = 0;
      policies.push({ label: 'Uniform π₀', policy: uni });
      policyIdx = 0;
      loadPolicy(uni, 'Uniform π₀');
    });

    // ── Right: Q-value side panel ──
    const sideWrap = document.createElement('div');
    sideWrap.style.cssText = 'min-width:220px;max-width:280px';
    const sideTitle = document.createElement('div');
    sideTitle.style.cssText = 'font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui);margin-bottom:8px';
    sideTitle.textContent = 'Click a cell to inspect Q-values';
    sideWrap.appendChild(sideTitle);
    this.sidePanel = document.createElement('div');
    sideWrap.appendChild(this.sidePanel);
    mainRow.appendChild(sideWrap);

    // init
    policies.push({ label: 'Uniform π₀', policy: uniform });
    this.computeAndUpdate();
    this.updateSidePanel(0);
  }
}

customElements.define('policy-improvement-inspector', PolicyImprovementInspector);
