/**
 * V6 — GPI Visualizer.
 * Abstract 2D plane: V-consistency (vertical) vs π-consistency (horizontal).
 * Target: (V*, π*). Staircase trajectory showing alternation.
 * Algorithm overlays: PI, VI, async, actor-critic preview.
 */
import * as d3 from 'd3';
import { createPanel } from './PanelChrome';

type AlgoOverlay = 'PI' | 'VI' | 'Async' | 'Actor-Critic';

interface TrajectoryPoint { x: number; y: number }

function makePITrajectory(): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [{ x: 0.05, y: 0.08 }];
  let x = 0.05, y = 0.08;
  const steps = 4;
  for (let i = 0; i < steps; i++) {
    const newY = y + (1 - y) * 0.7;
    pts.push({ x, y: newY });
    const newX = x + (1 - x) * 0.7;
    pts.push({ x: newX, y: newY });
    x = newX; y = newY;
  }
  return pts;
}

function makeVITrajectory(): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [{ x: 0.05, y: 0.08 }];
  let x = 0.05, y = 0.08;
  const steps = 6;
  for (let i = 0; i < steps; i++) {
    // small steps diagonally — each iteration advances both slightly
    const newY = y + (1 - y) * 0.35;
    const newX = x + (1 - x) * 0.35;
    pts.push({ x, y: newY });
    pts.push({ x: newX, y: newY });
    x = newX; y = newY;
  }
  return pts;
}

function makeAsyncTrajectory(): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [{ x: 0.05, y: 0.08 }];
  let x = 0.05, y = 0.08;
  for (let i = 0; i < 8; i++) {
    const jitter = (Math.random() - 0.5) * 0.08;
    const newY = y + (1 - y) * (0.3 + jitter);
    const jitter2 = (Math.random() - 0.5) * 0.08;
    const newX = x + (1 - x) * (0.3 + jitter2);
    pts.push({ x: x + jitter * 0.3, y: newY });
    pts.push({ x: newX, y: newY + jitter2 * 0.3 });
    x = newX; y = newY;
  }
  return pts;
}

function makeActorCriticTrajectory(): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [{ x: 0.05, y: 0.08 }];
  let x = 0.05, y = 0.08;
  for (let i = 0; i < 12; i++) {
    // smooth diagonal
    x = x + (1 - x) * 0.18;
    y = y + (1 - y) * 0.18;
    pts.push({ x, y });
  }
  return pts;
}

const ALGO_DATA: Record<AlgoOverlay, { label: string; color: string; traj: TrajectoryPoint[] }> = {
  PI: { label: 'Policy Iteration', color: 'var(--dp-phase-improvement)', traj: makePITrajectory() },
  VI: { label: 'Value Iteration', color: 'var(--dp-phase-optimality)', traj: makeVITrajectory() },
  Async: { label: 'Async DP', color: 'var(--dp-phase-async)', traj: makeAsyncTrajectory() },
  'Actor-Critic': { label: 'Actor-Critic (preview)', color: 'var(--rl-algo-ucb)', traj: makeActorCriticTrajectory() },
};

export class GPIVisualizer extends HTMLElement {
  private selected: Set<AlgoOverlay> = new Set(['PI', 'VI']);
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private W = 640; private H = 420;

  connectedCallback() { this.build(); }

  private drawDiagram() {
    const svg = this.svg;
    svg.selectAll('.gpi-layer').remove();
    const g = svg.append('g').attr('class', 'gpi-layer');
    const margin = { top: 30, right: 40, bottom: 60, left: 60 };
    const innerW = this.W - margin.left - margin.right;
    const innerH = this.H - margin.top - margin.bottom;

    const gInner = g.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xs = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const ys = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    // Background grid
    gInner.append('rect').attr('width', innerW).attr('height', innerH)
      .attr('fill', 'var(--rl-surface-2)').attr('rx', 6);

    // Axes
    gInner.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', innerH).attr('y2', innerH)
      .attr('stroke', 'var(--rl-border)').attr('stroke-width', 1.5);
    gInner.append('line').attr('x1', 0).attr('x2', 0).attr('y1', 0).attr('y2', innerH)
      .attr('stroke', 'var(--rl-border)').attr('stroke-width', 1.5);

    // Axis labels
    gInner.append('text').attr('x', innerW / 2).attr('y', innerH + 36)
      .attr('text-anchor', 'middle').attr('class', 'annot').attr('fill', 'var(--dp-phase-improvement)')
      .style('font-size', '13px').style('font-weight', '600')
      .text('π-consistency (greedy w.r.t. V)  →');
    gInner.append('text').attr('x', -innerH / 2).attr('y', -40)
      .attr('transform', 'rotate(-90)').attr('text-anchor', 'middle')
      .attr('class', 'annot').attr('fill', 'var(--dp-phase-evaluation)')
      .style('font-size', '13px').style('font-weight', '600')
      .text('← V-consistency (V = V^π)');

    // Target star (V*, π*)
    const tx = xs(0.97), ty = ys(0.97);
    gInner.append('circle').attr('cx', tx).attr('cy', ty).attr('r', 14)
      .attr('fill', 'var(--dp-phase-async)').attr('opacity', 0.15);
    gInner.append('circle').attr('cx', tx).attr('cy', ty).attr('r', 6)
      .attr('fill', 'var(--dp-phase-async)');
    gInner.append('text').attr('x', tx + 10).attr('y', ty - 10)
      .attr('class', 'annot').attr('fill', 'var(--dp-phase-async)')
      .style('font-size', '12px').style('font-weight', '600')
      .text('(V*, π*)');

    // Start dot
    gInner.append('circle').attr('cx', xs(0.05)).attr('cy', ys(0.08)).attr('r', 4)
      .attr('fill', 'var(--rl-ink-muted)');
    gInner.append('text').attr('x', xs(0.05) + 6).attr('y', ys(0.08) + 14)
      .attr('class', 'annot').attr('fill', 'var(--rl-ink-muted)').style('font-size', '10px')
      .text('start');

    // Draw trajectories for selected algorithms
    Array.from(this.selected).forEach(algo => {
      const { color, traj } = ALGO_DATA[algo];
      const lineGen = d3.line<TrajectoryPoint>().x(d => xs(d.x)).y(d => ys(d.y));
      gInner.append('path').datum(traj).attr('fill', 'none').attr('stroke', color)
        .attr('stroke-width', 2.5).attr('stroke-dasharray', algo === 'Actor-Critic' ? '6 3' : 'none')
        .attr('d', lineGen).attr('opacity', 0.85);
      traj.forEach((pt, i) => {
        if (i === 0 || i === traj.length - 1) return;
        gInner.append('circle').attr('cx', xs(pt.x)).attr('cy', ys(pt.y)).attr('r', 3)
          .attr('fill', color).attr('opacity', 0.6);
      });
      // End arrow dot
      const last = traj[traj.length - 1];
      gInner.append('circle').attr('cx', xs(last.x)).attr('cy', ys(last.y)).attr('r', 5)
        .attr('fill', color).attr('stroke', 'var(--rl-surface)').attr('stroke-width', 1.5);
    });
  }

  private build() {
    this.innerHTML = '';
    const { panel, body, setStatus } = createPanel({ id: 'v6-gpi-visualizer' });
    this.appendChild(panel);

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start';
    body.appendChild(row);

    // SVG
    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', String(this.W)); svgEl.setAttribute('height', String(this.H));
    svgEl.setAttribute('viewBox', `0 0 ${this.W} ${this.H}`);
    svgEl.style.maxWidth = '100%';
    svgEl.classList.add('rl-svg');
    row.appendChild(svgEl);
    this.svg = d3.select(svgEl as SVGSVGElement);

    // Legend / toggles
    const legend = document.createElement('div');
    legend.style.cssText = 'display:flex;flex-direction:column;gap:8px;min-width:160px';
    row.appendChild(legend);

    const legTitle = document.createElement('div');
    legTitle.style.cssText = 'font-size:11px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui);margin-bottom:4px';
    legTitle.textContent = 'Algorithm overlays:';
    legend.appendChild(legTitle);

    (Object.keys(ALGO_DATA) as AlgoOverlay[]).forEach(algo => {
      const { label, color } = ALGO_DATA[algo];
      const btn = document.createElement('button');
      btn.className = 'rl-btn';
      btn.style.cssText = `border-left:3px solid ${color};text-align:left;opacity:${this.selected.has(algo) ? 1 : 0.4}`;
      btn.textContent = label;
      btn.addEventListener('click', () => {
        if (this.selected.has(algo)) this.selected.delete(algo);
        else this.selected.add(algo);
        btn.style.opacity = this.selected.has(algo) ? '1' : '0.4';
        this.drawDiagram();
        setStatus(Array.from(this.selected).join(' | '));
      });
      legend.appendChild(btn);
    });

    // Explanation
    const explainer = document.createElement('div');
    explainer.style.cssText = 'font-size:12px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui);margin-top:12px;max-width:160px;line-height:1.5';
    explainer.innerHTML = `
      <strong style="color:var(--dp-phase-evaluation)">Vertical movement</strong> = policy evaluation step (V moves toward V^π).<br>
      <strong style="color:var(--dp-phase-improvement)">Horizontal movement</strong> = policy improvement step (π becomes greedier w.r.t. V).<br>
      All algorithms converge to the same target.
    `;
    legend.appendChild(explainer);

    this.drawDiagram();
    setStatus('PI + VI shown');
  }
}

customElements.define('gpi-visualizer', GPIVisualizer);
