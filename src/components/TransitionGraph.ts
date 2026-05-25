/**
 * TransitionGraph — the reusable node-link drawing of a Markov chain.
 *
 * Used by V1 (weather explorer), V3 (classification inspector), V4 (stationary
 * finder) and V5 (convergence lab), and earmarked for reuse by Lesson 2's MDP
 * visualizations. It is a plain class (not a custom element) so it can be
 * embedded inside any panel and driven programmatically.
 *
 * Layout: a fixed circular arrangement for K ≤ 4, a D3 force layout (run to
 * convergence synchronously, then frozen) for K ≥ 5. Edges are quadratic
 * béziers, curved so that a bidirectional pair i⇄j separates cleanly; the
 * arrowhead sits on the node boundary. Self-loops are arcs above the node.
 * Edge thickness scales with probability (1–6px); labels show inline when
 * p ≥ 0.05, otherwise on hover. An optional moving "current state" marker can
 * be tweened with animate({from,to}).
 */
import * as d3 from "d3";
import { prefersReducedMotion } from "./base";

export interface TransitionGraphProps {
  P: number[][];
  /** Per-state node fill (CSS color). Defaults to the --mc-state-* palette. */
  stateColors?: string[];
  /** Per-state labels. Defaults to 0,1,2,… */
  stateNames?: string[];
  /** Index of the highlighted "current" state, or null. */
  currentState?: number | null;
  /** Optional translucent background tint per state (V3 class shading). */
  classTint?: (i: number) => string | null;
  width?: number;
  height?: number;
}

interface Pt {
  x: number;
  y: number;
}

const NS = "http://www.w3.org/2000/svg";
const EPS = 1e-9;
const STATE_VARS = [
  "--mc-state-1", "--mc-state-2", "--mc-state-3", "--mc-state-4",
  "--mc-state-5", "--mc-state-6", "--mc-state-7", "--mc-state-8",
];

export class TransitionGraph {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private tip: HTMLDivElement;
  private props: Required<Pick<TransitionGraphProps, "P">> & TransitionGraphProps;
  private pos: Pt[] = [];
  private centroid: Pt = { x: 0, y: 0 };
  private W: number;
  private H: number;
  private r = 22; // node radius
  private marker?: d3.Selection<SVGGElement, unknown, null, undefined>;

  constructor(container: HTMLElement, props: TransitionGraphProps) {
    this.props = props;
    this.W = props.width ?? 720;
    this.H = props.height ?? 360;

    container.style.position = container.style.position || "relative";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${this.W} ${this.H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg", "mc-graph");
    container.appendChild(svgEl);
    this.svg = d3.select(svgEl as SVGSVGElement);

    this.tip = document.createElement("div");
    this.tip.className = "rl-tooltip";
    container.appendChild(this.tip);

    this.defs();
    this.layout();
    this.draw();
  }

  /** Replace props and redraw (recomputing layout only if K changed). */
  update(props: TransitionGraphProps): void {
    const kChanged = props.P.length !== this.props.P.length;
    this.props = props;
    if (props.width) this.W = props.width;
    if (props.height) this.H = props.height;
    if (kChanged || this.pos.length !== props.P.length) this.layout();
    this.draw();
  }

  setCurrentState(i: number | null): void {
    this.props.currentState = i;
    this.draw();
  }

  private color(i: number): string {
    if (this.props.stateColors && this.props.stateColors[i]) return this.props.stateColors[i];
    return `var(${STATE_VARS[i % STATE_VARS.length]})`;
  }

  private label(i: number): string {
    return this.props.stateNames?.[i] ?? String(i);
  }

  private defs(): void {
    const defs = this.svg.append("defs");
    defs
      .append("marker")
      .attr("id", "tg-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M0,0 L10,5 L0,10 z")
      .attr("fill", "var(--mc-edge)");
  }

  private layout(): void {
    const K = this.props.P.length;
    const cx = this.W / 2;
    const cy = this.H / 2;
    if (K <= 4) {
      this.pos = this.circularLayout(K, cx, cy);
    } else {
      this.pos = this.forceLayout(K);
    }
    this.centroid = {
      x: this.pos.reduce((a, p) => a + p.x, 0) / K,
      y: this.pos.reduce((a, p) => a + p.y, 0) / K,
    };
  }

  private circularLayout(K: number, cx: number, cy: number): Pt[] {
    if (K === 1) return [{ x: cx, y: cy }];
    if (K === 2) {
      const dx = Math.min(this.W, this.H) * 0.3;
      return [
        { x: cx - dx, y: cy },
        { x: cx + dx, y: cy },
      ];
    }
    // Inset enough that outward self-loops + their labels stay inside the viewBox.
    const R = Math.min(this.W, this.H) * 0.3;
    const pts: Pt[] = [];
    for (let i = 0; i < K; i++) {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / K;
      pts.push({ x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) });
    }
    return pts;
  }

  private forceLayout(K: number): Pt[] {
    interface N extends d3.SimulationNodeDatum {
      id: number;
    }
    const nodes: N[] = d3.range(K).map((id) => ({ id }));
    const links: { source: number; target: number }[] = [];
    const seen = new Set<string>();
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        if (i === j || this.props.P[i][j] <= EPS) continue;
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (seen.has(key)) continue;
        seen.add(key);
        links.push({ source: i, target: j });
      }
    }
    const sim = d3
      .forceSimulation<N>(nodes)
      .force("link", d3.forceLink<N, { source: number; target: number }>(links).id((d) => d.id).distance(120).strength(0.6))
      .force("charge", d3.forceManyBody().strength(-450))
      .force("center", d3.forceCenter(this.W / 2, this.H / 2))
      .force("collide", d3.forceCollide(this.r * 1.8))
      .stop();
    for (let t = 0; t < 320; t++) sim.tick();

    const pad = this.r + 14;
    return nodes.map((n) => ({
      x: Math.max(pad, Math.min(this.W - pad, n.x ?? this.W / 2)),
      y: Math.max(pad, Math.min(this.H - pad, n.y ?? this.H / 2)),
    }));
  }

  private draw(): void {
    const K = this.props.P.length;
    this.svg.selectAll("g.tg-layer, .tg-marker").remove();
    const root = this.svg.append("g").attr("class", "tg-layer");

    // class tint halos behind nodes
    if (this.props.classTint) {
      for (let i = 0; i < K; i++) {
        const tint = this.props.classTint(i);
        if (!tint) continue;
        root
          .append("circle")
          .attr("cx", this.pos[i].x)
          .attr("cy", this.pos[i].y)
          .attr("r", this.r + 10)
          .attr("fill", tint);
      }
    }

    // edges first (under nodes)
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const p = this.props.P[i][j];
        if (p <= EPS) continue;
        if (i === j) this.drawSelfLoop(root, i, p);
        else this.drawEdge(root, i, j, p);
      }
    }

    // nodes
    for (let i = 0; i < K; i++) this.drawNode(root, i);

    // current-state marker
    if (this.props.currentState != null && this.props.currentState >= 0) {
      this.marker = this.makeMarker();
      this.placeMarker(this.props.currentState);
    }
  }

  private strokeWidth(p: number): number {
    return 1 + 5 * Math.min(1, p);
  }

  private trim(a: Pt, b: Pt, byA: number, byB: number): [Pt, Pt] {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    return [
      { x: a.x + ux * byA, y: a.y + uy * byA },
      { x: b.x - ux * byB, y: b.y - uy * byB },
    ];
  }

  private drawEdge(
    root: d3.Selection<SVGGElement, unknown, null, undefined>,
    i: number,
    j: number,
    p: number,
  ): void {
    const [a, b] = this.trim(this.pos[i], this.pos[j], this.r, this.r + 4);
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    // normal (rotate +90°); flips for the reverse edge so a pair separates.
    const nx = -dy / len;
    const ny = dx / len;
    const bidir = this.props.P[j][i] > EPS;
    const curv = bidir ? 26 : 8;
    const cx = mx + nx * curv;
    const cy = my + ny * curv;

    const path = `M${a.x},${a.y} Q${cx},${cy} ${b.x},${b.y}`;
    const strong = p >= 0.5;
    root
      .append("path")
      .attr("d", path)
      .attr("fill", "none")
      .attr("stroke", strong ? "var(--mc-edge-strong)" : "var(--mc-edge)")
      .attr("stroke-width", this.strokeWidth(p))
      .attr("opacity", 0.85)
      .attr("marker-end", "url(#tg-arrow)")
      .append("title")
      .text(`P(${this.label(i)}→${this.label(j)}) = ${p.toFixed(3)}`);

    // label at the bézier midpoint (t=0.5): 0.25 a + 0.5 c + 0.25 b
    const lx = 0.25 * a.x + 0.5 * cx + 0.25 * b.x;
    const ly = 0.25 * a.y + 0.5 * cy + 0.25 * b.y;
    if (p >= 0.05) {
      const g = root.append("g").attr("class", "tg-edge-label");
      const txt = g
        .append("text")
        .attr("x", lx)
        .attr("y", ly)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("class", "annot")
        .style("font-family", "var(--rl-font-mono)")
        .style("font-size", "10px")
        .style("fill", "var(--rl-ink-muted)")
        .text(p.toFixed(2));
      // readability backing
      const bb = (txt.node() as SVGTextElement).getBBox();
      g.insert("rect", "text")
        .attr("x", bb.x - 2)
        .attr("y", bb.y - 1)
        .attr("width", bb.width + 4)
        .attr("height", bb.height + 2)
        .attr("rx", 2)
        .attr("fill", "var(--rl-bg)")
        .attr("opacity", 0.82);
    }
  }

  private drawSelfLoop(
    root: d3.Selection<SVGGElement, unknown, null, undefined>,
    i: number,
    p: number,
  ): void {
    const c = this.pos[i];
    const r = this.r;
    // loop bulging radially outward from the graph centroid, so it never
    // crowds the interior where the inter-node edges live.
    let ox = c.x - this.centroid.x;
    let oy = c.y - this.centroid.y;
    const ol = Math.hypot(ox, oy) || 1;
    ox /= ol;
    oy /= ol;
    const ang = Math.atan2(oy, ox);
    const spread = 0.5;
    const apex = r + 30;
    const start = { x: c.x + r * Math.cos(ang - spread), y: c.y + r * Math.sin(ang - spread) };
    const end = { x: c.x + r * Math.cos(ang + spread), y: c.y + r * Math.sin(ang + spread) };
    const c1 = { x: c.x + apex * Math.cos(ang - spread * 1.1), y: c.y + apex * Math.sin(ang - spread * 1.1) };
    const c2 = { x: c.x + apex * Math.cos(ang + spread * 1.1), y: c.y + apex * Math.sin(ang + spread * 1.1) };
    const labelPt = { x: c.x + (apex + 9) * ox, y: c.y + (apex + 9) * oy };
    const path = `M${start.x},${start.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${end.x},${end.y}`;
    root
      .append("path")
      .attr("d", path)
      .attr("fill", "none")
      .attr("stroke", p >= 0.5 ? "var(--mc-edge-strong)" : "var(--mc-edge)")
      .attr("stroke-width", this.strokeWidth(p))
      .attr("opacity", 0.85)
      .attr("marker-end", "url(#tg-arrow)")
      .append("title")
      .text(`P(${this.label(i)}→${this.label(i)}) = ${p.toFixed(3)}`);

    if (p >= 0.05) {
      root
        .append("text")
        .attr("x", labelPt.x)
        .attr("y", labelPt.y)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("class", "annot")
        .style("font-family", "var(--rl-font-mono)")
        .style("font-size", "10px")
        .style("fill", "var(--rl-ink-muted)")
        .text(p.toFixed(2));
    }
  }

  private drawNode(root: d3.Selection<SVGGElement, unknown, null, undefined>, i: number): void {
    const c = this.pos[i];
    const isCur = this.props.currentState === i;
    const g = root.append("g").attr("transform", `translate(${c.x},${c.y})`);
    g.append("circle")
      .attr("r", this.r)
      .attr("fill", this.color(i))
      .attr("stroke", isCur ? "var(--mc-current)" : "rgba(0,0,0,0.18)")
      .attr("stroke-width", isCur ? 3 : 1.5);
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .style("font-family", "var(--rl-font-ui)")
      .style("font-size", this.label(i).length > 2 ? "10px" : "13px")
      .style("font-weight", "600")
      .style("fill", "#fff")
      .style("pointer-events", "none")
      .text(this.label(i));
  }

  private makeMarker(): d3.Selection<SVGGElement, unknown, null, undefined> {
    const g = this.svg.append("g").attr("class", "tg-marker").style("pointer-events", "none");
    g.append("circle")
      .attr("r", this.r + 6)
      .attr("fill", "none")
      .attr("stroke", "var(--mc-current)")
      .attr("stroke-width", 3)
      .attr("stroke-dasharray", "4 3");
    return g;
  }

  private placeMarker(i: number): void {
    this.marker?.attr("transform", `translate(${this.pos[i].x},${this.pos[i].y})`);
  }

  /** Tween the current-state marker from one node to another (~250ms). */
  animate(from: number, to: number, durationMs = 250): Promise<void> {
    this.props.currentState = to;
    if (!this.marker) {
      this.marker = this.makeMarker();
    }
    if (prefersReducedMotion()) {
      this.placeMarker(to);
      this.draw();
      return Promise.resolve();
    }
    const a = this.pos[from];
    const b = this.pos[to];
    return new Promise((resolve) => {
      this.marker!
        .attr("transform", `translate(${a.x},${a.y})`)
        .transition()
        .duration(durationMs)
        .attrTween("transform", () => {
          const ix = d3.interpolateNumber(a.x, b.x);
          const iy = d3.interpolateNumber(a.y, b.y);
          return (t: number) => `translate(${ix(t)},${iy(t)})`;
        })
        .on("end", () => {
          this.draw();
          resolve();
        });
    });
  }

  destroy(): void {
    this.svg.remove();
    this.tip.remove();
  }
}
