/**
 * GridworldRenderer — the core 3×3 grid drawing, reused by V1, V2, V4, V5, V6,
 * V7. A plain class (like TransitionGraph) so any panel can embed it and drive
 * it programmatically via update(props). Render layers, bottom to top:
 *
 *   1. cell background  — shaded by valueFn (or neutral)
 *   2. action quadrants — four triangles per cell, shaded by qValues (V5)
 *   3. text labels      — V(s) per cell, or Q/A per quadrant
 *   4. policy arrows     — four per cell, opacity ∝ π(a|s), action-colored
 *   5. highlight rings   — selection, backup source, backup inputs
 *   6. terminal glyphs   — pit (−1 ☠) / goal (+1 ⚑), dark borders
 *
 * The build is done once in the constructor; update() patches attributes, with
 * an optional fill tween so the Bellman backup "ripples" rather than snapping.
 */
import * as d3 from "d3";
import {
  ACTION_NAMES,
  GRID_SIZE,
  rc,
  type MDP,
  type Policy,
} from "../mdp/types";
import { makeValueColorScale, textColorOn, fmtV } from "./value-scale";
import { cssVar, prefersReducedMotion } from "./base";

export interface GridworldProps {
  mdp: MDP;
  /** Per-state value for cell shading + label. */
  valueFn?: number[];
  /** Policy for arrow overlays. */
  policy?: Policy;
  /** Per-(state,action) values for quadrant shading (V5). */
  qValues?: number[][];
  /** Render four action quadrants per cell instead of a single fill. */
  showQuadrants?: boolean;
  /** In quadrant mode, treat qValues as advantages (0 → neutral/white). */
  advantageMode?: boolean;
  /** Draw policy arrows (defaults to true when a policy is supplied & !quadrants). */
  showArrows?: boolean;
  /** Print the numeric value inside each cell. */
  showValues?: boolean;
  /** Selected cell (V1) — gets a focus ring. */
  selectedState?: number | null;
  /** Backup source cell (V6) — --mdp-backup-source ring. */
  highlightState?: number | null;
  /** Backup input cells (V6) — --mdp-backup-input rings. */
  highlightInputs?: number[];
  /** Marks the start cell with a small tag. */
  startState?: number | null;
  /** Per-state corner badge text (e.g. optimal-action count); null = none. */
  badges?: (string | null)[];
  colorScale?: (v: number) => string;
  cellPx?: number;
  onCellClick?: (s: number) => void;
  onCellHover?: (s: number | null) => void;
}

const NS = "http://www.w3.org/2000/svg";
const PAD = 16;
const GAP = 8;
const ACTION_VARS = [
  "--mdp-action-up",
  "--mdp-action-right",
  "--mdp-action-down",
  "--mdp-action-left",
];

interface CellRefs {
  bg: d3.Selection<SVGRectElement, unknown, null, undefined>;
  quads: d3.Selection<SVGPolygonElement, unknown, null, undefined>[];
  quadLabels: d3.Selection<SVGTextElement, unknown, null, undefined>[];
  vLabel: d3.Selection<SVGTextElement, unknown, null, undefined>;
  arrows: d3.Selection<SVGGElement, unknown, null, undefined>[];
  ring: d3.Selection<SVGRectElement, unknown, null, undefined>;
  glyph: d3.Selection<SVGTextElement, unknown, null, undefined>;
  tag: d3.Selection<SVGTextElement, unknown, null, undefined>;
  badge: d3.Selection<SVGGElement, unknown, null, undefined>;
  cx: number;
  cy: number;
  x: number;
  y: number;
}

export class GridworldRenderer {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private tip: HTMLDivElement;
  private cells: CellRefs[] = [];
  private props: GridworldProps;
  private cell: number;
  private size: number;

  constructor(container: HTMLElement, props: GridworldProps) {
    this.props = props;
    this.cell = props.cellPx ?? 92;
    const inner = GRID_SIZE * this.cell + (GRID_SIZE - 1) * GAP;
    const size = inner + 2 * PAD;
    this.size = size;

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    wrap.style.position = "relative";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${size} ${size}`);
    svgEl.setAttribute("width", `${size}`);
    svgEl.style.maxWidth = "100%";
    svgEl.style.height = "auto";
    svgEl.classList.add("rl-svg", "mdp-grid");
    wrap.appendChild(svgEl);

    this.tip = document.createElement("div");
    this.tip.className = "rl-tooltip";
    this.tip.style.maxWidth = "230px";
    this.tip.style.whiteSpace = "normal";
    wrap.appendChild(this.tip);
    container.appendChild(wrap);

    this.svg = d3.select(svgEl as SVGSVGElement);
    this.build(props.mdp);
    this.update(props);
  }

  /** Build the static skeleton once: a group per cell with all layers. */
  private build(mdp: MDP): void {
    const c = this.cell;
    for (let s = 0; s < mdp.nS; s++) {
      const { r, c: col } = rc(s);
      const x = PAD + col * (c + GAP);
      const y = PAD + r * (c + GAP);
      const cx = x + c / 2;
      const cy = y + c / 2;
      const g = this.svg.append("g").attr("data-state", s);

      const bg = g
        .append("rect")
        .attr("x", x)
        .attr("y", y)
        .attr("width", c)
        .attr("height", c)
        .attr("rx", 4)
        .attr("class", "mdp-cell-rect")
        .attr("fill", "var(--rl-surface-2)")
        .attr("stroke", "var(--rl-border)")
        .attr("stroke-width", 1)
        .style("cursor", this.props.onCellClick ? "pointer" : "default");

      // four action quadrants (Up/Right/Down/Left), hidden until showQuadrants
      const C: [number, number] = [cx, cy];
      const TL: [number, number] = [x, y];
      const TR: [number, number] = [x + c, y];
      const BR: [number, number] = [x + c, y + c];
      const BL: [number, number] = [x, y + c];
      const tris = [
        [TL, TR, C],
        [TR, BR, C],
        [BR, BL, C],
        [BL, TL, C],
      ];
      const quads = tris.map((pts) =>
        g
          .append("polygon")
          .attr("points", pts.map((p) => p.join(",")).join(" "))
          .attr("stroke", "var(--rl-surface)")
          .attr("stroke-width", 0.75)
          .attr("display", "none"),
      );
      // quadrant label anchors
      const qPos: [number, number][] = [
        [cx, y + c * 0.26],
        [x + c * 0.74, cy],
        [cx, y + c * 0.74],
        [x + c * 0.26, cy],
      ];
      const quadLabels = qPos.map(([qx, qy]) =>
        g
          .append("text")
          .attr("x", qx)
          .attr("y", qy)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("class", "mdp-cell-label")
          .style("font-size", "9px")
          .attr("display", "none"),
      );

      const vLabel = g
        .append("text")
        .attr("x", cx)
        .attr("y", cy)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("class", "mdp-cell-label")
        .style("font-size", "14px");

      const glyph = g
        .append("text")
        .attr("x", x + c - 7)
        .attr("y", y + 13)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .attr("display", "none");

      const tag = g
        .append("text")
        .attr("x", x + 6)
        .attr("y", y + 13)
        .attr("text-anchor", "start")
        .attr("class", "annot")
        .attr("fill", "var(--rl-ink-faint)")
        .style("font-family", "var(--rl-font-ui)")
        .style("font-size", "9px")
        .attr("display", "none");

      // policy arrows (one group each), built but hidden by default
      const arrows = [0, 1, 2, 3].map((a) => {
        const ag = g.append("g").attr("display", "none");
        const L = c * 0.34;
        const dir: [number, number][] = [
          [0, -1],
          [1, 0],
          [0, 1],
          [-1, 0],
        ];
        const [dx, dy] = dir[a];
        const sx = cx + dx * c * 0.1;
        const sy = cy + dy * c * 0.1;
        const tx = cx + dx * L;
        const ty = cy + dy * L;
        ag
          .append("line")
          .attr("x1", sx)
          .attr("y1", sy)
          .attr("x2", tx)
          .attr("y2", ty)
          .attr("stroke-width", 3)
          .attr("stroke-linecap", "round");
        // arrowhead triangle perpendicular to direction
        const hw = 4.5;
        const px = -dy;
        const py = dx;
        const baseX = tx - dx * 5;
        const baseY = ty - dy * 5;
        ag
          .append("polygon")
          .attr(
            "points",
            [
              [tx, ty],
              [baseX + px * hw, baseY + py * hw],
              [baseX - px * hw, baseY - py * hw],
            ]
              .map((p) => p.join(","))
              .join(" "),
          );
        return ag;
      });

      const ring = g
        .append("rect")
        .attr("x", x + 1.5)
        .attr("y", y + 1.5)
        .attr("width", c - 3)
        .attr("height", c - 3)
        .attr("rx", 4)
        .attr("fill", "none")
        .attr("stroke-width", 3)
        .attr("display", "none")
        .style("pointer-events", "none");

      // optimal-action-count badge (bottom-left corner), hidden by default
      const badge = g.append("g").attr("display", "none");
      badge
        .append("rect")
        .attr("x", x + 5)
        .attr("y", y + c - 21)
        .attr("width", 16)
        .attr("height", 16)
        .attr("rx", 5)
        .attr("fill", "var(--mdp-backup-source)");
      badge
        .append("text")
        .attr("x", x + 13)
        .attr("y", y + c - 13)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", "#fff")
        .style("font-family", "var(--rl-font-mono)")
        .style("font-size", "10px")
        .style("font-weight", "600");

      // hit area on top for hover/click
      g
        .append("rect")
        .attr("x", x)
        .attr("y", y)
        .attr("width", c)
        .attr("height", c)
        .attr("fill", "transparent")
        .style("cursor", this.props.onCellClick ? "pointer" : "default")
        .on("click", () => this.props.onCellClick?.(s))
        .on("mousemove", (ev: MouseEvent) => this.showTip(ev, s))
        .on("mouseleave", () => {
          this.tip.style.opacity = "0";
          this.props.onCellHover?.(null);
        });

      this.cells.push({ bg, quads, quadLabels, vLabel, arrows, ring, glyph, tag, badge, cx, cy, x, y });
    }
  }

  /** Patch the skeleton to reflect new props. */
  update(props: Partial<GridworldProps>, opts: { animate?: boolean } = {}): void {
    this.props = { ...this.props, ...props };
    const p = this.props;
    const mdp = p.mdp;
    const scale = p.colorScale ?? makeValueColorScale();
    const actionColors = ACTION_VARS.map((v) => cssVar(v) || "#666");
    const animate = opts.animate && !prefersReducedMotion();

    for (let s = 0; s < mdp.nS; s++) {
      const cell = this.cells[s];
      const terminal = mdp.terminals[s];

      // --- background fill ---
      let bgFill = "var(--rl-surface-2)";
      if (p.showQuadrants && p.qValues) {
        bgFill = "var(--rl-surface)";
      } else if (terminal) {
        // pit / goal tint by their entry reward sign (read from any predecessor)
        bgFill = this.terminalTint(mdp, s);
      } else if (p.valueFn) {
        bgFill = scale(p.valueFn[s]);
      }
      const bgSel = animate ? cell.bg.transition().duration(180) : cell.bg;
      bgSel.attr("fill", bgFill);
      cell.bg.attr(
        "stroke",
        terminal ? "var(--mdp-terminal)" : "var(--rl-border)",
      );
      cell.bg.attr("stroke-width", terminal ? 2 : 1);

      // --- quadrants ---
      const showQ = !!(p.showQuadrants && p.qValues) && !terminal;
      cell.quads.forEach((q, a) => {
        if (!showQ) {
          q.attr("display", "none");
          return;
        }
        const val = p.qValues![s][a];
        q.attr("display", null);
        const sel = animate ? q.transition().duration(180) : q;
        sel.attr("fill", scale(val));
      });
      cell.quadLabels.forEach((lab, a) => {
        if (!showQ) {
          lab.attr("display", "none");
          return;
        }
        const val = p.qValues![s][a];
        lab
          .attr("display", null)
          .attr("fill", textColorOn(scale(val)))
          .text(val.toFixed(2));
      });

      // --- center value label ---
      const showVal = p.showValues ?? !!p.valueFn;
      if (showQ) {
        cell.vLabel.attr("display", "none");
      } else if (terminal) {
        cell.vLabel
          .attr("display", null)
          .attr("fill", textColorOn(bgFill))
          .style("font-size", "13px")
          .text(this.terminalReward(mdp, s) > 0 ? "+1" : "−1");
      } else if (showVal && p.valueFn) {
        cell.vLabel
          .attr("display", null)
          .attr("fill", textColorOn(bgFill))
          .style("font-size", "14px")
          .text(fmtV(p.valueFn[s]));
      } else {
        cell.vLabel.attr("display", "none");
      }

      // --- terminal glyph ---
      if (terminal) {
        const goal = this.terminalReward(mdp, s) > 0;
        cell.glyph
          .attr("display", null)
          .attr("fill", goal ? "var(--mdp-reward-pos)" : "var(--mdp-reward-neg)")
          .text(goal ? "⚑" : "☠");
      } else {
        cell.glyph.attr("display", "none");
      }

      // --- start tag ---
      if (p.startState === s) cell.tag.attr("display", null).text("start");
      else cell.tag.attr("display", "none");

      // --- corner badge ---
      const badgeText = p.badges?.[s] ?? null;
      if (badgeText != null && !terminal) {
        cell.badge.attr("display", null);
        cell.badge.select("text").text(badgeText);
      } else {
        cell.badge.attr("display", "none");
      }

      // --- arrows ---
      const showArrows =
        (p.showArrows ?? (!!p.policy && !p.showQuadrants)) && !terminal && !!p.policy;
      cell.arrows.forEach((ag, a) => {
        if (!showArrows) {
          ag.attr("display", "none");
          return;
        }
        const w = p.policy!.pi[s][a];
        if (w < 1e-4) {
          ag.attr("display", "none");
          return;
        }
        ag.attr("display", null).attr("opacity", 0.25 + 0.75 * w);
        ag.selectAll("line, polygon").attr("stroke", actionColors[a]).attr("fill", actionColors[a]);
      });

      // --- highlight ring ---
      let ringColor: string | null = null;
      if (p.highlightState === s) ringColor = "var(--mdp-backup-source)";
      else if (p.highlightInputs?.includes(s)) ringColor = "var(--mdp-backup-input)";
      else if (p.selectedState === s) ringColor = "var(--rl-algo-ucb)";
      if (ringColor) cell.ring.attr("display", null).attr("stroke", ringColor);
      else cell.ring.attr("display", "none");
    }
  }

  /** Entry reward of a terminal state, inferred from a predecessor that enters it. */
  private terminalReward(mdp: MDP, s: number): number {
    for (let ss = 0; ss < mdp.nS; ss++) {
      if (mdp.terminals[ss]) continue;
      for (let a = 0; a < mdp.nA; a++) {
        if (mdp.P[ss][a][s] > 0) return mdp.r[ss][a] / mdp.P[ss][a][s];
      }
    }
    return 0;
  }

  private terminalTint(mdp: MDP, s: number): string {
    return this.terminalReward(mdp, s) > 0
      ? "color-mix(in srgb, var(--mdp-reward-pos) 22%, var(--rl-surface))"
      : "color-mix(in srgb, var(--mdp-reward-neg) 22%, var(--rl-surface))";
  }

  /** Which action-quadrant the cursor sits in, from its offset to the cell center. */
  private quadrantAt(ev: MouseEvent, s: number): number {
    const rect = (this.svg.node() as SVGSVGElement).getBoundingClientRect();
    const ratio = this.size / rect.width;
    const dx = (ev.clientX - rect.left) * ratio - this.cells[s].cx;
    const dy = (ev.clientY - rect.top) * ratio - this.cells[s].cy;
    if (Math.abs(dy) >= Math.abs(dx)) return dy < 0 ? 0 : 2; // Up / Down
    return dx > 0 ? 1 : 3; // Right / Left
  }

  private showTip(ev: MouseEvent, s: number): void {
    const p = this.props;
    const mdp = p.mdp;
    const { r, c } = rc(s);
    let html = `<strong>state (${r},${c})</strong>`;
    if (mdp.terminals[s]) {
      html += `<br>terminal · V = 0`;
    } else if (p.showQuadrants && p.qValues) {
      // per-quadrant detail: the action under the cursor
      const a = this.quadrantAt(ev, s);
      const label = p.advantageMode ? "A" : "Q";
      html += `<br><strong style="color:var(--rl-ink)">${ACTION_NAMES[a]}</strong> · ${label}(s,a) = ${p.qValues[s][a].toFixed(3)}`;
      const dist = mdp.P[s][a]
        .map((pr, sp) => (pr > 0 ? `(${rc(sp).r},${rc(sp).c}) ${pr.toFixed(2)}` : null))
        .filter(Boolean)
        .join(", ");
      html += `<br><span style="color:var(--rl-ink-faint)">→ ${dist}</span>`;
    } else {
      if (p.valueFn) html += `<br>V = ${fmtV(p.valueFn[s])}`;
      if (p.qValues) {
        html += `<br><span style="color:var(--rl-ink-faint)">Q(s,·):</span>`;
        for (let a = 0; a < mdp.nA; a++)
          html += `<br>&nbsp;${ACTION_NAMES[a]}: ${p.qValues[s][a].toFixed(3)}`;
      }
      if (p.policy) {
        const parts = p.policy.pi[s]
          .map((w, a) => (w > 1e-3 ? `${ACTION_NAMES[a][0]} ${w.toFixed(2)}` : null))
          .filter(Boolean)
          .join(", ");
        html += `<br><span style="color:var(--rl-ink-faint)">π:</span> ${parts}`;
      }
    }
    this.tip.innerHTML = html;
    this.tip.style.opacity = "1";
    const rect = (this.svg.node() as SVGSVGElement).getBoundingClientRect();
    this.tip.style.left = `${ev.clientX - rect.left + 12}px`;
    this.tip.style.top = `${ev.clientY - rect.top + 12}px`;
    this.props.onCellHover?.(s);
  }
}
