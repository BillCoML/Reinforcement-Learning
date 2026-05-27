/**
 * V4 — SARSA vs Q-learning.
 * Side-by-side panels: left=SARSA (on-policy), right=Q-learning (off-policy).
 * Each panel: Q(s,·) bar chart for current state, greedy policy arrows on
 * gridworld, and a learning-curve chart. Dual reference lines at 0.6274 / 0.729.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld } from "../mdp/gridworld";
import { mulberry32 } from "../importance-sampling/gaussian";
import { sarsa } from "../td/sarsa";
import { qLearning } from "../td/q-learning";

const GAMMA = 0.9;
const mdp = buildGridworld({ slippery: false, gamma: GAMMA });
const nA = mdp.nA, nS = mdp.nS;

const SARSA_TARGET  = 0.6274;
const QL_TARGET     = 0.729;
const ACTION_NAMES  = ["↑", "→", "↓", "←"];
const CELL = 56, GAP = 4;

function argmax(Q: Float64Array, s: number): number {
  let best = 0, bestVal = Q[s * nA];
  for (let a = 1; a < nA; a++) if (Q[s * nA + a] > bestVal) { bestVal = Q[s * nA + a]; best = a; }
  return best;
}

class SARSAvsQLearning extends HTMLElement {
  private episodesRun = 0;
  private chunk = 500;
  private seed = 0;
  private sarsaQ: Float64Array = new Float64Array(nS * nA);
  private qlQ: Float64Array = new Float64Array(nS * nA);
  private sarsaHist: number[] = [];
  private qlHist: number[] = [];
  private hoveredState = 0;

  connectedCallback() { this.build(); }

  private advance() {
    const N = this.chunk;
    const r1 = sarsa(mdp, N, 0.1, 0.1, { rng: mulberry32(this.seed * 50) });
    const r2 = qLearning(mdp, N, 0.1, 0.1, { rng: mulberry32(this.seed * 50 + 7) });
    this.sarsaQ = r1.Q;
    this.qlQ = r2.Q;
    this.sarsaHist.push(...Array.from(r1.history));
    this.qlHist.push(...Array.from(r2.history));
    this.episodesRun += N;
    this.seed++;
  }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "sarsa-vs-qlearning", heavy: true });

    const chunks = [100, 500, 1000, 5000];
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label">Episodes:
        <select id="svq-chunk" class="rl-select">
          ${chunks.map((c, i) => `<option value="${i}" ${i === 1 ? "selected" : ""}>${c}</option>`).join("")}
        </select>
      </label>
      <button class="rl-btn" id="svq-advance">▶ Advance</button>
      <button class="rl-btn" id="svq-reset">↺ Reset</button>
      <span id="svq-hover-tip" style="margin-left:12px;font-size:11px;color:var(--rl-ink-muted)">Hover grid to inspect Q(s,·)</span>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const W = 920, H = 480;
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    svgEl.style.cursor = "default";
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);

    const render = () => {
      svg.selectAll("*").remove();
      setStatus(`episodes: ${this.episodesRun}`);
      this.draw(svg, W, H);
    };

    // Mouse hover: detect which state the cursor is near in the gridworld cells
    svgEl.addEventListener("mousemove", (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const mx = (ev.clientX - rect.left) * (W / rect.width);
      const my = (ev.clientY - rect.top) * (H / rect.height);
      const s = this.hitTestGrid(mx, my, W);
      if (s >= 0 && s !== this.hoveredState) {
        this.hoveredState = s;
        render();
      }
    });

    panel.querySelector("#svq-chunk")!.addEventListener("change", (e) => {
      this.chunk = chunks[+(e.target as HTMLSelectElement).value];
    });
    panel.querySelector("#svq-advance")!.addEventListener("click", () => {
      this.advance(); render();
    });
    panel.querySelector("#svq-reset")!.addEventListener("click", () => {
      this.episodesRun = 0; this.seed = 0;
      this.sarsaQ = new Float64Array(nS * nA);
      this.qlQ = new Float64Array(nS * nA);
      this.sarsaHist = []; this.qlHist = [];
      render();
    });

    render();
  }

  private hitTestGrid(mx: number, _my: number, W: number): number {
    const halfW = W / 2;
    // Left panel grid starts at gridOX(0), right panel at gridOX(1)
    for (let panel = 0; panel < 2; panel++) {
      const ox = panel * halfW + (halfW - (3 * (CELL + GAP) - GAP)) / 2;
      const oy = 34;
      for (let s = 0; s < 9; s++) {
        const row = Math.floor(s / 3), col = s % 3;
        const cx = ox + col * (CELL + GAP);
        void (oy + row * (CELL + GAP)); // cy not needed for x-only hit test
        if (mx >= cx && mx <= cx + CELL) return s;
      }
    }
    return -1;
  }

  private draw(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, W: number, H: number) {
    const halfW = W / 2;
    const gs = 3 * (CELL + GAP) - GAP;

    // Vertical divider
    svg.append("line").attr("x1", halfW).attr("x2", halfW).attr("y1", 0).attr("y2", H)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    const panels = [
      { label: "SARSA (on-policy)", color: "var(--td-sarsa)", Q: this.sarsaQ, hist: this.sarsaHist, target: SARSA_TARGET, refLabel: "V^π≈0.627" },
      { label: "Q-learning (off-policy)", color: "var(--td-qlearning)", Q: this.qlQ, hist: this.qlHist, target: QL_TARGET, refLabel: "V*≈0.729" },
    ];

    panels.forEach(({ label, color, Q, hist, target, refLabel }, pi) => {
      const ox = pi * halfW;
      const gridOX = ox + (halfW - gs) / 2;
      const gridOY = 34;

      svg.append("text").attr("x", ox + halfW / 2).attr("y", 16)
        .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 700)
        .attr("fill", color).text(label);

      // Gridworld + greedy arrows
      for (let s = 0; s < 9; s++) {
        const row = Math.floor(s / 3), col = s % 3;
        const cx = gridOX + col * (CELL + GAP), cy = gridOY + row * (CELL + GAP);
        const isTerminal = mdp.terminals[s];
        const isHov = s === this.hoveredState;

        svg.append("rect").attr("x", cx).attr("y", cy).attr("width", CELL).attr("height", CELL)
          .attr("fill", isHov ? "#f0f9ff" : "#f8fafc")
          .attr("rx", 4).attr("stroke", isHov ? color : "#e2e8f0").attr("stroke-width", isHov ? 2 : 1);

        if (isTerminal) {
          svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 4)
            .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#64748b")
            .text(s === 4 ? "pit" : "★");
        } else {
          const best = argmax(Q, s);
          const bv = Q[s * nA + best];
          // Arrow for greedy action
          const ax = cx + CELL / 2, ay = cy + CELL / 2;
          svg.append("text").attr("x", ax).attr("y", ay + 4)
            .attr("text-anchor", "middle").attr("font-size", 16)
            .attr("fill", this.episodesRun > 0 ? color : "#cbd5e1")
            .text(ACTION_NAMES[best]);
          if (this.episodesRun > 0) {
            svg.append("text").attr("x", ax).attr("y", cy + CELL - 4)
              .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#94a3b8")
              .text(bv.toFixed(2));
          }
        }
      }

      // Q(s,·) bar chart for hovered state
      const barY = gridOY + gs + 12;
      const barW = gs, barH = 80;
      const s = this.hoveredState;

      svg.append("text").attr("x", gridOX + gs / 2).attr("y", barY - 2)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#475569")
        .text(`Q(s${s}, ·) — hover grid state to change`);

      const qVals = Array.from({ length: nA }, (_, a) => Q[s * nA + a]);
      const qMax = Math.max(...qVals, 0.01);
      const qMin = Math.min(...qVals, -0.01);
      const bw = barW / nA - 4;

      const yQ = d3.scaleLinear([qMin * 1.1, qMax * 1.1], [barY + barH, barY]);
      svg.append("line").attr("x1", gridOX).attr("x2", gridOX + barW)
        .attr("y1", yQ(0)).attr("y2", yQ(0)).attr("stroke", "#e2e8f0").attr("stroke-width", 1);

      qVals.forEach((v, a) => {
        const bx = gridOX + a * (bw + 4) + 2;
        const isBest = a === argmax(Q, s);
        svg.append("rect")
          .attr("x", bx).attr("y", v >= 0 ? yQ(v) : yQ(0))
          .attr("width", bw).attr("height", Math.abs(yQ(v) - yQ(0)))
          .attr("fill", isBest ? color : "#94a3b8").attr("rx", 2);
        svg.append("text").attr("x", bx + bw / 2).attr("y", barY + barH + 12)
          .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#475569")
          .text(ACTION_NAMES[a]);
        if (this.episodesRun > 0) {
          svg.append("text").attr("x", bx + bw / 2).attr("y", (v >= 0 ? yQ(v) : yQ(0)) - 2)
            .attr("text-anchor", "middle").attr("font-size", 8)
            .attr("fill", isBest ? color : "#94a3b8").text(v.toFixed(2));
        }
      });

      // Learning curve
      const curveY = barY + barH + 30;
      const curveH = H - curveY - 16;
      const marginL = 36, marginR = 8;
      const cW = halfW - marginL - marginR - 8;

      svg.append("line").attr("x1", ox + marginL).attr("x2", ox + marginL + cW)
        .attr("y1", curveY + curveH).attr("y2", curveY + curveH)
        .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

      if (hist.length > 1) {
        const yMin = Math.min(target - 0.2, ...hist.slice(0, 10));
        const yMax = Math.max(target + 0.05, ...hist.slice(0, 10));
        const xS = d3.scaleLinear([0, hist.length - 1], [ox + marginL, ox + marginL + cW]);
        const yS = d3.scaleLinear([yMin, yMax], [curveY + curveH, curveY]);

        // Reference line
        svg.append("line").attr("x1", ox + marginL).attr("x2", ox + marginL + cW)
          .attr("y1", yS(target)).attr("y2", yS(target))
          .attr("stroke", "#94a3b8").attr("stroke-width", 1).attr("stroke-dasharray", "3,2");
        svg.append("text").attr("x", ox + marginL + cW - 2).attr("y", yS(target) - 2)
          .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8").text(refLabel);

        // Y ticks
        for (const t of yS.ticks(3)) {
          svg.append("text").attr("x", ox + marginL - 3).attr("y", yS(t) + 3)
            .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8")
            .text(t.toFixed(1));
        }

        const stride = Math.max(1, Math.floor(hist.length / 150));
        const pts: [number, number][] = [];
        for (let i = 0; i < hist.length; i += stride) pts.push([i, hist[i]]);

        svg.append("path").datum(pts)
          .attr("d", d3.line<[number, number]>().x(p => xS(p[0])).y(p => yS(p[1])).curve(d3.curveMonotoneX))
          .attr("fill", "none").attr("stroke", color).attr("stroke-width", 1.5);
      } else {
        svg.append("text").attr("x", ox + marginL + cW / 2).attr("y", curveY + curveH / 2)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#cbd5e1")
          .text("press Advance →");
      }
    });
  }
}

customElements.define("sarsa-vs-qlearning", SARSAvsQLearning);
