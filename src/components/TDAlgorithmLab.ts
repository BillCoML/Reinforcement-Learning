/**
 * V3 — TD Algorithm Lab. CENTERPIECE (arena, ~1000px).
 * Four algorithms side-by-side: TD(0), SARSA, Q-learning, n-step TD(4).
 * All share a master seed. "Advance N episodes" button runs all four in sync.
 * Per-algorithm reference lines, live value heatmaps and learning curves.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { mulberry32 } from "../importance-sampling/gaussian";
import { tdZero } from "../td/td-zero";
import { sarsa } from "../td/sarsa";
import { qLearning } from "../td/q-learning";
import { nStepTD } from "../td/n-step-td";

const GAMMA = 0.9;
const mdp = buildGridworld({ slippery: false, gamma: GAMMA });
const uniform = uniformPolicy(mdp);

const CELL = 46, GAP = 3;
const TRUE_V0 = -0.4205;   // V^π(s₀) under uniform
const TRUE_Q_STAR = 0.729; // V*(s₀) under optimal (SARSA/Q-learning target)

interface AlgoState {
  name: string;
  color: string;
  /** null ⟹ not yet run */
  V: Float64Array | null;
  history: Float64Array | null;
  episodesRun: number;
}

function valueColor(v: number): string {
  if (v >= 0) {
    const t = Math.min(v, 1);
    return `rgb(${Math.round(235 - t * 80)},${Math.round(235)},${Math.round(235 - t * 80)})`;
  }
  const t = Math.min(Math.abs(v), 1);
  return `rgb(${Math.round(255)},${Math.round(220 - t * 120)},${Math.round(220 - t * 120)})`;
}

class TDAlgorithmLab extends HTMLElement {
  private seed = 0;
  private totalEpisodes = 0;
  private chunk = 500;
  private algos: AlgoState[] = [
    { name: "TD(0)", color: "var(--td-td0)", V: null, history: null, episodesRun: 0 },
    { name: "SARSA", color: "var(--td-sarsa)", V: null, history: null, episodesRun: 0 },
    { name: "Q-learning", color: "var(--td-qlearning)", V: null, history: null, episodesRun: 0 },
    { name: "n-step (n=4)", color: "var(--td-nstep)", V: null, history: null, episodesRun: 0 },
  ];

  connectedCallback() { this.build(); }

  private runChunk() {
    const N = this.chunk;
    const seed = this.seed;
    const eps = this.totalEpisodes;

    // Each algo gets the same seed offset so results are comparable
    const r0 = tdZero(mdp, uniform, N, 0.1, { rng: mulberry32(seed * 100) });
    const r1 = sarsa(mdp, N, 0.1, 0.1, { rng: mulberry32(seed * 100 + 1) });
    const r2 = qLearning(mdp, N, 0.1, 0.1, { rng: mulberry32(seed * 100 + 2) });
    const r3 = nStepTD(mdp, uniform, 4, N, 0.1, { rng: mulberry32(seed * 100 + 3) });

    // Q→V for SARSA and Q-learning (V(s) = max_a Q(s,a))
    const nA = mdp.nA;
    const qToV = (Q: Float64Array) => {
      const V = new Float64Array(mdp.nS);
      for (let s = 0; s < mdp.nS; s++) {
        let mx = -Infinity;
        for (let a = 0; a < nA; a++) mx = Math.max(mx, Q[s * nA + a]);
        V[s] = mx;
      }
      return V;
    };

    this.algos[0] = { ...this.algos[0], V: r0.V, history: r0.history, episodesRun: eps + N };
    this.algos[1] = { ...this.algos[1], V: qToV(r1.Q), history: r1.history, episodesRun: eps + N };
    this.algos[2] = { ...this.algos[2], V: qToV(r2.Q), history: r2.history, episodesRun: eps + N };
    this.algos[3] = { ...this.algos[3], V: r3.V, history: r3.history, episodesRun: eps + N };
    this.totalEpisodes += N;
    this.seed++;
  }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "td-algorithm-lab", arena: true, heavy: true });

    const chunks = [100, 500, 1000, 5000];
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label">Episodes per advance:
        <select id="tal-chunk" class="rl-select">
          ${chunks.map((c, i) => `<option value="${i}" ${i === 1 ? "selected" : ""}>${c}</option>`).join("")}
        </select>
      </label>
      <button class="rl-btn" id="tal-advance">▶ Advance N Episodes</button>
      <button class="rl-btn" id="tal-reset">↺ Reset</button>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const W = 1000, H = 480;
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);

    const render = () => {
      svg.selectAll("*").remove();
      setStatus(`episodes: ${this.totalEpisodes}`);
      this.draw(svg, W, H);
    };

    panel.querySelector("#tal-chunk")!.addEventListener("change", (e) => {
      this.chunk = chunks[+(e.target as HTMLSelectElement).value];
    });
    panel.querySelector("#tal-advance")!.addEventListener("click", () => {
      this.runChunk();
      render();
    });
    panel.querySelector("#tal-reset")!.addEventListener("click", () => {
      this.seed = 0;
      this.totalEpisodes = 0;
      this.algos.forEach(a => { a.V = null; a.history = null; a.episodesRun = 0; });
      render();
    });

    render();
  }

  private draw(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, W: number, H: number) {
    const colW = W / 4;
    const gridSize = GRID_SIZE();
    const heatH = gridSize + 30;
    const curveH = H - heatH - 60;
    const PAD = { top: 24, bottom: 16, left: 32, right: 8 };

    this.algos.forEach((algo, col) => {
      const ox = col * colW;

      // Column label
      svg.append("text").attr("x", ox + colW / 2).attr("y", 16)
        .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 700)
        .attr("fill", algo.color).text(algo.name);
      if (algo.episodesRun > 0) {
        svg.append("text").attr("x", ox + colW / 2).attr("y", 28)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8")
          .text(`N=${algo.episodesRun}`);
      }

      // Heatmap
      const hmX = ox + (colW - gridSize) / 2;
      const hmY = 34;
      for (let s = 0; s < 9; s++) {
        const row = Math.floor(s / 3), c = s % 3;
        const cx = hmX + c * (CELL + GAP), cy = hmY + row * (CELL + GAP);
        const v = algo.V ? algo.V[s] : 0;
        const isTerminal = mdp.terminals[s];
        svg.append("rect").attr("x", cx).attr("y", cy).attr("width", CELL).attr("height", CELL)
          .attr("fill", algo.V ? (isTerminal ? (v >= 0 ? "#bbf7d0" : "#fca5a5") : valueColor(v)) : "#f8fafc")
          .attr("rx", 3).attr("stroke", "#e2e8f0").attr("stroke-width", 1);
        if (algo.V) {
          svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 4)
            .attr("text-anchor", "middle").attr("font-size", 10).attr("font-weight", 600)
            .attr("fill", "#1e293b")
            .text(isTerminal ? (s === 4 ? "pit" : "★") : v.toFixed(2));
        } else {
          svg.append("text").attr("x", cx + CELL / 2).attr("y", cy + CELL / 2 + 4)
            .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#cbd5e1").text("—");
        }
      }

      // Learning curve
      const curveX = ox + PAD.left, curveY = heatH + 16;
      const cW = colW - PAD.left - PAD.right - 4;
      const cH = curveH - PAD.top - PAD.bottom;

      svg.append("line").attr("x1", curveX).attr("x2", curveX + cW)
        .attr("y1", curveY + PAD.top + cH).attr("y2", curveY + PAD.top + cH)
        .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

      // Reference lines
      const refVal = col <= 1 ? TRUE_V0 : TRUE_Q_STAR;
      const refLabel = col <= 1 ? "V^π" : "V*";

      if (algo.history) {
        const hist = algo.history;
        const yMin = Math.min(refVal - 0.15, ...Array.from(hist).slice(0, Math.min(20, hist.length)));
        const yMax = Math.max(refVal + 0.1, ...Array.from(hist).slice(0, Math.min(20, hist.length)));
        const xS = d3.scaleLinear([0, hist.length - 1], [curveX, curveX + cW]);
        const yS = d3.scaleLinear([yMin, yMax + (yMax - yMin) * 0.1], [curveY + PAD.top + cH, curveY + PAD.top]);

        // Reference line
        svg.append("line").attr("x1", curveX).attr("x2", curveX + cW)
          .attr("y1", yS(refVal)).attr("y2", yS(refVal))
          .attr("stroke", "#94a3b8").attr("stroke-width", 1).attr("stroke-dasharray", "3,2");
        svg.append("text").attr("x", curveX + cW - 2).attr("y", yS(refVal) - 2)
          .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8").text(refLabel);

        // Y ticks
        for (const t of yS.ticks(3)) {
          svg.append("text").attr("x", curveX - 2).attr("y", yS(t) + 3)
            .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8")
            .text(t.toFixed(1));
        }

        // Curve (decimated)
        const stride = Math.max(1, Math.floor(hist.length / 150));
        const pts: [number, number][] = [];
        for (let i = 0; i < hist.length; i += stride) pts.push([i, hist[i]]);

        svg.append("path")
          .datum(pts)
          .attr("d", d3.line<[number, number]>().x(p => xS(p[0])).y(p => yS(p[1])).curve(d3.curveMonotoneX))
          .attr("fill", "none").attr("stroke", algo.color).attr("stroke-width", 1.5);
      } else {
        svg.append("text").attr("x", curveX + cW / 2).attr("y", curveY + PAD.top + cH / 2)
          .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#cbd5e1")
          .text("press Advance →");
      }

      // Vertical divider
      if (col < 3) {
        svg.append("line").attr("x1", (col + 1) * colW - 1).attr("x2", (col + 1) * colW - 1)
          .attr("y1", 0).attr("y2", H)
          .attr("stroke", "#f1f5f9").attr("stroke-width", 1);
      }
    });

    // Bottom label
    svg.append("text").attr("x", W / 2).attr("y", H - 4)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8")
      .text("TD(0)/n-step track V^π(s₀)≈−0.42; SARSA/Q-learning track V^π resp. V*(s₀)≈0.73");
  }
}

function GRID_SIZE() { return GRID * (CELL + GAP) - GAP; }
const GRID = 3;

customElements.define("td-algorithm-lab", TDAlgorithmLab);
