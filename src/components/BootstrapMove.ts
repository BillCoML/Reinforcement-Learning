/**
 * V1 — The Bootstrap Move.
 * Two-panel animation: MC (left) waits for the full return; TD (right) substitutes
 * V̂(s') at the first step and produces its target immediately.
 * Width 880, height 380.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { mulberry32 } from "../importance-sampling/gaussian";


const W = 880, H = 380;
const PAD = { left: 40, right: 40, top: 48, bottom: 32 };
const PW = (W - PAD.left - PAD.right - 40) / 2; // width of each panel
const PH = H - PAD.top - PAD.bottom;
// Colors matching spec
const COL_BOOT = "var(--td-bootstrap)"; // bootstrap value (cyan)
const COL_TRAJ = "#64748b"; // trajectory arrow
const COL_ACTIVE = "#1e293b";

// Simple 3×3 gridworld for trajectory sampling
const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const uniform = uniformPolicy(mdp);
const GAMMA = 0.9;
const START = 0;

function sampleFrom(row: number[], rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < row.length; i++) {
    acc += row[i];
    if (u <= acc) return i;
  }
  return row.length - 1;
}

interface Step { s: number; a: number; r: number; sp: number; done: boolean; }

function sampleTrajectory(seed: number, maxLen = 12): Step[] {
  const rng = mulberry32(seed);
  const steps: Step[] = [];
  let s = START;
  for (let i = 0; i < maxLen; i++) {
    if (mdp.terminals[s]) break;
    const a = sampleFrom(uniform.pi[s], rng);
    const sp = sampleFrom(mdp.P[s][a], rng);
    const r = mdp.r[s][a];
    const done = mdp.terminals[sp];
    steps.push({ s, a, r, sp, done });
    if (done) break;
    s = sp;
  }
  return steps;
}


class BootstrapMove extends HTMLElement {
  private timer: number | null = null;
  private frame = 0;
  private steps: Step[] = [];
  private seed = 7;

  connectedCallback() { this.build(); }
  disconnectedCallback() { this.stop(); }

  private stop() {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({
      id: "bootstrap-move",
      heavy: false,
    });

    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <button class="rl-btn" id="bm-play">▶ Play</button>
      <button class="rl-btn" id="bm-reset">↺ New trajectory</button>
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);

    const newTraj = () => {
      this.stop();
      this.frame = 0;
      this.steps = sampleTrajectory(this.seed++);
      this.render(svg, this.steps, this.frame, setStatus);
      playBtn.textContent = "▶ Play";
    };

    const playBtn = panel.querySelector("#bm-play") as HTMLButtonElement;
    panel.querySelector("#bm-reset")!.addEventListener("click", newTraj);
    playBtn.addEventListener("click", () => {
      if (this.timer !== null) {
        this.stop();
        playBtn.textContent = "▶ Play";
      } else {
        playBtn.textContent = "⏸ Pause";
        this.timer = window.setInterval(() => {
          this.frame = (this.frame + 1) % (this.steps.length + 2);
          if (this.frame === 0) playBtn.textContent = "▶ Play";
          this.render(svg, this.steps, this.frame, setStatus);
          if (this.frame >= this.steps.length + 1) { this.stop(); playBtn.textContent = "▶ Play"; }
        }, 900);
      }
    });

    newTraj();
  }

  private render(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    steps: Step[],
    frame: number,
    setStatus: (t: string) => void,
  ) {
    svg.selectAll("*").remove();
    const T = steps.length;
    setStatus(`step ${frame} / ${T}`);

    // Panel labels
    svg.append("text").attr("x", PAD.left + PW / 2).attr("y", 22)
      .attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", "var(--td-mc)").text("Monte Carlo — waits for episode end");
    svg.append("text").attr("x", PAD.left + PW + 40 + PW / 2).attr("y", 22)
      .attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", COL_BOOT).text("TD — bootstraps at step t+1");

    // Vertical divider
    svg.append("line")
      .attr("x1", PAD.left + PW + 20).attr("x2", PAD.left + PW + 20)
      .attr("y1", 28).attr("y2", H - 16)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    this.drawPanel(svg, steps, frame, "mc", PAD.left, PAD.top);
    this.drawPanel(svg, steps, frame, "td", PAD.left + PW + 40, PAD.top);
  }

  private drawPanel(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    steps: Step[],
    frame: number,
    phase: "mc" | "td",
    ox: number,
    oy: number,
  ) {
    const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
    const T = steps.length;
    const freezeAt = Math.min(frame - 1, T - 1); // step index being highlighted

    // Draw trajectory as dots + arrows (vertical stack)
    const dotY = (i: number) => i * 32;
    const dotX = 16;
    const infoX = dotX + 28;

    for (let i = 0; i <= Math.min(frame, T); i++) {
      const y = dotY(i);
      const isActive = (i === frame && frame < T) || (i === T - 1 && frame >= T);
      const isFrozen = phase === "td" && i === freezeAt + 1 && frame > 0;

      // State node
      g.append("circle")
        .attr("cx", dotX).attr("cy", y)
        .attr("r", 8)
        .attr("fill", isActive ? COL_ACTIVE : (i < frame ? "#94a3b8" : "#e2e8f0"))
        .attr("stroke", "white").attr("stroke-width", 1.5);

      g.append("text").attr("x", dotX).attr("y", y + 4)
        .attr("text-anchor", "middle").attr("font-size", 9)
        .attr("fill", "white").text(`s${i}`);

      if (i < Math.min(frame + 1, T)) {
        const step = steps[i];
        const gk = Math.pow(GAMMA, i);
        g.append("text").attr("x", infoX).attr("y", y - 2)
          .attr("font-size", 11).attr("fill", i < frame ? "#64748b" : COL_ACTIVE)
          .text(`r=${step.r >= 0 ? "+" : ""}${step.r.toFixed(0)}, γ${i > 0 ? `^${i}` : ""}=${gk.toFixed(2)}`);

        // Arrow to next state
        if (i < T - 1 || !step.done) {
          g.append("line")
            .attr("x1", dotX).attr("y1", y + 9)
            .attr("x2", dotX).attr("y2", dotY(i + 1) - 9)
            .attr("stroke", COL_TRAJ).attr("stroke-width", 1.5)
            .attr("marker-end", "url(#arrow)");
        }
      }

      // TD bootstrap freeze highlight
      if (isFrozen) {
        g.append("rect")
          .attr("x", -4).attr("y", y - 12).attr("width", PW - 4).attr("height", 22)
          .attr("fill", "none").attr("stroke", COL_BOOT).attr("stroke-width", 2)
          .attr("rx", 4);
        g.append("text").attr("x", infoX + 120).attr("y", y + 3)
          .attr("font-size", 11).attr("fill", COL_BOOT).attr("font-weight", 600)
          .text(`← V̂(s${i})`);
      }
    }

    // G_t accumulator (bottom of panel)
    const bottomY = PH - 24;
    if (phase === "mc") {
      let G = 0;
      for (let i = 0; i < Math.min(frame, T); i++) G += Math.pow(GAMMA, i) * steps[i].r;
      const done = frame >= T;
      g.append("text").attr("x", 0).attr("y", bottomY)
        .attr("font-size", 12).attr("fill", done ? COL_ACTIVE : "#94a3b8")
        .attr("font-weight", done ? 600 : 400)
        .text(done ? `G_t = ${G.toFixed(3)} ← update V(s₀)` : `G_t = ${G.toFixed(3)} (accumulating…)`);
    } else {
      const tStep = Math.min(frame, 1);
      if (tStep >= 1 && steps.length > 0) {
        const r0 = steps[0].r;
        const Vhat = 0; // initial estimate
        const target = r0 + GAMMA * Vhat;
        g.append("text").attr("x", 0).attr("y", bottomY)
          .attr("font-size", 12).attr("fill", COL_BOOT).attr("font-weight", 600)
          .text(`target = r + γV̂(s₁) = ${target.toFixed(3)} ← update V(s₀)`);
      } else {
        g.append("text").attr("x", 0).attr("y", bottomY)
          .attr("font-size", 12).attr("fill", "#94a3b8")
          .text("waiting for first step…");
      }
    }

    // Arrow marker def (add once per SVG)
    const defs = (svg.select("defs").empty() ? svg.append("defs") : svg.select("defs")) as d3.Selection<SVGDefsElement, unknown, null, undefined>;
    if (defs.select("#arrow").empty()) {
      defs.append("marker").attr("id", "arrow").attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("refX", 3).attr("refY", 3).attr("orient", "auto")
        .append("path").attr("d", "M0,0 L0,6 L6,3 z").attr("fill", COL_TRAJ);
    }
  }
}

customElements.define("bootstrap-move", BootstrapMove);
