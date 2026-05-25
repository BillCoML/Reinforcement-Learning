/**
 * V3 — Return Composer. Samples a trajectory through the gridworld and shows
 * how the discounted return G₀ = Σ γᵏ R_{k+1} is assembled: a trajectory strip
 * on top, a bar per step below (solid = discounted contribution γᵏ R_{k+1},
 * faded = the raw reward R_{k+1}). The γ slider rescales the bars in real time;
 * "resample" draws a fresh trajectory under uniform or the saved policy.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { rollout, type Step } from "../mdp/rollout";
import { savedPolicy } from "./mdp-shared";
import { cssVar } from "./base";
import { ACTION_NAMES, idx, rc, type MDP, type Policy } from "../mdp/types";

const NS = "http://www.w3.org/2000/svg";
const ACTION_VARS = ["--mdp-action-up", "--mdp-action-right", "--mdp-action-down", "--mdp-action-left"];
const ARROWS = ["↑", "→", "↓", "←"];
const W = 760;
const H = 240;
const M = { top: 16, right: 112, bottom: 28, left: 34 };

export class ReturnComposer extends HTMLElement {
  private mdp: MDP = buildGridworld({ slippery: false, gamma: 0.9 });
  private gamma = 0.9;
  private usesSaved = false;
  private traj: Step[] = [];
  private strip!: HTMLElement;
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private readout!: HTMLElement;
  private gammaLabel!: HTMLElement;

  connectedCallback(): void {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "return-composer" });

    const controls = document.createElement("div");
    controls.className = "mdp-controls";

    const gLabel = document.createElement("label");
    gLabel.append("γ ");
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    slider.value = String(this.gamma);
    slider.setAttribute("aria-label", "discount factor gamma");
    this.gammaLabel = document.createElement("span");
    this.gammaLabel.className = "mdp-stat";
    this.gammaLabel.textContent = this.gamma.toFixed(2);
    slider.addEventListener("input", () => {
      this.gamma = Number(slider.value);
      this.gammaLabel.textContent = this.gamma.toFixed(2);
      this.draw();
    });
    gLabel.append(slider, " ", this.gammaLabel);
    controls.appendChild(gLabel);

    const polLabel = document.createElement("label");
    polLabel.append("policy ");
    const polSel = document.createElement("select");
    polSel.setAttribute("aria-label", "rollout policy");
    for (const [v, t] of [["uniform", "uniform random"], ["saved", "saved (from V2)"]]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t;
      polSel.appendChild(o);
    }
    polSel.addEventListener("change", () => {
      this.usesSaved = polSel.value === "saved";
      this.resample();
    });
    polLabel.appendChild(polSel);
    controls.appendChild(polLabel);

    const resample = document.createElement("button");
    resample.type = "button";
    resample.textContent = "Resample trajectory";
    resample.addEventListener("click", () => this.resample());
    controls.appendChild(resample);
    body.appendChild(controls);

    this.strip = document.createElement("div");
    this.strip.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0 14px;font-family:var(--rl-font-ui);font-size:12px";
    body.appendChild(this.strip);

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.svg = d3.select(svgEl as SVGSVGElement);

    this.readout = document.createElement("div");
    this.readout.className = "mdp-readout";
    body.appendChild(this.readout);

    this.appendChild(panel);
    this.resample();
  }

  private policy(): Policy {
    const saved = savedPolicy.get();
    if (this.usesSaved && saved.pi) return { pi: saved.pi };
    return uniformPolicy(this.mdp);
  }

  private resample(): void {
    // Cap length so a flailing uniform walk still terminates the strip cleanly.
    this.traj = rollout(this.mdp, this.policy(), idx(0, 0), 24);
    this.draw();
  }

  private draw(): void {
    this.drawStrip();
    this.drawBars();
  }

  private drawStrip(): void {
    this.strip.innerHTML = "";
    const mk = (text: string, opts: { bg?: string; muted?: boolean } = {}) => {
      const box = document.createElement("span");
      box.textContent = text;
      box.style.cssText = `padding:3px 7px;border-radius:5px;border:1px solid var(--rl-border);background:${opts.bg ?? "var(--rl-surface)"};${opts.muted ? "color:var(--rl-ink-faint)" : ""}`;
      return box;
    };
    const arrow = () => {
      const s = document.createElement("span");
      s.textContent = "→";
      s.style.color = "var(--rl-ink-faint)";
      return s;
    };
    const stateBox = (s: number) => {
      const { r, c } = rc(s);
      const tag = s === idx(1, 1) ? " pit" : s === idx(2, 2) ? " goal" : "";
      return mk(`(${r},${c})${tag}`, { bg: tag ? "var(--rl-surface-2)" : "var(--rl-surface)" });
    };
    if (this.traj.length === 0) {
      this.strip.appendChild(mk("(0,0) — already terminal", { muted: true }));
      return;
    }
    this.strip.appendChild(stateBox(this.traj[0].s));
    this.traj.forEach((step, k) => {
      const act = document.createElement("span");
      act.textContent = `${ARROWS[step.a]} ${ACTION_NAMES[step.a]}`;
      act.style.cssText = `padding:2px 6px;border-radius:5px;color:#fff;background:${cssVar(ACTION_VARS[step.a])};font-size:11px`;
      this.strip.appendChild(act);
      const rew = document.createElement("span");
      rew.textContent = `R${k + 1}=${step.r > 0 ? "+" : ""}${step.r}`;
      rew.className = "mdp-stat";
      rew.style.color = step.r > 0 ? "var(--mdp-reward-pos)" : step.r < 0 ? "var(--mdp-reward-neg)" : "var(--rl-ink-faint)";
      this.strip.appendChild(rew);
      this.strip.appendChild(arrow());
      this.strip.appendChild(stateBox(step.sp));
    });
  }

  private drawBars(): void {
    this.svg.selectAll("*").remove();
    const n = this.traj.length;
    const innerW = W - M.left - M.right;
    const innerH = H - M.top - M.bottom;
    const x = d3.scaleBand<number>().domain(d3.range(Math.max(n, 1))).range([M.left, M.left + innerW]).padding(0.25);
    const y = d3.scaleLinear().domain([-1, 1]).range([M.top + innerH, M.top]);

    // zero axis
    this.svg.append("line")
      .attr("x1", M.left).attr("x2", M.left + innerW)
      .attr("y1", y(0)).attr("y2", y(0))
      .attr("stroke", "var(--rl-border)");
    this.svg.append("text").attr("x", M.left - 6).attr("y", y(0)).attr("text-anchor", "end")
      .attr("dominant-baseline", "central").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px").text("0");

    let G = 0;
    this.traj.forEach((step, k) => {
      const disc = Math.pow(this.gamma, k) * step.r;
      G += disc;
      const bx = x(k)!;
      const bw = x.bandwidth();
      // faded undiscounted reward (full width, behind)
      if (step.r !== 0) {
        this.svg.append("rect")
          .attr("x", bx).attr("width", bw)
          .attr("y", Math.min(y(step.r), y(0)))
          .attr("height", Math.abs(y(step.r) - y(0)))
          .attr("fill", step.r > 0 ? "var(--mdp-reward-pos)" : "var(--mdp-reward-neg)")
          .attr("opacity", 0.18);
      }
      // discounted contribution (foreground, narrower)
      if (disc !== 0) {
        this.svg.append("rect")
          .attr("x", bx + bw * 0.18).attr("width", bw * 0.64)
          .attr("y", Math.min(y(disc), y(0)))
          .attr("height", Math.abs(y(disc) - y(0)))
          .attr("rx", 2)
          .attr("fill", disc > 0 ? "var(--mdp-reward-pos)" : "var(--mdp-reward-neg)");
      }
      this.svg.append("text").attr("x", bx + bw / 2).attr("y", M.top + innerH + 14)
        .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px").text(`k=${k}`);
    });

    // running total marker (right-anchored so it never clips at the viewBox edge)
    this.svg.append("text").attr("x", W - 6).attr("y", y(0)).attr("text-anchor", "end")
      .attr("dominant-baseline", "central").style("font-family", "var(--rl-font-mono)").style("font-size", "13px").style("font-weight", "600")
      .attr("fill", "var(--rl-ink)").text(`G₀ = ${G.toFixed(3)}`);

    this.readout.innerHTML =
      `Discounted return <span class="mdp-stat">G₀ = Σₖ γᵏ R₍ₖ₊₁₎ = ${G.toFixed(3)}</span> at γ = ${this.gamma.toFixed(2)}. ` +
      `Solid bars are the discounted contributions; faded bars are the raw rewards. ` +
      `Slide γ to 0 (myopic — only R₁ counts) or toward 1 (every step weighed equally).`;
  }
}

customElements.define("return-composer", ReturnComposer);
