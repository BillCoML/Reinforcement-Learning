/**
 * V4 — Per-Decision IS Step-by-Step (800 × 380).
 *
 * Samples one trajectory from the uniform behavior policy and shows
 * a step-by-step breakdown comparing:
 *   - Trajectory IS: every reward weighted by the full ρ_{0:T-1}
 *   - Per-decision IS: each reward weighted by ρ_{0:t} (its own prefix)
 *
 * A sparse/dense toggle makes the per-decision advantage visible.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { optimalPolicy } from "../mdp/policies";
import { rollout, type Step } from "../mdp/rollout";
import { trajectoryISWeight, perDecisionIS } from "../importance-sampling/estimators";
import { mulberry32 } from "../importance-sampling/gaussian";

const W = 800, H = 380;
const DENSE_STEP_R = -0.01;
const GAMMA = 0.9;
const ACTION_NAMES = ["↑", "→", "↓", "←"];

const MDP = buildGridworld({ slippery: false, gamma: GAMMA });
const BEHAVIOR = uniformPolicy(MDP);
const TARGET   = optimalPolicy(MDP);
const PI_T = (s: number, a: number) => TARGET.pi[s][a];
const PI_B = (s: number, a: number) => BEHAVIOR.pi[s][a];

function stateLabel(s: number) {
  return `(${Math.floor(s / 3)},${s % 3})`;
}

function effSteps(steps: Step[], dense: boolean) {
  return steps.map((st) => ({ ...st, r: st.r + (dense ? DENSE_STEP_R : 0) }));
}

export class PerDecisionISStepwise extends HTMLElement {
  private dense = false;
  private rng = mulberry32(7);
  private svgEl!: SVGSVGElement;
  private steps: Step[] = [];
  private setStatus!: (t: string) => void;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "pd-is-stepwise" });
    this.setStatus = setStatus;

    // Controls
    const controls = document.createElement("div");
    controls.className = "rl-controls-row";
    controls.style.paddingBottom = "4px";

    const sampleBtn = document.createElement("button");
    sampleBtn.className = "rl-btn";
    sampleBtn.textContent = "Sample trajectory";
    sampleBtn.addEventListener("click", () => this.sample());
    controls.appendChild(sampleBtn);

    const toggleWrap = document.createElement("label");
    toggleWrap.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = false;
    checkbox.addEventListener("change", () => {
      this.dense = checkbox.checked;
      this.render();
    });
    toggleWrap.appendChild(checkbox);
    toggleWrap.append("Dense reward (−0.01/step)");
    controls.appendChild(toggleWrap);

    body.appendChild(controls);

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const ns = "http://www.w3.org/2000/svg";
    this.svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    this.svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svgEl.setAttribute("width", "100%");
    this.svgEl.classList.add("rl-svg");
    wrap.appendChild(this.svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    this.sample();
  }

  private sample() {
    this.steps = rollout(MDP, BEHAVIOR, 0, 20, this.rng);
    this.render();
  }

  private render() {
    const svg = d3.select(this.svgEl as SVGSVGElement);
    svg.selectAll("*").remove();

    const steps = effSteps(this.steps, this.dense);

    if (steps.length === 0) {
      svg.append("text").attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-muted)")
        .text("Click 'Sample trajectory' to begin.");
      return;
    }

    // ---- Compute IS values ----
    const trajSA = this.steps.map(({ s, a }) => ({ s, a }));
    const rhoFull = trajectoryISWeight(trajSA, PI_T, PI_B);
    const pdTotal = perDecisionIS(
      steps.map(({ s, a, r }) => ({ s, a, r })),
      PI_T, PI_B, GAMMA,
    );
    const trajTotal = rhoFull * steps.reduce((acc, { r }, t) => acc + Math.pow(GAMMA, t) * r, 0);

    // Per-step ρ_{0:t} values
    const rhoPrefix: number[] = [];
    let rho = 1;
    for (const { s, a } of this.steps) {
      const pb = PI_B(s, a);
      rho = pb === 0 ? 0 : rho * (PI_T(s, a) / pb);
      rhoPrefix.push(rho);
    }
    const T = steps.length;

    // Per-step contributions
    const trajContribs = steps.map(({ r }, t) => Math.pow(GAMMA, t) * r * rhoFull);
    const pdContribs   = steps.map(({ r }, t) => Math.pow(GAMMA, t) * r * rhoPrefix[t]);

    // ---- Layout ----
    const TABLE_W = 380;
    const CHART_X = TABLE_W + 32;
    const CHART_W = W - CHART_X - 20;
    const TABLE_MARGIN = { top: 40, left: 16, bottom: 16 };

    // ---- Left: step table ----
    const headers = ["t", "s", "a", "r", "ρ₀:t", "traj wt", "pd wt"];
    const colW = [28, 40, 24, 52, 60, 70, 70];
    let cx = TABLE_MARGIN.left;
    const colX: number[] = [];
    for (const w of colW) { colX.push(cx); cx += w; }

    const ROW_H = Math.min(26, (H - TABLE_MARGIN.top - TABLE_MARGIN.bottom - 32) / (T + 2));
    const tableG = svg.append("g").attr("transform", `translate(0,${TABLE_MARGIN.top})`);

    // Header
    headers.forEach((h, i) => {
      tableG.append("text")
        .attr("x", colX[i] + colW[i] / 2).attr("y", -6)
        .attr("text-anchor", "middle")
        .attr("font-size", "9px").attr("fill", "var(--rl-ink-muted)")
        .attr("font-weight", "600")
        .text(h);
    });
    tableG.append("line")
      .attr("x1", TABLE_MARGIN.left).attr("x2", colX[colW.length - 1] + colW[colW.length - 1])
      .attr("y1", 0).attr("y2", 0)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);

    steps.forEach(({ s, a, r }, t) => {
      const y = t * ROW_H + ROW_H / 2;
      const row = tableG.append("g");
      const vals = [
        String(t),
        stateLabel(s),
        ACTION_NAMES[a],
        r.toFixed(3),
        rhoPrefix[t].toFixed(1),
        trajContribs[t].toExponential(2),
        pdContribs[t].toExponential(2),
      ];
      const colors = [
        "var(--rl-ink)",
        "var(--rl-ink-muted)",
        "var(--rl-ink-muted)",
        Math.abs(r) > 0.001 ? "var(--is-weight)" : "var(--rl-ink-faint)",
        rhoPrefix[t] > 0 ? "var(--is-ordinary)" : "var(--is-explosion)",
        "var(--is-ordinary)",
        "var(--is-weighted)",
      ];
      vals.forEach((v, i) => {
        row.append("text")
          .attr("x", colX[i] + colW[i] / 2).attr("y", y)
          .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
          .attr("font-size", "9px").attr("fill", colors[i])
          .text(v);
      });

      // Zebra stripe
      if (t % 2 === 0) {
        row.insert("rect", ":first-child")
          .attr("x", TABLE_MARGIN.left).attr("y", y - ROW_H / 2)
          .attr("width", colX[6] + colW[6] - TABLE_MARGIN.left)
          .attr("height", ROW_H)
          .attr("fill", "var(--rl-surface-2)").attr("rx", 2);
      }
    });

    // Totals row
    const totY = T * ROW_H + ROW_H;
    tableG.append("line")
      .attr("x1", TABLE_MARGIN.left).attr("x2", colX[6] + colW[6])
      .attr("y1", T * ROW_H + 2).attr("y2", T * ROW_H + 2)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
    tableG.append("text").attr("x", colX[0]).attr("y", totY)
      .attr("font-size", "9px").attr("fill", "var(--rl-ink)").attr("font-weight", "600")
      .attr("dominant-baseline", "middle").text("Σ");
    tableG.append("text").attr("x", colX[5] + colW[5] / 2).attr("y", totY)
      .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
      .attr("font-size", "9px").attr("fill", "var(--is-ordinary)").attr("font-weight", "600")
      .text(trajTotal.toFixed(4));
    tableG.append("text").attr("x", colX[6] + colW[6] / 2).attr("y", totY)
      .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
      .attr("font-size", "9px").attr("fill", "var(--is-weighted)").attr("font-weight", "600")
      .text(pdTotal.toFixed(4));

    // ---- Right: per-step contribution bar chart ----
    const CHART_MARGIN = { top: 28, bottom: 40, right: 12 };
    const chartH = H - CHART_MARGIN.top - CHART_MARGIN.bottom;
    const chartG = svg.append("g")
      .attr("transform", `translate(${CHART_X},${CHART_MARGIN.top})`);

    const allC = [...trajContribs, ...pdContribs];
    const valMin = Math.min(0, ...allC);
    const valMax = Math.max(...allC, 1);
    const yScale = d3.scaleLinear().domain([valMin, valMax]).range([chartH, 0]);

    const barW = Math.max(8, (CHART_W / T / 2) - 3);
    const groupW = CHART_W / T;

    // Title
    chartG.append("text").attr("x", CHART_W / 2).attr("y", -12)
      .attr("text-anchor", "middle").attr("font-size", "10px")
      .attr("fill", "var(--rl-ink-faint)")
      .text("Per-step contributions");

    // Zero line
    if (valMin < 0) {
      chartG.append("line").attr("x1", 0).attr("x2", CHART_W)
        .attr("y1", yScale(0)).attr("y2", yScale(0))
        .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
    }

    for (let t = 0; t < T; t++) {
      const gx = t * groupW;
      const tc = trajContribs[t];
      const pc = pdContribs[t];

      // Traj IS bar
      const ty0 = Math.min(yScale(0), yScale(tc));
      const th  = Math.abs(yScale(tc) - yScale(0));
      chartG.append("rect")
        .attr("x", gx + groupW / 2 - barW - 2)
        .attr("y", ty0).attr("width", barW).attr("height", Math.max(1, th))
        .attr("fill", "var(--is-ordinary)").attr("opacity", 0.8);

      // PD IS bar
      const py0 = Math.min(yScale(0), yScale(pc));
      const ph  = Math.abs(yScale(pc) - yScale(0));
      chartG.append("rect")
        .attr("x", gx + groupW / 2 + 2)
        .attr("y", py0).attr("width", barW).attr("height", Math.max(1, ph))
        .attr("fill", "var(--is-weighted)").attr("opacity", 0.8);

      // t label
      chartG.append("text").attr("x", gx + groupW / 2).attr("y", chartH + 14)
        .attr("text-anchor", "middle").attr("font-size", "9px")
        .attr("fill", "var(--rl-ink-muted)").text(String(t));
    }

    // Y axis
    chartG.append("g")
      .call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0)
        .tickFormat((d) => d3.format(".2f")(+d)))
      .selectAll("text").style("font-size", "9px");

    // X axis label
    chartG.append("text").attr("x", CHART_W / 2).attr("y", chartH + 28)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-muted)").text("step t");

    // Legend
    const lx = CHART_W - 120;
    chartG.append("rect").attr("x", lx).attr("y", 0).attr("width", 10).attr("height", 10)
      .attr("fill", "var(--is-ordinary)").attr("opacity", 0.8);
    chartG.append("text").attr("x", lx + 14).attr("y", 9).attr("font-size", "9px")
      .attr("fill", "var(--is-ordinary)").text("traj IS");
    chartG.append("rect").attr("x", lx).attr("y", 14).attr("width", 10).attr("height", 10)
      .attr("fill", "var(--is-weighted)").attr("opacity", 0.8);
    chartG.append("text").attr("x", lx + 14).attr("y", 23).attr("font-size", "9px")
      .attr("fill", "var(--is-weighted)").text("PD IS");

    // Separator
    svg.append("line")
      .attr("x1", TABLE_W + 16).attr("x2", TABLE_W + 16)
      .attr("y1", 8).attr("y2", H - 8)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);

    // Status
    const same = Math.abs(trajTotal - pdTotal) < 1e-8;
    this.setStatus?.(
      this.dense
        ? `dense  traj=${trajTotal.toFixed(4)}  pd=${pdTotal.toFixed(4)}`
        : `sparse  both = ${trajTotal.toFixed(4)}${same ? " (identical ✓)" : ""}`,
    );
  }
}

customElements.define("per-decision-is-stepwise", PerDecisionISStepwise);
