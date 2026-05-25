/**
 * V2 — Regret Decomposition. A stacked-area chart of cumulative regret split by
 * arm: each suboptimal arm i contributes a band of height Δ_i · N_i(t). The
 * optimal arm contributes zero (transparent). A dashed Lai–Robbins floor
 * C·log t is overlaid for contrast. Scrub t to watch the bands fill.
 *
 * Trajectories are the *expected* pull-count paths N_i(t) for each preset —
 * analytic and clean, so the gap decomposition R_T = Σ Δ_i E[N_i(T)] is
 * literally what the chart draws.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { laiRobbinsConstant } from "../bandits/stats";

const MEANS = [0.3, 0.5, 0.7];
const T_MAX = 5000;

type CountFn = (t: number) => [number, number, number];

interface Preset {
  key: string;
  label: string;
  counts: CountFn;
}

const PRESETS: Preset[] = [
  { key: "oracle", label: "oracle (always arm 3)", counts: (t) => [0, 0, t] },
  {
    key: "greedy",
    label: "greedy-locked (onto arm 1)",
    counts: (t) => [Math.max(0, t - 2), 1, 1],
  },
  { key: "random", label: "uniform random", counts: (t) => [t / 3, t / 3, t / 3] },
  {
    key: "epsgreedy",
    label: "ε-greedy (ε=0.10)",
    counts: (t) => {
      const ex = 0.1 / 3;
      return [t * ex, t * ex, t * (0.9 + ex)];
    },
  },
];

export class RegretDecomposition extends HTMLElement {
  private gaps = MEANS.map((m) => Math.max(...MEANS) - m);
  private lrConst = laiRobbinsConstant(MEANS);
  private preset = PRESETS[3];
  private tCur = T_MAX;

  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private setStatus: (s: string) => void = () => {};
  private readout!: HTMLElement;

  private width = 720;
  private height = 320;
  private margin = { top: 18, right: 96, bottom: 36, left: 52 };

  connectedCallback(): void {
    this.render();
  }

  private render(): void {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "regret-decomposition" });
    this.setStatus = setStatus;

    // controls
    const controls = document.createElement("div");
    controls.className = "rl-controls";
    controls.style.marginBottom = "12px";

    const presetLabel = document.createElement("label");
    presetLabel.textContent = "trace:";
    const select = document.createElement("select");
    for (const p of PRESETS) {
      const opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.label;
      if (p.key === this.preset.key) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      this.preset = PRESETS.find((p) => p.key === select.value)!;
      this.draw();
    });
    presetLabel.appendChild(select);

    const scrubLabel = document.createElement("label");
    scrubLabel.textContent = "t:";
    const scrub = document.createElement("input");
    scrub.type = "range";
    scrub.min = "1";
    scrub.max = String(T_MAX);
    scrub.value = String(this.tCur);
    scrub.style.width = "220px";
    scrub.addEventListener("input", () => {
      this.tCur = +scrub.value;
      this.draw();
    });
    scrubLabel.appendChild(scrub);

    controls.append(presetLabel, scrubLabel);

    // chart
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const NS = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    this.svg = d3.select(svgEl as SVGSVGElement);

    this.readout = document.createElement("div");
    this.readout.className = "stats-readout";
    this.readout.style.marginTop = "10px";

    body.append(controls, wrap, this.readout);
    this.appendChild(panel);

    this.draw();
  }

  private draw(): void {
    const m = this.margin;
    const iw = this.width - m.left - m.right;
    const ih = this.height - m.top - m.bottom;

    // sample expected trajectories up to tCur
    const n = 240;
    const data: { t: number; r1: number; r2: number; regret: number; floor: number }[] = [];
    for (let k = 0; k <= n; k++) {
      const t = (this.tCur * k) / n;
      const [N1, N2] = this.preset.counts(t);
      const r1 = this.gaps[0] * N1;
      const r2 = this.gaps[1] * N2;
      const floor = t >= 1 ? this.lrConst * Math.log(t) : 0;
      data.push({ t, r1, r2, regret: r1 + r2, floor });
    }

    const finalRegret = data[data.length - 1].regret;
    const finalFloor = data[data.length - 1].floor;
    const yMax = Math.max(finalRegret * 1.08, finalFloor * 1.3, 1);

    const x = d3.scaleLinear().domain([0, this.tCur]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, yMax]).range([ih, 0]);

    this.svg.selectAll("*").remove();
    const g = this.svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    // axes
    g.append("g")
      .attr("class", "rl-axis")
      .attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("~s")));
    g.append("g").attr("class", "rl-axis").call(d3.axisLeft(y).ticks(5));

    g.append("text")
      .attr("class", "axis-label")
      .attr("x", iw / 2)
      .attr("y", ih + 30)
      .attr("text-anchor", "middle")
      .text("rounds  t");
    g.append("text")
      .attr("class", "axis-label")
      .attr("transform", "rotate(-90)")
      .attr("x", -ih / 2)
      .attr("y", -38)
      .attr("text-anchor", "middle")
      .text("cumulative regret  R_t");

    // stacked areas: band for arm 2 (bottom), arm 1 (stacked on top)
    const area2 = d3
      .area<(typeof data)[number]>()
      .x((d) => x(d.t))
      .y0(() => y(0))
      .y1((d) => y(d.r2));
    const area1 = d3
      .area<(typeof data)[number]>()
      .x((d) => x(d.t))
      .y0((d) => y(d.r2))
      .y1((d) => y(d.r2 + d.r1));

    g.append("path")
      .datum(data)
      .attr("fill", "var(--rl-regret-tint)")
      .attr("stroke", "var(--rl-regret)")
      .attr("stroke-width", 0.8)
      .attr("opacity", 0.7)
      .attr("d", area2);
    g.append("path")
      .datum(data)
      .attr("fill", "var(--rl-regret)")
      .attr("fill-opacity", 0.34)
      .attr("stroke", "var(--rl-regret)")
      .attr("stroke-width", 1)
      .attr("d", area1);

    // Lai-Robbins floor (dashed)
    const floorLine = d3
      .line<(typeof data)[number]>()
      .x((d) => x(d.t))
      .y((d) => y(d.floor));
    g.append("path")
      .datum(data.filter((d) => d.t >= 1))
      .attr("fill", "none")
      .attr("stroke", "var(--rl-rule)")
      .attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "5 4")
      .attr("d", floorLine);

    // labels at right edge
    const last = data[data.length - 1];
    const labelAt = (yVal: number, text: string, color: string) =>
      g
        .append("text")
        .attr("class", "annot")
        .attr("x", iw + 6)
        .attr("y", Math.max(8, Math.min(ih, y(yVal))) + 3)
        .attr("fill", color)
        .text(text);
    labelAt(last.r2 + last.r1, "arm 1 (Δ=0.4)", "var(--rl-regret)");
    if (last.r2 > 0.5) labelAt(last.r2, "arm 2 (Δ=0.2)", "var(--rl-regret)");
    labelAt(last.floor, "LR floor", "var(--rl-ink-muted)");

    // readout
    const [N1, N2, N3] = this.preset.counts(this.tCur);
    this.setStatus(`t=${Math.round(this.tCur)} / ${T_MAX}`);
    this.readout.innerHTML = `
      <div class="row"><span class="k">N₁ (Δ=0.4)</span><span>${Math.round(N1)}</span></div>
      <div class="row"><span class="k">N₂ (Δ=0.2)</span><span>${Math.round(N2)}</span></div>
      <div class="row"><span class="k">N₃ (Δ=0)</span><span>${Math.round(N3)}</span></div>
      <div class="row"><span class="k">regret R_t = 0.4·N₁ + 0.2·N₂</span><span>${finalRegret.toFixed(1)}</span></div>
      <div class="row"><span class="k">Lai–Robbins floor ${this.lrConst.toFixed(2)}·log t</span><span>${finalFloor.toFixed(1)}</span></div>`;
  }
}

customElements.define("regret-decomposition", RegretDecomposition);
