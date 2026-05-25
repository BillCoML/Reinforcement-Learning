/**
 * V5 — UCB Confidence Bounds. One horizontal track per arm showing the
 * empirical mean μ̂_i (dot), the confidence interval [μ̂_i ± √(2 ln t / N_i)]
 * (range bar), and the true mean μ_i (dashed reference, toggleable). Each step
 * the arm with the highest *upper* edge is pulled; its bar shrinks while the
 * others expand subtly as log t grows — the visual signature of optimism.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { StepLoop } from "./loop";
import { armColor } from "./base";
import { UCB1 } from "../bandits/algorithms";
import { BernoulliBandit } from "../bandits/env";
import { RegretTracker } from "../bandits/regret";
import { mulberry32 } from "../bandits/stats";

const MEANS = [0.3, 0.5, 0.7];

export class UCBConfidenceBounds extends HTMLElement {
  private seed = 7;
  private horizon = 2000;
  private showTrue = true;

  private env!: BernoulliBandit;
  private algo!: UCB1;
  private tracker!: RegretTracker;
  private rng!: () => number;
  private t = 0;
  private optimalArm = 2;
  private lastArm = -1;

  private loop!: StepLoop;
  private panel!: PanelHandle;
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private rows: d3.Selection<SVGGElement, unknown, null, undefined>[] = [];
  private axisG!: d3.Selection<SVGGElement, unknown, null, undefined>;
  private readout!: HTMLElement;
  private reasonEl!: HTMLElement;
  private playBtn!: HTMLButtonElement;

  private readonly W = 880;
  private readonly H = 360;
  private readonly m = { top: 18, right: 150, bottom: 40, left: 120 };

  connectedCallback(): void {
    const url = new URLSearchParams(location.search);
    if (url.has("seed")) this.seed = +url.get("seed")! || 7;
    this.reset();
    this.render();
  }
  disconnectedCallback(): void {
    this.loop?.destroy();
  }

  private reset(): void {
    this.rng = mulberry32(this.seed);
    this.env = new BernoulliBandit(MEANS, this.rng);
    this.algo = new UCB1(this.rng);
    this.algo.reset(MEANS.length);
    this.tracker = new RegretTracker();
    this.optimalArm = this.env.optimalArm();
    this.t = 0;
    this.lastArm = -1;
  }

  private stepOnce(): boolean {
    if (this.t >= this.horizon) return false;
    const arm = this.algo.selectArm(this.t);
    const r = this.env.pull(arm);
    this.algo.update(arm, r);
    this.tracker.observe(this.env, arm);
    this.lastArm = arm;
    this.t++;
    return true;
  }

  private get ih(): number {
    return this.H - this.m.top - this.m.bottom;
  }
  private get iw(): number {
    return this.W - this.m.left - this.m.right;
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "ucb-confidence-bounds",
      heavy: true,
      mobileNotice: "UCB's confidence bars animate live — view on a wider screen.",
    });

    const layout = document.createElement("div");
    layout.style.display = "flex";
    layout.style.gap = "16px";
    layout.style.alignItems = "flex-start";
    layout.style.flexWrap = "wrap";

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${this.W} ${this.H}`);
    svgEl.style.flex = "1 1 520px";
    svgEl.style.minWidth = "440px";
    svgEl.classList.add("rl-svg");
    this.svg = d3.select(svgEl as SVGSVGElement);

    this.readout = document.createElement("div");
    this.readout.className = "stats-readout";
    this.readout.style.flex = "0 0 200px";

    layout.append(svgEl, this.readout);

    this.reasonEl = document.createElement("div");
    this.reasonEl.className = "pick-reason";

    this.panel.body.append(this.buildControls(), layout, this.reasonEl);
    this.appendChild(this.panel.panel);

    this.buildTracks();

    this.loop = new StepLoop({
      baseIntervalMs: 750,
      maxStepsPerFrame: 8,
      onStep: () => this.stepOnce(),
      onFrame: () => this.update(true),
    });

    this.update(false);
  }

  private buildControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";
    c.style.marginBottom = "12px";

    this.playBtn = document.createElement("button");
    this.playBtn.className = "primary";
    this.playBtn.textContent = "Run";
    this.playBtn.addEventListener("click", () => this.togglePlay());

    const stepBtn = document.createElement("button");
    stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => {
      this.loop.step();
      this.syncBtn();
    });

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.loop.pause();
      this.reset();
      this.update(false);
      this.syncBtn();
    });

    const spWrap = document.createElement("label");
    spWrap.append("speed ");
    const sp = document.createElement("select");
    for (const s of [0.25, 0.5, 1, 2, 4, 8, 16]) {
      const o = document.createElement("option");
      o.value = String(s);
      o.textContent = `${s}×`;
      if (s === 1) o.selected = true;
      sp.appendChild(o);
    }
    sp.addEventListener("change", () => this.loop.setSpeed(+sp.value));
    spWrap.appendChild(sp);

    const trueWrap = document.createElement("label");
    const tc = document.createElement("input");
    tc.type = "checkbox";
    tc.checked = this.showTrue;
    tc.addEventListener("change", () => {
      this.showTrue = tc.checked;
      this.update(false);
    });
    trueWrap.append(tc, " show true means");

    c.append(this.playBtn, stepBtn, resetBtn, spWrap, trueWrap);
    queueMicrotask(() => this.loop?.setSpeed(+sp.value));
    return c;
  }

  private togglePlay(): void {
    if (this.t >= this.horizon) {
      this.reset();
      this.update(false);
    }
    if (this.loop.isPlaying()) this.loop.pause();
    else this.loop.play();
    this.syncBtn();
  }
  private syncBtn(): void {
    this.playBtn.textContent = this.loop.isPlaying()
      ? "Pause"
      : this.t >= this.horizon
        ? "Replay"
        : "Run";
  }

  private buildTracks(): void {
    const g = this.svg.append("g").attr("transform", `translate(${this.m.left},${this.m.top})`);
    this.axisG = g.append("g").attr("class", "rl-axis").attr("transform", `translate(0,${this.ih})`);
    g.append("text")
      .attr("class", "axis-label")
      .attr("x", this.iw / 2)
      .attr("y", this.ih + 34)
      .attr("text-anchor", "middle")
      .text("value (empirical mean ± confidence bonus)");

    const rowH = this.ih / MEANS.length;
    this.rows = [];
    for (let i = 0; i < MEANS.length; i++) {
      const cy = rowH * (i + 0.5);
      const row = g.append("g").attr("transform", `translate(0,${cy})`) as d3.Selection<
        SVGGElement,
        unknown,
        null,
        undefined
      >;
      // baseline
      row.append("line").attr("class", "baseline").attr("x1", 0).attr("x2", this.iw)
        .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
      // highlight halo (behind bar)
      row.append("rect").attr("class", "halo").attr("height", 30).attr("y", -15).attr("rx", 6)
        .attr("fill", "var(--rl-ucb-tint)").attr("opacity", 0);
      // range bar
      row.append("rect").attr("class", "bar").attr("height", 12).attr("y", -6).attr("rx", 6)
        .attr("fill", armColor(i, i === this.optimalArm)).attr("opacity", 0.32)
        .attr("stroke", armColor(i, i === this.optimalArm)).attr("stroke-width", 1);
      // mean dot
      row.append("circle").attr("class", "dot").attr("r", 5)
        .attr("fill", armColor(i, i === this.optimalArm));
      // upper-edge tick
      row.append("line").attr("class", "upper").attr("y1", -10).attr("y2", 10)
        .attr("stroke", armColor(i, i === this.optimalArm)).attr("stroke-width", 1.5);
      // true mean reference
      row.append("line").attr("class", "truth").attr("y1", -16).attr("y2", 16)
        .attr("stroke", "var(--rl-rule)").attr("stroke-width", 1.5).attr("stroke-dasharray", "3 3");
      // arm label
      row.append("text").attr("class", "arm-label legend-label").attr("x", -12).attr("text-anchor", "end")
        .attr("dy", "0.32em").text(`arm ${i + 1}`);
      this.rows.push(row);
    }
  }

  private update(animate: boolean): void {
    const st = this.algo.state();
    const bonuses = (st.extra?.bonus as number[]) ?? MEANS.map(() => Infinity);
    const ucb = (st.extra?.ucb as number[]) ?? MEANS.map(() => Infinity);
    const muhat = st.muhat;
    const N = st.N;

    const finiteUpper = ucb.map((u) => (isFinite(u) ? u : muhat.length ? 2 : 2));
    const maxUpper = Math.max(1.0, ...finiteUpper.filter((v) => isFinite(v)));
    const domMax = Math.min(2.2, Math.max(1.05, maxUpper * 1.04));
    const x = d3.scaleLinear().domain([0, domMax]).range([0, this.iw]);

    if (animate && !this.loop.reducedMotion) {
      this.axisG.transition().duration(250).call(d3.axisBottom(x).ticks(5));
    } else {
      this.axisG.call(d3.axisBottom(x).ticks(5));
    }

    const dur = animate && !this.loop.reducedMotion ? 250 : 0;
    const pickArm = this.lastArm;

    for (let i = 0; i < MEANS.length; i++) {
      const row = this.rows[i];
      const pulled = N[i] > 0;
      const lo = Math.max(0, muhat[i] - bonuses[i]);
      const hi = pulled ? muhat[i] + bonuses[i] : domMax; // un-pulled: full optimism
      const x0 = x(Math.min(lo, domMax));
      const x1 = x(Math.min(hi, domMax));

      row.select<SVGRectElement>("rect.bar").transition().duration(dur)
        .attr("x", x0).attr("width", Math.max(2, x1 - x0));
      row.select<SVGRectElement>("rect.halo").transition().duration(dur)
        .attr("x", Math.max(0, x0 - 6)).attr("width", Math.max(4, x1 - x0 + 12))
        .attr("opacity", i === pickArm ? 1 : 0);
      row.select<SVGCircleElement>("circle.dot").transition().duration(dur)
        .attr("cx", x(Math.min(muhat[i], domMax))).attr("opacity", pulled ? 1 : 0.25);
      row.select<SVGLineElement>("line.upper").transition().duration(dur)
        .attr("x1", x1).attr("x2", x1);
      row.select<SVGLineElement>("line.truth")
        .attr("x1", x(MEANS[i])).attr("x2", x(MEANS[i]))
        .attr("opacity", this.showTrue ? 0.8 : 0);
    }

    // side readout
    this.readout.innerHTML = MEANS.map((_, i) => {
      const sw = `<span class="swatch" style="background:${armColor(i, i === this.optimalArm)}"></span>`;
      const ucbStr = isFinite(ucb[i]) ? ucb[i].toFixed(3) : "∞";
      return `<div class="row"><span class="k">${sw}arm ${i + 1}</span><span>N=${N[i]}</span></div>
              <div class="row"><span class="k">UCB</span><span>${ucbStr}</span></div>`;
    }).join("") + `<div class="row"><span class="k">regret R_t</span><span>${this.tracker.current().toFixed(1)}</span></div>`;

    // pick reason
    if (pickArm >= 0 && this.t > MEANS.length) {
      const sorted = ucb
        .map((u, i) => ({ u, i }))
        .sort((a, b) => b.u - a.u);
      const top = sorted[0], second = sorted[1];
      this.reasonEl.textContent = `pulled arm ${top.i + 1} because UCB${top.i + 1} = ${top.u.toFixed(3)} > UCB${second.i + 1} = ${second.u.toFixed(3)}`;
    } else if (pickArm >= 0) {
      this.reasonEl.textContent = `initializing — pulling arm ${pickArm + 1} once`;
    } else {
      this.reasonEl.textContent = "press Run or Step to begin";
    }

    this.panel.setStatus(`t=${this.t} / ${this.horizon}`);
  }
}

customElements.define("ucb-confidence-bounds", UCBConfidenceBounds);
