/**
 * V6 — Thompson Posterior Evolution. Three Beta posteriors side by side. Each
 * step: sample θ_i from every posterior (a marker drops to the axis), pull the
 * arm with the highest sample, and tween its posterior to the updated
 * Beta(α+s, β+f). Replays the deterministic recorded trace (seed 0) so the
 * on-screen beliefs match the §6 prose; "Reset" restarts it.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { StepLoop } from "./loop";
import { betaPdf } from "../bandits/stats";
import { loadBetaSnapshots, type BetaTraceStep } from "../lesson/data";

const MEANS = [0.3, 0.5, 0.7];
const ARM_COLORS = ["#8b5cf6", "#6d28d9", "#4c1d95"]; // thompson violet ramp
const NX = 120;
const XS = Array.from({ length: NX }, (_, k) => (k + 0.5) / NX);

export class ThompsonPosteriorEvolution extends HTMLElement {
  private trace: BetaTraceStep[] = [];
  private t = 0; // number of steps applied
  private alpha = [1, 1, 1];
  private beta = [1, 1, 1];
  private displayed: number[][] = [];
  private displayedYmax = 4;

  private loop!: StepLoop;
  private panel!: PanelHandle;
  private arms: {
    g: d3.Selection<SVGGElement, unknown, null, undefined>;
    area: d3.Selection<SVGPathElement, unknown, null, undefined>;
    marker: d3.Selection<SVGLineElement, unknown, null, undefined>;
    markerDot: d3.Selection<SVGCircleElement, unknown, null, undefined>;
    label: d3.Selection<SVGTextElement, unknown, null, undefined>;
    yAxis: d3.Selection<SVGGElement, unknown, null, undefined>;
    x: d3.ScaleLinear<number, number>;
    iw: number;
    ih: number;
    m: { top: number; right: number; bottom: number; left: number };
  }[] = [];
  private playBtn!: HTMLButtonElement;
  private reasonEl!: HTMLElement;

  connectedCallback(): void {
    this.render();
    void this.init();
  }
  disconnectedCallback(): void {
    this.loop?.destroy();
  }

  private async init(): Promise<void> {
    try {
      const data = await loadBetaSnapshots();
      this.trace = data.trace;
    } catch {
      this.reasonEl.textContent = "could not load posterior trace";
      return;
    }
    this.resetState();
    this.update(false);
  }

  private resetState(): void {
    this.t = 0;
    this.alpha = [1, 1, 1];
    this.beta = [1, 1, 1];
    this.displayed = MEANS.map(() => XS.map((x) => betaPdf(x, 1, 1)));
    this.displayedYmax = 4;
  }

  private stepOnce(): boolean {
    if (this.t >= this.trace.length) return false;
    const step = this.trace[this.t];
    this.alpha = [...step.alpha];
    this.beta = [...step.beta];
    this.t++;
    return true;
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "thompson-posterior-evolution",
      heavy: true,
      mobileNotice: "The posterior animation is interactive — view on a wider screen.",
    });

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg");
    const W = 800, H = 300;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    const svg = d3.select(svgEl as SVGSVGElement);

    const panelW = W / 3;
    const m = { top: 16, right: 18, bottom: 28, left: 30 };
    for (let i = 0; i < 3; i++) {
      const gx = i * panelW;
      const iw = panelW - m.left - m.right;
      const ih = H - m.top - m.bottom;
      const g = svg.append("g").attr("transform", `translate(${gx + m.left},${m.top})`);
      const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
      g.append("g").attr("class", "rl-axis").attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(3));
      const yAxis = g.append("g").attr("class", "rl-axis");
      const area = g.append("path").attr("fill", ARM_COLORS[i]).attr("fill-opacity", 0.3)
        .attr("stroke", ARM_COLORS[i]).attr("stroke-width", 1.5) as d3.Selection<
        SVGPathElement, unknown, null, undefined
      >;
      const marker = g.append("line").attr("stroke", "var(--rl-rule)").attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "2 2").attr("opacity", 0) as d3.Selection<
        SVGLineElement, unknown, null, undefined
      >;
      const markerDot = g.append("circle").attr("r", 4).attr("fill", "var(--rl-rule)")
        .attr("opacity", 0) as d3.Selection<SVGCircleElement, unknown, null, undefined>;
      const label = g.append("text").attr("class", "legend-label").attr("x", iw / 2).attr("y", -2)
        .attr("text-anchor", "middle").attr("fill", ARM_COLORS[i]) as d3.Selection<
        SVGTextElement, unknown, null, undefined
      >;
      this.arms.push({ g, area, marker, markerDot, label, yAxis, x, iw, ih, m });
    }

    this.reasonEl = document.createElement("div");
    this.reasonEl.className = "pick-reason";

    this.panel.body.append(this.buildControls(), svgEl, this.reasonEl);
    this.appendChild(this.panel.panel);

    this.loop = new StepLoop({
      baseIntervalMs: 700,
      maxStepsPerFrame: 4,
      onStep: () => this.stepOnce(),
      onFrame: () => this.update(true),
    });
    this.resetState();
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

    const autoBtn = document.createElement("button");
    autoBtn.textContent = "Auto-play 50 steps";
    autoBtn.addEventListener("click", () => {
      const target = Math.min(this.trace.length, this.t + 50);
      this.loop.setSpeed(4);
      this.loop.pause();
      const run = () => {
        if (this.t >= target) {
          this.loop.pause();
          this.syncBtn();
          return;
        }
        if (this.stepOnce()) this.update(true);
        if (this.t < target) requestAnimationFrame(() => setTimeout(run, 90));
        else this.syncBtn();
      };
      run();
    });

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.loop.pause();
      this.resetState();
      this.update(false);
      this.syncBtn();
    });

    const spWrap = document.createElement("label");
    spWrap.append("speed ");
    const sp = document.createElement("select");
    for (const s of [0.5, 1, 2, 4, 8]) {
      const o = document.createElement("option");
      o.value = String(s);
      o.textContent = `${s}×`;
      if (s === 1) o.selected = true;
      sp.appendChild(o);
    }
    sp.addEventListener("change", () => this.loop.setSpeed(+sp.value));
    spWrap.appendChild(sp);

    c.append(this.playBtn, stepBtn, autoBtn, resetBtn, spWrap);
    return c;
  }

  private togglePlay(): void {
    if (this.t >= this.trace.length) {
      this.resetState();
      this.update(false);
    }
    if (this.loop.isPlaying()) this.loop.pause();
    else this.loop.play();
    this.syncBtn();
  }
  private syncBtn(): void {
    this.playBtn.textContent = this.loop.isPlaying()
      ? "Pause"
      : this.t >= this.trace.length
        ? "Replay"
        : "Run";
  }

  private update(animate: boolean): void {
    const target = MEANS.map((_, i) => XS.map((x) => betaPdf(x, this.alpha[i], this.beta[i])));
    const targetYmax = Math.max(4, ...target.flat()) * 1.05;

    const curStep = this.t > 0 ? this.trace[this.t - 1] : null;
    const pulled = curStep ? curStep.arm : -1;

    const dur = animate && !this.loop.reducedMotion ? 300 : 0;
    const fromY = this.displayedYmax;
    const fromD = this.displayed.map((a) => [...a]);
    const interpY = d3.interpolate(fromY, targetYmax);
    const interpD = target.map((tg, i) => d3.interpolateArray(fromD[i], tg));

    const redraw = (tt: number) => {
      const ymax = interpY(tt);
      for (let i = 0; i < 3; i++) {
        const arm = this.arms[i];
        const y = d3.scaleLinear().domain([0, ymax]).range([arm.ih, 0]);
        const dens = interpD[i](tt);
        const area = d3.area<number>()
          .x((_, k) => arm.x(XS[k]))
          .y0(arm.ih)
          .y1((d) => y(d));
        arm.area.attr("d", area(dens));
        if (tt === 1) {
          arm.yAxis.call(d3.axisLeft(y).ticks(3));
        }
      }
    };

    if (dur > 0) {
      d3.select(this).interrupt("post").transition("post").duration(dur).ease(d3.easeCubicOut)
        .tween("post", () => (tt: number) => redraw(tt))
        .on("end", () => {
          this.displayed = target.map((a) => [...a]);
          this.displayedYmax = targetYmax;
          redraw(1);
        });
    } else {
      this.displayed = target.map((a) => [...a]);
      this.displayedYmax = targetYmax;
      redraw(1);
    }

    // labels
    for (let i = 0; i < 3; i++) {
      this.arms[i].label.text(`arm ${i + 1}: Beta(${this.alpha[i]}, ${this.beta[i]})`);
    }

    // sample markers for the current step
    if (curStep) {
      const y0at = (i: number, x: number) => {
        const yScale = d3.scaleLinear().domain([0, targetYmax]).range([this.arms[i].ih, 0]);
        return yScale(betaPdf(x, this.alpha[i], this.beta[i]));
      };
      for (let i = 0; i < 3; i++) {
        const arm = this.arms[i];
        const theta = curStep.samples[i];
        const px = arm.x(theta);
        arm.marker.attr("x1", px).attr("x2", px).attr("y1", arm.ih).attr("y2", y0at(i, theta))
          .attr("opacity", i === pulled ? 1 : 0.45)
          .attr("stroke", i === pulled ? ARM_COLORS[i] : "var(--rl-ink-faint)")
          .attr("stroke-width", i === pulled ? 2 : 1.5);
        arm.markerDot.attr("cx", px).attr("cy", arm.ih).attr("opacity", i === pulled ? 1 : 0.45)
          .attr("fill", i === pulled ? ARM_COLORS[i] : "var(--rl-ink-faint)");
        if (animate && !this.loop.reducedMotion) {
          arm.markerDot.attr("cy", y0at(i, theta)).transition().duration(dur).attr("cy", arm.ih);
        }
      }
      const best = curStep.samples
        .map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s)[0];
      this.reasonEl.textContent = `sampled θ = (${curStep.samples.map((s) => s.toFixed(2)).join(", ")}) → highest is arm ${best.i + 1}, pull it (reward ${curStep.reward})`;
    } else {
      for (const arm of this.arms) {
        arm.marker.attr("opacity", 0);
        arm.markerDot.attr("opacity", 0);
      }
      this.reasonEl.textContent = "uniform priors Beta(1,1) — press Run to sample and pull";
    }

    this.panel.setStatus(`t=${this.t} / ${this.trace.length || "…"}`);
  }
}

customElements.define("thompson-posterior-evolution", ThompsonPosteriorEvolution);
