/**
 * V4 — ε-Greedy Explorer. A live run of ε-greedy on the running example:
 * a building regret curve (with optional Lai–Robbins overlay), per-arm pull
 * frequencies N_i(t)/t, and a color-coded timeline of recent pulls. Sliders
 * control ε, the constant-vs-decay schedule, the horizon, and playback speed.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { StepLoop } from "./loop";
import { armColor } from "./base";
import { EpsilonGreedy } from "../bandits/algorithms";
import { BernoulliBandit } from "../bandits/env";
import { RegretTracker } from "../bandits/regret";
import { laiRobbinsConstant, mulberry32 } from "../bandits/stats";

const MEANS = [0.3, 0.5, 0.7];
const LR = laiRobbinsConstant(MEANS);

export class EpsilonGreedyExplorer extends HTMLElement {
  private epsilon = 0.1;
  private decay = false;
  private horizon = 5000;
  private seed = 0;
  private overlayFloor = true;

  private env!: BernoulliBandit;
  private algo!: EpsilonGreedy;
  private tracker!: RegretTracker;
  private rng!: () => number;
  private t = 0;
  private regretHist: number[] = [];
  private recent: number[] = [];
  private optimalArm = 2;

  private loop!: StepLoop;
  private panel!: PanelHandle;
  private regretSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private barSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private timelineEl!: HTMLElement;
  private counterEl!: HTMLElement;
  private playBtn!: HTMLButtonElement;

  connectedCallback(): void {
    const url = new URLSearchParams(location.search);
    if (url.has("seed")) this.seed = +url.get("seed")! || 0;
    this.reset();
    this.render();
  }

  disconnectedCallback(): void {
    this.loop?.destroy();
  }

  private reset(): void {
    this.rng = mulberry32(this.seed);
    this.env = new BernoulliBandit(MEANS, this.rng);
    this.algo = new EpsilonGreedy(this.epsilon, this.rng, this.decay ? { c: 1, d: 0.2 } : null);
    this.algo.reset(MEANS.length);
    this.tracker = new RegretTracker();
    this.optimalArm = this.env.optimalArm();
    this.t = 0;
    this.regretHist = [];
    this.recent = [];
  }

  private stepOnce(): boolean {
    if (this.t >= this.horizon) return false;
    const arm = this.algo.selectArm(this.t);
    const r = this.env.pull(arm);
    this.algo.update(arm, r);
    this.tracker.observe(this.env, arm);
    this.regretHist.push(this.tracker.current());
    this.recent.push(arm);
    if (this.recent.length > 80) this.recent.shift();
    this.t++;
    return true;
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "epsilon-greedy-explorer",
      heavy: true,
      mobileNotice: "The ε-greedy explorer is interactive — view on a wider screen.",
    });

    const ns = "http://www.w3.org/2000/svg";
    const mkSvg = (w: number, h: number) => {
      const s = document.createElementNS(ns, "svg");
      s.setAttribute("viewBox", `0 0 ${w} ${h}`);
      s.setAttribute("width", "100%");
      s.classList.add("rl-svg");
      return s;
    };

    const regretEl = mkSvg(800, 230);
    this.regretSvg = d3.select(regretEl as SVGSVGElement);
    const barEl = mkSvg(800, 120);
    this.barSvg = d3.select(barEl as SVGSVGElement);

    this.timelineEl = document.createElement("div");
    this.timelineEl.className = "pull-timeline";
    this.timelineEl.style.marginTop = "8px";
    const tlLabel = document.createElement("div");
    tlLabel.className = "axis-label";
    tlLabel.style.fontFamily = "var(--rl-font-ui)";
    tlLabel.style.fontSize = "11px";
    tlLabel.style.color = "var(--rl-ink-faint)";
    tlLabel.style.margin = "10px 0 4px";
    tlLabel.textContent = "recent pulls (newest →)";

    this.counterEl = document.createElement("div");
    this.counterEl.className = "pick-reason";

    this.panel.body.append(
      this.buildControls(),
      regretEl,
      barEl,
      tlLabel,
      this.timelineEl,
      this.counterEl,
    );
    this.appendChild(this.panel.panel);

    this.loop = new StepLoop({
      baseIntervalMs: 55,
      onStep: () => this.stepOnce(),
      onFrame: () => this.draw(),
    });

    this.draw();
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
      this.syncPlayBtn();
    });

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.loop.pause();
      this.reset();
      this.draw();
      this.syncPlayBtn();
    });

    // epsilon slider
    const epsWrap = document.createElement("label");
    const epsVal = document.createElement("span");
    epsVal.className = "rl-mono";
    epsVal.textContent = this.epsilon.toFixed(2);
    const eps = document.createElement("input");
    eps.type = "range";
    eps.min = "0";
    eps.max = "0.5";
    eps.step = "0.01";
    eps.value = String(this.epsilon);
    eps.addEventListener("input", () => {
      this.epsilon = +eps.value;
      epsVal.textContent = this.epsilon.toFixed(2);
      this.loop.pause();
      this.reset();
      this.draw();
      this.syncPlayBtn();
    });
    epsWrap.append("ε ", eps, epsVal);

    // decay toggle
    const decayBtn = document.createElement("button");
    decayBtn.textContent = "schedule: constant";
    decayBtn.setAttribute("aria-pressed", "false");
    decayBtn.addEventListener("click", () => {
      this.decay = !this.decay;
      decayBtn.setAttribute("aria-pressed", String(this.decay));
      decayBtn.textContent = this.decay ? "schedule: decay 1/t" : "schedule: constant";
      eps.disabled = this.decay;
      this.loop.pause();
      this.reset();
      this.draw();
      this.syncPlayBtn();
    });

    // horizon
    const Twrap = document.createElement("label");
    Twrap.append("T ");
    const Tsel = document.createElement("select");
    for (const T of [100, 500, 1000, 5000, 10000]) {
      const o = document.createElement("option");
      o.value = String(T);
      o.textContent = String(T);
      if (T === this.horizon) o.selected = true;
      Tsel.appendChild(o);
    }
    Tsel.addEventListener("change", () => {
      this.horizon = +Tsel.value;
      this.loop.pause();
      this.reset();
      this.draw();
      this.syncPlayBtn();
    });
    Twrap.appendChild(Tsel);

    // speed
    const spWrap = document.createElement("label");
    spWrap.append("speed ");
    const sp = document.createElement("select");
    for (const s of [1, 2, 4, 8, 16, 32]) {
      const o = document.createElement("option");
      o.value = String(s);
      o.textContent = `${s}×`;
      if (s === 4) o.selected = true;
      sp.appendChild(o);
    }
    this.loop?.setSpeed(4);
    sp.addEventListener("change", () => this.loop.setSpeed(+sp.value));
    spWrap.appendChild(sp);

    // floor overlay
    const floorWrap = document.createElement("label");
    const floor = document.createElement("input");
    floor.type = "checkbox";
    floor.checked = this.overlayFloor;
    floor.addEventListener("change", () => {
      this.overlayFloor = floor.checked;
      this.draw();
    });
    floorWrap.append(floor, " Lai–Robbins floor");

    c.append(this.playBtn, stepBtn, resetBtn, epsWrap, decayBtn, Twrap, spWrap, floorWrap);
    // ensure speed default applied once loop exists
    queueMicrotask(() => this.loop?.setSpeed(+sp.value));
    return c;
  }

  private togglePlay(): void {
    if (this.t >= this.horizon) {
      this.reset();
      this.draw();
    }
    if (this.loop.isPlaying()) this.loop.pause();
    else this.loop.play();
    this.syncPlayBtn();
  }

  private syncPlayBtn(): void {
    this.playBtn.textContent = this.loop.isPlaying() ? "Pause" : this.t >= this.horizon ? "Replay" : "Run";
  }

  private draw(): void {
    this.drawRegret();
    this.drawBars();
    this.drawTimeline();
    const st = this.algo.state();
    const pctOpt = this.t ? ((st.N[this.optimalArm] / this.t) * 100).toFixed(1) : "—";
    this.counterEl.innerHTML = `running regret <strong>R_t = ${this.tracker.current().toFixed(1)}</strong> · % pulls on optimal arm: ${pctOpt}%`;
    this.panel.setStatus(`t=${this.t} / ${this.horizon}`);
  }

  private drawRegret(): void {
    const W = 800, H = 230, m = { top: 14, right: 16, bottom: 30, left: 48 };
    const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
    this.regretSvg.selectAll("*").remove();
    const g = this.regretSvg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain([0, this.horizon]).range([0, iw]);
    const floorMax = LR * Math.log(Math.max(2, this.horizon));
    const yMax = Math.max(this.regretHist[this.regretHist.length - 1] ?? 1, floorMax, 1) * 1.1;
    const y = d3.scaleLinear().domain([0, yMax]).range([ih, 0]);

    g.append("g").attr("class", "rl-axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("~s")));
    g.append("g").attr("class", "rl-axis").call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("class", "axis-label").attr("x", iw / 2).attr("y", ih + 26)
      .attr("text-anchor", "middle").text("rounds  t");
    g.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)")
      .attr("x", -ih / 2).attr("y", -36).attr("text-anchor", "middle").text("regret R_t");

    if (this.overlayFloor) {
      const fp = d3.range(1, this.horizon, Math.max(1, Math.floor(this.horizon / 300)))
        .map((t) => ({ t, v: LR * Math.log(t) }));
      g.append("path").datum(fp).attr("fill", "none").attr("stroke", "var(--rl-rule)")
        .attr("stroke-width", 1.2).attr("stroke-dasharray", "5 4")
        .attr("d", d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v)));
    }

    const stride = Math.max(1, Math.floor(this.regretHist.length / 500));
    const pts = this.regretHist
      .filter((_, i) => i % stride === 0 || i === this.regretHist.length - 1)
      .map((v, i) => ({ t: Math.min(i * stride, this.regretHist.length - 1), v }));
    g.append("path").datum(pts).attr("fill", "none").attr("stroke", "var(--rl-algo-greedy)")
      .attr("stroke-width", 2)
      .attr("d", d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v)));
  }

  private drawBars(): void {
    const W = 800, H = 120, m = { top: 16, right: 16, bottom: 22, left: 48 };
    const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
    this.barSvg.selectAll("*").remove();
    const g = this.barSvg.append("g").attr("transform", `translate(${m.left},${m.top})`);
    const st = this.algo.state();
    const freqs = st.N.map((n) => (this.t ? n / this.t : 0));

    const x = d3.scaleBand<number>().domain(d3.range(MEANS.length)).range([0, iw]).padding(0.4);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
    g.append("g").attr("class", "rl-axis").call(d3.axisLeft(y).ticks(3).tickFormat(d3.format(".0%")));
    g.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)")
      .attr("x", -ih / 2).attr("y", -36).attr("text-anchor", "middle").text("N_i / t");

    g.selectAll("rect").data(freqs).join("rect")
      .attr("x", (_, i) => x(i)!)
      .attr("width", x.bandwidth())
      .attr("y", (d) => y(d))
      .attr("height", (d) => ih - y(d))
      .attr("rx", 2)
      .attr("fill", (_, i) => armColor(i, i === this.optimalArm));
    g.selectAll("text.bl").data(freqs).join("text").attr("class", "bl annot")
      .attr("x", (_, i) => x(i)! + x.bandwidth() / 2).attr("y", ih + 16)
      .attr("text-anchor", "middle").text((_, i) => `arm ${i + 1}`);
  }

  private drawTimeline(): void {
    this.timelineEl.innerHTML = "";
    for (const arm of this.recent) {
      const tick = document.createElement("div");
      tick.className = "tick";
      tick.style.background = arm === this.optimalArm
        ? "var(--rl-algo-optimal)"
        : armColor(arm, false);
      this.timelineEl.appendChild(tick);
    }
  }
}

customElements.define("epsilon-greedy-explorer", EpsilonGreedyExplorer);
