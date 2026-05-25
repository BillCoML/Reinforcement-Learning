/**
 * V7 — Algorithm Battle Arena. THE CENTERPIECE.
 *
 * A configurable, in-browser bandit tournament. Left rail configures the
 * problem (K, reward family, per-arm means, horizon, seeds) and the algorithm
 * roster. The main canvas animates seed-averaged regret curves (±1σ bands) as
 * the simulation computes, with the Lai–Robbins floor overlaid. The bottom
 * mini-charts show each algorithm's pull distribution — the single clearest
 * window into *how* the algorithms differ. PNG export and URL-shareable seeds
 * round it out.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { ArenaSim, type AlgoSpec, type ArenaConfig, type AlgoResult, type RewardFamily } from "../bandits/arena-sim";
import { laiRobbinsConstant, mulberry32 } from "../bandits/stats";
import { loadRegretCurves } from "../lesson/data";

const PAL = {
  ink: "#1c1e22",
  muted: "#5a5d63",
  faint: "#8a8d93",
  border: "#d9d3c4",
  optimal: "#15803d",
  random: "#6b7280",
  greedy: "#b45309",
  customEps: "#d97706",
  ucb: "#0e7490",
  thompson: "#6d28d9",
  rule: "#1c1e22",
};

interface AlgoMeta {
  key: string;
  short: string; // URL code
  label: string;
  color: string;
  dash: string;
  make: (eps?: number) => AlgoSpec;
  defaultOn: boolean;
}

const ALGO_META: AlgoMeta[] = [
  { key: "random", short: "rand", label: "Random", color: PAL.random, dash: "", defaultOn: true,
    make: () => ({ key: "random", kind: "random", label: "Random" }) },
  { key: "eps001", short: "eps001", label: "ε-greedy(0.01)", color: PAL.greedy, dash: "4 3", defaultOn: true,
    make: () => ({ key: "eps001", kind: "epsilon", label: "ε-greedy(0.01)", epsilon: 0.01 }) },
  { key: "eps01", short: "eps01", label: "ε-greedy(0.10)", color: PAL.greedy, dash: "", defaultOn: true,
    make: () => ({ key: "eps01", kind: "epsilon", label: "ε-greedy(0.10)", epsilon: 0.1 }) },
  { key: "ucb1", short: "ucb", label: "UCB1", color: PAL.ucb, dash: "", defaultOn: true,
    make: () => ({ key: "ucb1", kind: "ucb1", label: "UCB1" }) },
  { key: "thompson", short: "ts", label: "Thompson", color: PAL.thompson, dash: "", defaultOn: true,
    make: () => ({ key: "thompson", kind: "thompson", label: "Thompson" }) },
  { key: "custom", short: "custom", label: "ε-greedy(custom)", color: PAL.customEps, dash: "1 3", defaultOn: false,
    make: (eps = 0.05) => ({ key: "custom", kind: "epsilon", label: `ε-greedy(${eps})`, epsilon: eps }) },
];

function armColorHex(i: number, isOpt: boolean): string {
  if (isOpt) return PAL.optimal;
  const ramp = ["#0e7490", "#4b8ba3", "#7aa3b5", "#a9bcc6", "#c9d3d8", "#dde3e6", "#eef1f2"];
  return ramp[i % ramp.length];
}

export class AlgorithmBattleArena extends HTMLElement {
  // config
  private K = 3;
  private means = [0.3, 0.5, 0.7];
  private family: RewardFamily = "bernoulli";
  private gaussianSigma = 0.3;
  private betaConcentration = 6;
  private T = 5000;
  private seeds = 100;
  private baseSeed = 42;
  private customEps = 0.05;
  private enabled: Record<string, boolean> = {};
  private showFloor = true;
  private speed = 8;

  private sim: ArenaSim | null = null;
  private running = false;
  private raf = 0;
  private initialResults: AlgoResult[] | null = null; // offline fallback display
  private usingOffline = false;

  private panel!: PanelHandle;
  private regretSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private pullWrap!: HTMLElement;
  private statsEl!: HTMLElement;
  private meansWrap!: HTMLElement;
  private runBtn!: HTMLButtonElement;
  private tooltip!: HTMLElement;

  private readonly RW = 600;
  private readonly RH = 300;

  connectedCallback(): void {
    for (const m of ALGO_META) this.enabled[m.key] = m.defaultOn;
    this.readURL();
    this.render();
    void this.loadInitial();
  }
  disconnectedCallback(): void {
    cancelAnimationFrame(this.raf);
  }

  // ---- URL state ----
  private readURL(): void {
    const u = new URLSearchParams(location.search);
    if (u.has("seed")) this.baseSeed = +u.get("seed")! || 42;
    if (u.has("algos")) {
      const codes = new Set(u.get("algos")!.split(","));
      for (const m of ALGO_META) this.enabled[m.key] = codes.has(m.short);
    }
  }
  private writeURL(): void {
    const codes = ALGO_META.filter((m) => this.enabled[m.key]).map((m) => m.short);
    const u = new URLSearchParams(location.search);
    u.set("seed", String(this.baseSeed));
    u.set("algos", codes.join(","));
    history.replaceState(null, "", `${location.pathname}?${u.toString()}${location.hash}`);
  }

  private selectedSpecs(): AlgoSpec[] {
    return ALGO_META.filter((m) => this.enabled[m.key]).map((m) => m.make(this.customEps));
  }
  private metaFor(key: string): AlgoMeta {
    return ALGO_META.find((m) => m.key === key)!;
  }
  private get optArm(): number {
    return this.means.indexOf(Math.max(...this.means));
  }
  private get lrConst(): number {
    return this.family === "bernoulli" ? laiRobbinsConstant(this.means) : laiRobbinsConstant(this.means);
  }
  private isDefaultConfig(): boolean {
    return (
      this.K === 3 &&
      this.family === "bernoulli" &&
      Math.abs(this.means[0] - 0.3) < 1e-9 &&
      Math.abs(this.means[1] - 0.5) < 1e-9 &&
      Math.abs(this.means[2] - 0.7) < 1e-9
    );
  }

  // ---- initial offline display ----
  private async loadInitial(): Promise<void> {
    if (!this.isDefaultConfig()) {
      this.drawAll([]);
      return;
    }
    try {
      const data = await loadRegretCurves();
      const stride = Math.max(1, Math.floor(data.T / 600));
      this.initialResults = ALGO_META.filter((m) => this.enabled[m.key] && data.algos[m.key]).map(
        (m) => {
          const c = data.algos[m.key];
          const mean: number[] = [];
          const std: number[] = [];
          for (let i = 0; i < c.mean.length; i += stride) {
            mean.push(c.mean[i]);
            std.push(c.std[i]);
          }
          return {
            key: m.key,
            mean,
            std,
            pullFrac: [],
            finalRegret: c.final,
            pctOptimal: 0,
            regretOverLogT: c.final / Math.log(data.T),
          } as AlgoResult;
        },
      );
      this.usingOffline = true;
      this.drawAll(this.initialResults, data.T, stride);
    } catch {
      this.drawAll([]);
    }
  }

  // ---- layout ----
  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "algorithm-battle-arena",
      arena: true,
      heavy: true,
      mobileNotice: "The Battle Arena runs a live tournament — view on a wider screen.",
    });

    const grid = document.createElement("div");
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = "210px minmax(0,1fr) 190px";
    grid.style.gap = "16px";
    grid.style.alignItems = "start";

    grid.append(this.buildLeftRail(), this.buildMain(), this.buildRightRail());
    this.panel.body.append(grid, this.buildBottomControls());
    this.appendChild(this.panel.panel);
    this.rebuildMeanSliders();
  }

  private buildLeftRail(): HTMLElement {
    const rail = document.createElement("div");
    rail.className = "arena-rail";

    const cfgTitle = document.createElement("div");
    cfgTitle.className = "arena-rail__title";
    cfgTitle.textContent = "problem";

    // K slider
    const kWrap = document.createElement("label");
    kWrap.className = "arena-field";
    const kVal = document.createElement("span");
    kVal.className = "rl-mono";
    kVal.textContent = String(this.K);
    const k = document.createElement("input");
    k.type = "range"; k.min = "2"; k.max = "8"; k.step = "1"; k.value = String(this.K);
    k.addEventListener("input", () => {
      this.K = +k.value; kVal.textContent = k.value;
      this.regenerateMeans(false);
      this.rebuildMeanSliders();
      this.resetSim();
    });
    kWrap.append(this.fieldLabel(`arms K`, kVal), k);

    // family
    const famWrap = document.createElement("label");
    famWrap.className = "arena-field";
    const fam = document.createElement("select");
    for (const f of ["bernoulli", "gaussian", "beta"] as RewardFamily[]) {
      const o = document.createElement("option"); o.value = f; o.textContent = f; fam.appendChild(o);
    }
    fam.addEventListener("change", () => {
      this.family = fam.value as RewardFamily;
      this.resetSim();
    });
    famWrap.append(this.fieldLabel("rewards"), fam);

    // means container
    this.meansWrap = document.createElement("div");
    this.meansWrap.className = "arena-means";

    // randomize
    const rand = document.createElement("button");
    rand.textContent = "Randomize problem";
    rand.addEventListener("click", () => {
      this.regenerateMeans(true);
      this.rebuildMeanSliders();
      this.resetSim();
    });

    // horizon
    const tWrap = document.createElement("label");
    tWrap.className = "arena-field";
    const tSel = document.createElement("select");
    for (const T of [100, 500, 1000, 5000, 20000]) {
      const o = document.createElement("option"); o.value = String(T); o.textContent = String(T);
      if (T === this.T) o.selected = true; tSel.appendChild(o);
    }
    tSel.addEventListener("change", () => { this.T = +tSel.value; this.resetSim(); });
    tWrap.append(this.fieldLabel("horizon T"), tSel);

    // seeds
    const sWrap = document.createElement("label");
    sWrap.className = "arena-field";
    const sSel = document.createElement("select");
    for (const S of [1, 25, 100, 200]) {
      const o = document.createElement("option"); o.value = String(S); o.textContent = `${S}`;
      if (S === this.seeds) o.selected = true; sSel.appendChild(o);
    }
    sSel.addEventListener("change", () => { this.seeds = +sSel.value; this.resetSim(); });
    sWrap.append(this.fieldLabel("seeds avg"), sSel);

    // algorithm selector
    const algTitle = document.createElement("div");
    algTitle.className = "arena-rail__title";
    algTitle.style.marginTop = "14px";
    algTitle.textContent = "algorithms";

    const algBox = document.createElement("div");
    algBox.className = "arena-algos";
    for (const m of ALGO_META) {
      const row = document.createElement("label");
      row.className = "arena-algo-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = this.enabled[m.key];
      cb.addEventListener("change", () => {
        this.enabled[m.key] = cb.checked;
        this.writeURL();
        this.resetSim();
      });
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = m.color;
      if (m.dash) sw.style.opacity = "0.7";
      row.append(cb, sw, document.createTextNode(m.label));
      algBox.appendChild(row);

      if (m.key === "custom") {
        const epsR = document.createElement("input");
        epsR.type = "range"; epsR.min = "0"; epsR.max = "0.5"; epsR.step = "0.01";
        epsR.value = String(this.customEps);
        epsR.className = "arena-custom-eps";
        epsR.addEventListener("input", () => {
          this.customEps = +epsR.value;
          if (this.enabled.custom) this.resetSim();
        });
        algBox.appendChild(epsR);
      }
    }

    rail.append(cfgTitle, kWrap, famWrap, this.meansWrap, rand, tWrap, sWrap, algTitle, algBox);
    return rail;
  }

  private fieldLabel(text: string, valueEl?: HTMLElement): HTMLElement {
    const l = document.createElement("span");
    l.className = "arena-field__label";
    l.textContent = text;
    if (valueEl) {
      l.append(" ");
      l.appendChild(valueEl);
    }
    return l;
  }

  private rebuildMeanSliders(): void {
    this.meansWrap.innerHTML = "";
    const lbl = document.createElement("div");
    lbl.className = "arena-field__label";
    lbl.textContent = "arm means μ";
    this.meansWrap.appendChild(lbl);
    const opt = this.optArm;
    for (let i = 0; i < this.K; i++) {
      const row = document.createElement("div");
      row.className = "arena-mean-row";
      const tag = document.createElement("span");
      tag.className = "rl-mono arena-mean-tag";
      tag.style.color = i === opt ? PAL.optimal : PAL.muted;
      tag.textContent = `${i + 1}`;
      const r = document.createElement("input");
      r.type = "range"; r.min = "0.02"; r.max = "0.98"; r.step = "0.01";
      r.value = String(this.means[i]);
      const v = document.createElement("span");
      v.className = "rl-mono arena-mean-val";
      v.textContent = this.means[i].toFixed(2);
      r.addEventListener("input", () => {
        this.means[i] = +r.value;
        v.textContent = this.means[i].toFixed(2);
        this.refreshMeanColors();
        this.resetSim();
      });
      row.append(tag, r, v);
      this.meansWrap.appendChild(row);
    }
  }
  private refreshMeanColors(): void {
    const opt = this.optArm;
    this.meansWrap.querySelectorAll<HTMLElement>(".arena-mean-tag").forEach((tag, i) => {
      tag.style.color = i === opt ? PAL.optimal : PAL.muted;
    });
  }

  private regenerateMeans(randomize: boolean): void {
    if (!randomize && this.means.length === this.K) return;
    if (randomize) {
      const rng = mulberry32((Date.now() & 0xffffff) >>> 0);
      this.means = Array.from({ length: this.K }, () => +(0.1 + rng() * 0.8).toFixed(2));
    } else {
      // resize keeping spread
      const next = Array.from({ length: this.K }, (_, i) => +(0.2 + (0.6 * i) / Math.max(1, this.K - 1)).toFixed(2));
      this.means = next;
    }
  }

  private buildMain(): HTMLElement {
    const main = document.createElement("div");
    main.style.minWidth = "0";

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${this.RW} ${this.RH}`);
    svg.setAttribute("width", "100%");
    svg.classList.add("rl-svg");
    wrap.appendChild(svg);
    this.regretSvg = d3.select(svg as SVGSVGElement);

    this.tooltip = document.createElement("div");
    this.tooltip.className = "rl-tooltip";
    wrap.appendChild(this.tooltip);

    const pullTitle = document.createElement("div");
    pullTitle.className = "arena-subtitle";
    pullTitle.textContent = "pull distribution — how each algorithm spends its budget";

    this.pullWrap = document.createElement("div");
    this.pullWrap.className = "arena-pulls";

    main.append(wrap, pullTitle, this.pullWrap);
    return main;
  }

  private buildRightRail(): HTMLElement {
    const rail = document.createElement("div");
    rail.className = "arena-rail";
    const title = document.createElement("div");
    title.className = "arena-rail__title";
    title.textContent = "results";
    this.statsEl = document.createElement("div");
    this.statsEl.className = "stats-readout";
    rail.append(title, this.statsEl);
    return rail;
  }

  private buildBottomControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";
    c.style.marginTop = "16px";

    this.runBtn = document.createElement("button");
    this.runBtn.className = "primary";
    this.runBtn.textContent = "Run";
    this.runBtn.addEventListener("click", () => this.toggleRun());

    const stepBtn = document.createElement("button");
    stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => this.stepChunk());

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => this.resetSim());

    const spWrap = document.createElement("label");
    spWrap.append("speed ");
    const sp = document.createElement("select");
    for (const s of [1, 2, 4, 8, 16, 32, 64]) {
      const o = document.createElement("option"); o.value = String(s); o.textContent = `${s}×`;
      if (s === this.speed) o.selected = true; sp.appendChild(o);
    }
    sp.addEventListener("change", () => (this.speed = +sp.value));
    spWrap.appendChild(sp);

    const floorWrap = document.createElement("label");
    const fl = document.createElement("input");
    fl.type = "checkbox"; fl.checked = this.showFloor;
    fl.addEventListener("change", () => { this.showFloor = fl.checked; this.renderResults(); });
    floorWrap.append(fl, " Lai–Robbins floor");

    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export PNG";
    exportBtn.addEventListener("click", () => this.exportPNG());

    c.append(this.runBtn, stepBtn, resetBtn, spWrap, floorWrap, exportBtn);
    return c;
  }

  // ---- simulation control ----
  private buildConfig(): ArenaConfig {
    return {
      K: this.K,
      means: [...this.means],
      family: this.family,
      gaussianSigma: this.gaussianSigma,
      betaConcentration: this.betaConcentration,
      T: this.T,
      seeds: this.seeds,
      baseSeed: this.baseSeed,
      algos: this.selectedSpecs(),
    };
  }

  private toggleRun(): void {
    if (this.running) {
      this.running = false;
      cancelAnimationFrame(this.raf);
      this.syncRun();
      return;
    }
    if (!this.sim || this.sim.done) {
      this.sim = new ArenaSim(this.buildConfig());
      this.usingOffline = false;
    }
    this.running = true;
    this.syncRun();
    const loop = () => {
      if (!this.running || !this.sim) return;
      this.sim.advance(this.sim.frontier + this.speed * 8, 12);
      this.renderResults();
      if (this.sim.done) {
        this.running = false;
        this.syncRun();
        return;
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  private stepChunk(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    if (!this.sim || this.sim.done) {
      this.sim = new ArenaSim(this.buildConfig());
      this.usingOffline = false;
    }
    this.sim.advance(this.sim.frontier + Math.max(1, Math.floor(this.T / 50)), 1e9);
    this.renderResults();
    this.syncRun();
  }

  private resetSim(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.sim = null;
    this.usingOffline = false;
    this.initialResults = null;
    this.writeURL();
    if (this.isDefaultConfig()) void this.loadInitial();
    else this.drawAll([]);
    this.syncRun();
  }

  private syncRun(): void {
    this.runBtn.textContent = this.running ? "Pause" : this.sim && this.sim.done ? "Replay" : "Run";
  }

  // ---- rendering ----
  private renderResults(): void {
    if (!this.sim) return;
    const res = this.sim.results();
    this.drawAll(res, this.T, 1);
    this.panel.setStatus(`t=${this.sim.frontier} / ${this.T} · ${this.seeds} seeds`);
  }

  private drawAll(results: AlgoResult[], T = this.T, stride = 1): void {
    this.drawRegret(results, T, stride);
    this.drawPulls(results);
    this.drawStats(results);
    if (!this.sim) {
      this.panel.setStatus(
        this.usingOffline ? `pre-computed averages · press Run` : `configure, then press Run`,
      );
    }
  }

  private drawRegret(results: AlgoResult[], T: number, stride: number): void {
    const m = { top: 14, right: 14, bottom: 30, left: 46 };
    const iw = this.RW - m.left - m.right;
    const ih = this.RH - m.top - m.bottom;
    this.regretSvg.selectAll("*").remove();
    const g = this.regretSvg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain([0, T]).range([0, iw]);
    let yMax = 1;
    for (const r of results) {
      const last = r.mean.length - 1;
      if (last >= 0) yMax = Math.max(yMax, r.mean[last] + r.std[last]);
    }
    if (this.showFloor) yMax = Math.max(yMax, this.lrConst * Math.log(Math.max(2, T)));
    const y = d3.scaleLinear().domain([0, yMax * 1.08]).range([ih, 0]).nice();

    g.append("g").attr("class", "rl-axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("~s")));
    g.append("g").attr("class", "rl-axis").call(d3.axisLeft(y).ticks(5));
    g.append("text").attr("class", "axis-label").attr("x", iw / 2).attr("y", ih + 26)
      .attr("text-anchor", "middle").attr("fill", PAL.muted).text("rounds  t");
    g.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)")
      .attr("x", -ih / 2).attr("y", -34).attr("text-anchor", "middle").attr("fill", PAL.muted)
      .text("cumulative regret  R_t");

    const xi = (i: number) => x(i * stride);

    // floor
    if (this.showFloor) {
      const fp = d3.range(1, T, Math.max(1, Math.floor(T / 300))).map((t) => ({ t, v: this.lrConst * Math.log(t) }));
      g.append("path").datum(fp).attr("fill", "none").attr("stroke", PAL.rule)
        .attr("stroke-width", 1.2).attr("stroke-dasharray", "5 4")
        .attr("d", d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v)));
    }

    for (const r of results) {
      const meta = this.metaFor(r.key);
      // std band
      const band = d3.area<number>()
        .x((_, i) => xi(i))
        .y0((d, i) => y(Math.max(0, d - r.std[i])))
        .y1((d, i) => y(d + r.std[i]));
      g.append("path").datum(r.mean).attr("fill", meta.color).attr("fill-opacity", 0.12)
        .attr("d", band);
      // mean line
      g.append("path").datum(r.mean).attr("fill", "none").attr("stroke", meta.color)
        .attr("stroke-width", 2).attr("stroke-dasharray", meta.dash || null)
        .attr("d", d3.line<number>().x((_, i) => xi(i)).y((d) => y(d)));
    }

    // hover interaction
    const tip = this.tooltip;
    const focus = g.append("line").attr("y1", 0).attr("y2", ih).attr("stroke", PAL.faint)
      .attr("stroke-width", 1).attr("opacity", 0);
    g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "transparent")
      .style("cursor", "crosshair")
      .on("mousemove", (ev: MouseEvent) => {
        if (!results.length) return;
        const [mx] = d3.pointer(ev);
        const t = Math.round(x.invert(mx));
        focus.attr("x1", x(t)).attr("x2", x(t)).attr("opacity", 1);
        const lines = results.map((r) => {
          const idx = Math.min(r.mean.length - 1, Math.floor(t / stride));
          const meta = this.metaFor(r.key);
          return idx >= 0 ? `${meta.label}: ${r.mean[idx].toFixed(1)}` : "";
        }).filter(Boolean);
        tip.textContent = `t=${t}\n` + lines.join("\n");
        tip.style.opacity = "1";
        tip.style.left = `${m.left + x(t) + 12}px`;
        tip.style.top = `${m.top + 4}px`;
      })
      .on("mouseleave", () => { focus.attr("opacity", 0); tip.style.opacity = "0"; });
  }

  private drawPulls(results: AlgoResult[]): void {
    this.pullWrap.innerHTML = "";
    const opt = this.optArm;
    const withPulls = results.filter((r) => r.pullFrac.length);
    if (!withPulls.length) {
      const note = document.createElement("div");
      note.className = "arena-subtitle";
      note.style.opacity = "0.6";
      note.textContent = this.usingOffline
        ? "(run the live simulation to see pull distributions)"
        : "";
      this.pullWrap.appendChild(note);
      return;
    }
    for (const r of withPulls) {
      const meta = this.metaFor(r.key);
      const row = document.createElement("div");
      row.className = "arena-pull-row";
      const label = document.createElement("span");
      label.className = "arena-pull-label";
      label.style.color = meta.color;
      label.textContent = meta.label;
      const bar = document.createElement("div");
      bar.className = "arena-pull-bar";
      r.pullFrac.forEach((f, i) => {
        const seg = document.createElement("div");
        seg.className = "arena-pull-seg";
        seg.style.width = `${(f * 100).toFixed(2)}%`;
        seg.style.background = armColorHex(i, i === opt);
        seg.title = `arm ${i + 1}${i === opt ? " (optimal)" : ""}: ${(f * 100).toFixed(1)}%`;
        bar.appendChild(seg);
      });
      const pct = document.createElement("span");
      pct.className = "arena-pull-pct rl-mono";
      pct.textContent = `${r.pctOptimal.toFixed(0)}% opt`;
      row.append(label, bar, pct);
      this.pullWrap.appendChild(row);
    }
  }

  private drawStats(results: AlgoResult[]): void {
    const rows: string[] = [];
    for (const r of results) {
      const meta = this.metaFor(r.key);
      rows.push(
        `<div class="row"><span class="k"><span class="swatch" style="background:${meta.color}"></span>${meta.label}</span><span>${r.finalRegret.toFixed(1)}</span></div>`,
      );
      if (r.pullFrac.length) {
        rows.push(
          `<div class="row"><span class="k">R/log T</span><span>${r.regretOverLogT.toFixed(2)}</span></div>`,
        );
      }
    }
    const tNow = this.sim ? this.sim.frontier : this.T;
    rows.push(
      `<div class="row"><span class="k">LR · log T</span><span>${(this.lrConst * Math.log(Math.max(2, tNow))).toFixed(1)}</span></div>`,
    );
    this.statsEl.innerHTML = rows.join("") || `<div class="row"><span class="k">no algorithms selected</span></div>`;
  }

  // ---- PNG export ----
  private exportPNG(): void {
    const node = this.regretSvg.node();
    if (!node) return;
    const clone = node.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(this.RW));
    clone.setAttribute("height", String(this.RH));
    const xml = new XMLSerializer().serializeToString(clone);
    const svg64 = btoa(unescape(encodeURIComponent(xml)));
    const img = new Image();
    img.onload = () => {
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = this.RW * scale;
      canvas.height = this.RH * scale;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(scale, scale);
      ctx.fillStyle = "#faf8f3";
      ctx.fillRect(0, 0, this.RW, this.RH);
      ctx.drawImage(img, 0, 0);
      const a = document.createElement("a");
      a.download = `battle-arena-seed${this.baseSeed}.png`;
      a.href = canvas.toDataURL("image/png");
      a.click();
    };
    img.src = "data:image/svg+xml;base64," + svg64;
  }
}

customElements.define("algorithm-battle-arena", AlgorithmBattleArena);
