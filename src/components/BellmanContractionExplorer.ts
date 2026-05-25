/**
 * V5 — Bellman Contraction Explorer. CENTERPIECE.
 *
 * Three synchronized panels:
 *   A (left): Two 3×3 gridworld renderers showing V1 and V2 side by side.
 *             Click cells to edit. "Randomize V2" button.
 *   B (center): Live readouts — sup-dist, empirical ratio, iteration counter.
 *              Apply T^π or T^* buttons; γ slider; mode toggle.
 *   C (right): Log-scale convergence plot showing ‖V1^(k) − V2^(k)‖∞ vs k,
 *              with a dashed γ^k · d0 bound line.
 *
 * Imports everything from Lesson 2 — no new MDP code.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { buildGridworld, uniformPolicy } from "../mdp/gridworld";
import { bellmanExpectationBackup } from "../mdp/policy-evaluation";
import { bellmanOptimalityBackup } from "../mdp/value-iteration";
import { supDist } from "../contractions/ops";
import type { MDP } from "../mdp/types";

const PLOT_W = 300, PLOT_H = 280;

export class BellmanContractionExplorer extends HTMLElement {
  private gamma = 0.9;
  private mode: "expectation" | "optimality" = "expectation";
  private mdp!: MDP;
  private V1: number[] = new Array(9).fill(0);
  private V2: number[] = new Array(9).fill(2);
  private history: Array<{ d: number }> = [];
  private playing = false;
  private playInterval = 0;
  private renderer1!: GridworldRenderer;
  private renderer2!: GridworldRenderer;
  private plotSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private distEl!: HTMLElement;
  private ratioEl!: HTMLElement;
  private iterEl!: HTMLElement;
  private statusEl!: HTMLElement;

  connectedCallback() { this.build(); }

  private rebuild() {
    const uniform = uniformPolicy(this.mdp);
    return { uniform };
  }

  private applyBackup(V: number[]): number[] {
    const { uniform } = this.rebuild();
    if (this.mode === "expectation") {
      return bellmanExpectationBackup(this.mdp, uniform, V);
    } else {
      return bellmanOptimalityBackup(this.mdp, V);
    }
  }

  private build() {
    this.innerHTML = "";
    this.mdp = buildGridworld({ slippery: false, gamma: this.gamma });
    this.V1 = new Array(9).fill(0);
    this.V2 = new Array(9).fill(2);
    this.history = [{ d: supDist(this.V1, this.V2) }];

    const { panel, body } = createPanel({ id: "v5-bellman-contraction-explorer", arena: true });
    this.appendChild(panel);

    const mainRow = document.createElement("div");
    mainRow.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start";
    body.appendChild(mainRow);

    // ---- Panel A: two gridworlds ----
    const panelA = document.createElement("div");
    panelA.style.cssText = "display:flex;flex-direction:column;gap:8px";
    mainRow.appendChild(panelA);

    const makeGrid = (label: string, V: number[], idx: 1 | 2) => {
      const wrap = document.createElement("div");
      const lbl = Object.assign(document.createElement("div"), {
        style: "font-size:12px;color:var(--rl-ink-faint);margin-bottom:4px;font-family:var(--rl-font-ui)",
        textContent: label,
      });
      wrap.appendChild(lbl);
      const container = document.createElement("div");
      wrap.appendChild(container);
      const renderer = new GridworldRenderer(container, {
        mdp: this.mdp,
        valueFn: V,
        showValues: true,
        cellPx: 72,
        onCellClick: (s: number) => {
          if (this.mdp.terminals[s]) return;
          const arr = idx === 1 ? this.V1 : this.V2;
          arr[s] = parseFloat(prompt(`New V${idx}(s${s}) [e.g. 0-2]:`, String(arr[s].toFixed(2))) ?? String(arr[s]));
          this.refresh();
        },
      });
      return { wrap, renderer };
    };

    const { wrap: w1, renderer: r1 } = makeGrid("V₁ (click cell to edit)", this.V1, 1);
    const { wrap: w2, renderer: r2 } = makeGrid("V₂ (click cell to edit)", this.V2, 2);
    this.renderer1 = r1;
    this.renderer2 = r2;
    panelA.appendChild(w1);
    panelA.appendChild(w2);

    const randBtn = document.createElement("button");
    randBtn.className = "rl-btn";
    randBtn.textContent = "Randomize V₂";
    randBtn.addEventListener("click", () => {
      this.V2 = this.mdp.nS ? Array.from({ length: this.mdp.nS }, (_, s) =>
        this.mdp.terminals[s] ? 0 : Math.random() * 4 - 1) : new Array(9).fill(0);
      this.history = [{ d: supDist(this.V1, this.V2) }];
      this.refresh();
    });
    panelA.appendChild(randBtn);

    const resetBtn = document.createElement("button");
    resetBtn.className = "rl-btn";
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.V1 = new Array(9).fill(0);
      this.V2 = new Array(9).fill(2);
      this.history = [{ d: supDist(this.V1, this.V2) }];
      this.refresh();
    });
    panelA.appendChild(resetBtn);

    // ---- Panel B: center readouts ----
    const panelB = document.createElement("div");
    panelB.style.cssText = "min-width:200px;display:flex;flex-direction:column;gap:8px;padding:8px";
    mainRow.appendChild(panelB);

    const makeReadout = (label: string): HTMLElement => {
      const div = document.createElement("div");
      div.style.cssText = "background:var(--rl-surface-2);border-radius:8px;padding:8px 12px";
      const lbl = Object.assign(document.createElement("div"), {
        style: "font-size:10px;color:var(--rl-ink-faint);font-family:var(--rl-font-ui);margin-bottom:2px",
        textContent: label,
      });
      const val = Object.assign(document.createElement("div"), {
        style: "font-size:20px;font-family:var(--rl-font-mono);font-weight:600;color:var(--rl-ink)",
      });
      div.appendChild(lbl);
      div.appendChild(val);
      panelB.appendChild(div);
      return val;
    };

    this.iterEl = makeReadout("Iteration k");
    this.distEl = makeReadout("‖V₁ − V₂‖∞ (sup-distance)");
    this.ratioEl = makeReadout("empirical ratio / γ");

    // Mode toggle
    const modeRow = document.createElement("div");
    modeRow.style.cssText = "display:flex;gap:4px;margin-top:4px";
    panelB.appendChild(modeRow);

    const makeMode = (label: string, m: "expectation" | "optimality") => {
      const btn = document.createElement("button");
      btn.className = "rl-btn";
      btn.style.fontSize = "11px";
      btn.textContent = label;
      btn.addEventListener("click", () => {
        this.mode = m;
        modeRow.querySelectorAll("button").forEach(b => (b as HTMLButtonElement).style.fontWeight = "400");
        btn.style.fontWeight = "700";
        this.history = [{ d: supDist(this.V1, this.V2) }];
        this.refresh();
      });
      if (m === this.mode) btn.style.fontWeight = "700";
      modeRow.appendChild(btn);
    };
    makeMode("T^π  (expectation)", "expectation");
    makeMode("T*  (optimality)", "optimality");

    // γ slider
    const gammaRow = document.createElement("div");
    gammaRow.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px;margin-top:4px";
    const gLbl = Object.assign(document.createElement("span"), { style: "color:var(--rl-ink-faint)", innerHTML: "γ:" });
    const gSl = document.createElement("input");
    gSl.type = "range"; gSl.min = "0.1"; gSl.max = "0.99"; gSl.step = "0.01"; gSl.value = String(this.gamma);
    gSl.style.width = "100px";
    const gVal = Object.assign(document.createElement("span"), {
      className: "rl-mono", style: "font-size:12px", textContent: String(this.gamma),
    });
    gSl.addEventListener("input", () => {
      this.gamma = parseFloat(gSl.value);
      gVal.textContent = this.gamma.toFixed(2);
      this.mdp = buildGridworld({ slippery: false, gamma: this.gamma });
      this.history = [{ d: supDist(this.V1, this.V2) }];
      this.refresh();
    });
    gammaRow.appendChild(gLbl); gammaRow.appendChild(gSl); gammaRow.appendChild(gVal);
    panelB.appendChild(gammaRow);

    // Step / Play / Reset buttons
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-top:8px";
    panelB.appendChild(btnRow);

    const stepBtn = document.createElement("button");
    stepBtn.className = "rl-btn"; stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => this.doStep());
    btnRow.appendChild(stepBtn);

    const playBtn = document.createElement("button");
    playBtn.className = "rl-btn"; playBtn.textContent = "Play";
    playBtn.addEventListener("click", () => {
      if (this.playing) {
        clearInterval(this.playInterval); this.playing = false; playBtn.textContent = "Play";
      } else {
        this.playing = true; playBtn.textContent = "Pause";
        this.playInterval = window.setInterval(() => { this.doStep(); }, 800);
      }
    });
    btnRow.appendChild(playBtn);

    const speedSl = document.createElement("input");
    speedSl.type = "range"; speedSl.min = "100"; speedSl.max = "2000"; speedSl.step = "100"; speedSl.value = "800";
    speedSl.style.width = "70px";
    speedSl.addEventListener("input", () => {
      if (this.playing) {
        clearInterval(this.playInterval);
        this.playInterval = window.setInterval(() => this.doStep(), parseInt(speedSl.value));
      }
    });
    btnRow.appendChild(Object.assign(document.createElement("span"), { style: "font-size:10px;color:var(--rl-ink-faint);align-self:center", textContent: "speed" }));
    btnRow.appendChild(speedSl);

    this.statusEl = Object.assign(document.createElement("div"), {
      style: "font-size:11px;color:var(--rl-ink-faint);margin-top:4px",
    });
    panelB.appendChild(this.statusEl);

    // ---- Panel C: convergence plot ----
    const panelC = document.createElement("div");
    mainRow.appendChild(panelC);

    const plotLabel = Object.assign(document.createElement("div"), {
      style: "font-size:11px;color:var(--rl-ink-faint);margin-bottom:4px",
      textContent: "Convergence (log scale) — actual vs γᵏ bound",
    });
    panelC.appendChild(plotLabel);

    const ns = "http://www.w3.org/2000/svg";
    const plotEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    plotEl.setAttribute("viewBox", `0 0 ${PLOT_W} ${PLOT_H}`);
    plotEl.setAttribute("width", String(PLOT_W));
    plotEl.setAttribute("height", String(PLOT_H));
    plotEl.classList.add("rl-svg");
    panelC.appendChild(plotEl);
    this.plotSvg = d3.select(plotEl);

    this.refresh();
  }

  private doStep() {
    this.V1 = this.applyBackup(this.V1);
    this.V2 = this.applyBackup(this.V2);
    this.history.push({ d: supDist(this.V1, this.V2) });
    this.refresh();
  }

  private refresh() {
    const k = this.history.length - 1;
    const d = this.history[k].d;
    const d0 = this.history[0].d;
    const prevD = k > 0 ? this.history[k - 1].d : null;
    const ratio = (prevD != null && prevD > 1e-12) ? d / prevD : null;

    this.iterEl.textContent = String(k);
    this.distEl.textContent = d.toFixed(6);

    if (ratio !== null) {
      const ok = ratio <= this.gamma + 1e-9;
      this.ratioEl.textContent = ratio.toFixed(4);
      this.ratioEl.style.color = ok ? "var(--contr-ok)" : "var(--contr-warn)";
      const parentDiv = this.ratioEl.parentElement!;
      parentDiv.style.background = ok ? "var(--contr-ok-bg)" : "var(--contr-warn-bg)";
      this.statusEl.textContent = ok
        ? `ratio ≤ γ = ${this.gamma.toFixed(2)} ✓ theorem holds`
        : `⚠ ratio > γ — implementation error?`;
    } else {
      this.ratioEl.textContent = "—";
      this.ratioEl.style.color = "var(--rl-ink)";
    }

    this.renderer1.update({ valueFn: this.V1, mdp: this.mdp });
    this.renderer2.update({ valueFn: this.V2, mdp: this.mdp });
    this.drawPlot(d0);
  }

  private drawPlot(d0: number) {
    const PP = { l: 48, r: 16, t: 16, b: 36 };
    const pW = PLOT_W - PP.l - PP.r;
    const pH = PLOT_H - PP.t - PP.b;
    this.plotSvg.selectAll("*").remove();

    const n = this.history.length;
    if (n < 2) {
      this.plotSvg.append("text").attr("x", PLOT_W / 2).attr("y", PLOT_H / 2)
        .attr("text-anchor", "middle").attr("class", "annot")
        .attr("fill", "var(--rl-ink-faint)").text("Press Step to start");
      return;
    }

    const ks = d3.range(n);
    const actuals = this.history.map(h => Math.max(h.d, 1e-10));
    const bounds = ks.map(k => Math.max(Math.pow(this.gamma, k) * d0, 1e-10));

    const yMax = Math.max(d0 * 1.2, 1e-9);
    const yMin = Math.min(...actuals) * 0.5;

    const xSc = d3.scaleLinear([0, n - 1], [PP.l, PP.l + pW]);
    const ySc = d3.scaleLog([Math.max(yMin, 1e-10), yMax], [PP.t + pH, PP.t]);

    const xAxis = d3.axisBottom(xSc).ticks(Math.min(n - 1, 8)).tickFormat(d3.format("d"));
    const yAxis = d3.axisLeft(ySc).ticks(5, ".1e");

    this.plotSvg.append("g").attr("transform", `translate(0,${PP.t + pH})`).call(xAxis)
      .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
    this.plotSvg.append("g").attr("transform", `translate(${PP.l},0)`).call(yAxis)
      .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "9px");
    this.plotSvg.selectAll("line,path").attr("stroke", "var(--rl-border)");

    this.plotSvg.append("text").attr("x", PP.l + pW / 2).attr("y", PLOT_H - 4)
      .attr("text-anchor", "middle").attr("class", "annot")
      .attr("fill", "var(--rl-ink-faint)").text("iteration k");
    this.plotSvg.append("text").attr("x", 10).attr("y", PP.t + pH / 2)
      .attr("text-anchor", "middle")
      .attr("transform", `rotate(-90,10,${PP.t + pH / 2})`)
      .attr("class", "annot").attr("fill", "var(--rl-ink-faint)").text("‖V₁ − V₂‖∞");

    const safe = (y: number) => { try { const v = ySc(y); return isFinite(v) ? v : PP.t + pH; } catch { return PP.t + pH; } };

    // Bound line (dashed)
    const boundPath = ks.map(k => [xSc(k), safe(bounds[k])]);
    if (boundPath.length > 1) {
      this.plotSvg.append("path")
        .attr("d", "M" + boundPath.map(p => p.join(",")).join("L"))
        .attr("fill", "none").attr("stroke", "var(--contr-bound)")
        .attr("stroke-width", 1.5).attr("stroke-dasharray", "6 3").attr("opacity", 0.6);
    }

    // Actual line
    const actualPath = ks.map(k => [xSc(k), safe(actuals[k])]);
    if (actualPath.length > 1) {
      this.plotSvg.append("path")
        .attr("d", "M" + actualPath.map(p => p.join(",")).join("L"))
        .attr("fill", "none").attr("stroke", "var(--contr-input)")
        .attr("stroke-width", 2.2);
    }

    // Legend
    this.plotSvg.append("line").attr("x1", PP.l + 4).attr("y1", PP.t + pH - 12)
      .attr("x2", PP.l + 22).attr("y2", PP.t + pH - 12)
      .attr("stroke", "var(--contr-input)").attr("stroke-width", 2);
    this.plotSvg.append("text").attr("x", PP.l + 26).attr("y", PP.t + pH - 8)
      .attr("class", "annot").attr("fill", "var(--rl-ink-muted)").text("actual");
    this.plotSvg.append("line").attr("x1", PP.l + 4).attr("y1", PP.t + pH + 4)
      .attr("x2", PP.l + 22).attr("y2", PP.t + pH + 4)
      .attr("stroke", "var(--contr-bound)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4 2");
    this.plotSvg.append("text").attr("x", PP.l + 26).attr("y", PP.t + pH + 8)
      .attr("class", "annot").attr("fill", "var(--rl-ink-muted)").text("γᵏ·d₀ bound");
  }
}

customElements.define("bellman-contraction-explorer", BellmanContractionExplorer);
