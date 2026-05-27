/**
 * V2 — Variance Explorer.
 * Demonstrates infinite-variance behavior when σ_q < 1/√2 ≈ 0.707.
 * Center: weight histogram. Right: running-average plot of ordinary IS.
 * Red zone on slider below threshold.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { normalPdf, sampleNormal, mulberry32 } from "../importance-sampling/gaussian";
import { ordinaryIS, effectiveSampleSize } from "../importance-sampling/estimators";

const THRESHOLD = 1 / Math.sqrt(2); // ≈ 0.707
const SEED = 13;

const W = 760, H = 400;
const HIST_W = 320, HIST_H = 200;
const LINE_W = 300, LINE_H = 200;
const MARGIN = { top: 32, right: 16, bottom: 40, left: 48 };

export class VarianceExplorer extends HTMLElement {
  private sigmaQ = 1.0;
  private N = 1000;
  private svgEl!: SVGSVGElement;
  private readoutEl!: HTMLElement;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "variance-explorer" });

    // Controls
    const controls = document.createElement("div");
    controls.className = "rl-controls-row";
    controls.style.flexWrap = "wrap";
    controls.style.gap = "16px 32px";
    controls.innerHTML = `
      <div>
        <label class="rl-label">
          σ<sub>q</sub>
          <input type="range" min="0.3" max="2.5" step="0.02" value="${this.sigmaQ}"
                 id="v2-sq" style="width:180px">
          <span class="rl-mono" id="v2-sq-val">${this.sigmaQ.toFixed(2)}</span>
        </label>
        <div id="v2-regime" style="font-size:11px;margin-top:4px;height:18px"></div>
      </div>
      <label class="rl-label">N
        <input type="range" min="100" max="5000" step="100" value="${this.N}"
               id="v2-n" style="width:140px">
        <span class="rl-mono" id="v2-n-val">${this.N}</span>
      </label>`;
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

    this.readoutEl = document.createElement("div");
    this.readoutEl.className = "rl-readout-row";
    this.readoutEl.style.cssText = "display:flex;gap:24px;font-size:12px;padding:8px 0;font-family:var(--rl-font-mono)";
    body.appendChild(this.readoutEl);

    this.appendChild(panel);

    const sqSlider = controls.querySelector<HTMLInputElement>("#v2-sq")!;
    const sqVal = controls.querySelector<HTMLElement>("#v2-sq-val")!;
    const regimeEl = controls.querySelector<HTMLElement>("#v2-regime")!;
    const nSlider = controls.querySelector<HTMLInputElement>("#v2-n")!;
    const nVal = controls.querySelector<HTMLElement>("#v2-n-val")!;

    const updateRegime = () => {
      const finite = this.sigmaQ > THRESHOLD;
      regimeEl.textContent = finite
        ? `✓ finite-variance regime  (σ_q > 1/√2 ≈ 0.707)`
        : `⚠ infinite-variance regime  (σ_q < 1/√2)`;
      regimeEl.style.color = finite ? "var(--is-weight)" : "var(--is-explosion)";
    };

    sqSlider.addEventListener("input", () => {
      this.sigmaQ = parseFloat(sqSlider.value);
      sqVal.textContent = this.sigmaQ.toFixed(2);
      updateRegime();
      this.render(setStatus);
    });
    nSlider.addEventListener("input", () => {
      this.N = parseInt(nSlider.value);
      nVal.textContent = String(this.N);
      this.render(setStatus);
    });

    updateRegime();
    this.render(setStatus);
  }

  private render(setStatus: (t: string) => void) {
    const rng = mulberry32(SEED);
    const qSamples = sampleNormal(this.N, 0, this.sigmaQ, rng);

    const wFn = (x: number) => normalPdf(x, 0, 1) / normalPdf(x, 0, this.sigmaQ);
    const fFn = (x: number) => x * x;

    const weights = qSamples.map(wFn);
    const isEst = ordinaryIS(qSamples, fFn, wFn);
    const ess = effectiveSampleSize(weights);
    const essRatio = ess / this.N;

    // Running IS average
    const running: number[] = [];
    let cumSum = 0;
    for (let i = 0; i < qSamples.length; i++) {
      cumSum += wFn(qSamples[i]) * fFn(qSamples[i]);
      running.push(cumSum / (i + 1));
    }

    const svg = d3.select(this.svgEl as SVGSVGElement);
    svg.selectAll("*").remove();

    // ---- Weight histogram (left) ----
    const maxW = Math.min(
      Math.max(...weights.filter((w) => isFinite(w)).sort().slice(-10).slice(0, 1), 5),
      50,
    );
    const wBinGen = d3.bin<number, number>()
      .value((d) => d)
      .domain([0, maxW])
      .thresholds(20);
    const wBins = wBinGen(weights.map((w) => Math.min(w, maxW)));
    const wMax = d3.max(wBins, (b) => b.length) ?? 1;

    const hx = d3.scaleLinear().domain([0, maxW]).range([0, HIST_W]);
    const hy = d3.scaleLinear().domain([0, wMax]).range([HIST_H, 0]);

    const hg = svg.append("g").attr("transform",
      `translate(${MARGIN.left},${MARGIN.top + 30})`);

    // Title
    hg.append("text").attr("x", HIST_W / 2).attr("y", -18)
      .attr("text-anchor", "middle").attr("font-size", "11px")
      .attr("fill", "var(--rl-ink-faint)")
      .text("Weight histogram  w(x) = p(x)/q(x)");

    wBins.forEach((b) => {
      hg.append("rect")
        .attr("x", hx(b.x0 ?? 0))
        .attr("y", hy(b.length))
        .attr("width", Math.max(0, hx(b.x1 ?? 0) - hx(b.x0 ?? 0) - 1))
        .attr("height", HIST_H - hy(b.length))
        .attr("fill", this.sigmaQ < THRESHOLD ? "var(--is-explosion)" : "var(--is-weight)")
        .attr("opacity", 0.7);
    });

    hg.append("g").attr("transform", `translate(0,${HIST_H})`)
      .call(d3.axisBottom(hx).ticks(5).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");
    hg.append("g")
      .call(d3.axisLeft(hy).ticks(4).tickSizeOuter(0))
      .selectAll("text").style("font-size", "9px");

    // Threshold marker on weight histogram x-axis at w=1 (where q=p)
    hg.append("line")
      .attr("x1", hx(1)).attr("x2", hx(1))
      .attr("y1", 0).attr("y2", HIST_H + 6)
      .attr("stroke", "var(--is-truth)").attr("stroke-dasharray", "3 2")
      .attr("stroke-width", 1.2);
    hg.append("text").attr("x", hx(1) + 3).attr("y", HIST_H + 18)
      .attr("font-size", "9px").attr("fill", "var(--is-truth)").text("w=1");

    // ---- Running IS average (right) ----
    const stride = Math.max(1, Math.floor(this.N / 300));
    const runPts = running.filter((_, i) => i % stride === 0 || i === running.length - 1);
    const runIdx = running.map((_, i) => i).filter((i) => i % stride === 0 || i === running.length - 1);

    const rx = d3.scaleLinear().domain([0, this.N]).range([0, LINE_W]);
    const runMin = Math.min(d3.min(runPts) ?? 0, 0.5);
    const runMax = Math.max(d3.max(runPts) ?? 2, 1.5);
    const ry = d3.scaleLinear().domain([runMin, runMax]).range([LINE_H, 0]);

    const lg = svg.append("g").attr("transform",
      `translate(${MARGIN.left + HIST_W + 40},${MARGIN.top + 30})`);

    lg.append("text").attr("x", LINE_W / 2).attr("y", -18)
      .attr("text-anchor", "middle").attr("font-size", "11px")
      .attr("fill", "var(--rl-ink-faint)")
      .text("Running ordinary IS estimate");

    // Infinite-variance red zone shading (when applicable)
    if (this.sigmaQ < THRESHOLD) {
      lg.append("rect")
        .attr("x", 0).attr("y", 0)
        .attr("width", LINE_W).attr("height", LINE_H)
        .attr("fill", "var(--is-explosion)").attr("opacity", 0.06);
    }

    // True value reference
    if (ry.domain()[0] <= 1 && ry.domain()[1] >= 1) {
      lg.append("line")
        .attr("x1", 0).attr("x2", LINE_W)
        .attr("y1", ry(1)).attr("y2", ry(1))
        .attr("stroke", "var(--is-weight)").attr("stroke-dasharray", "5 3")
        .attr("stroke-width", 1.5);
      lg.append("text").attr("x", LINE_W - 2).attr("y", ry(1) - 4)
        .attr("font-size", "9px").attr("text-anchor", "end")
        .attr("fill", "var(--is-weight)").text("true = 1");
    }

    // Running average line
    const linePath = d3.line<[number, number]>()
      .x(([i]) => rx(i))
      .y(([, v]) => ry(Math.max(ry.domain()[0], Math.min(ry.domain()[1], v))));

    lg.append("path")
      .datum(runIdx.map((i, j) => [i, runPts[j]] as [number, number]))
      .attr("d", linePath)
      .attr("fill", "none")
      .attr("stroke", this.sigmaQ < THRESHOLD ? "var(--is-explosion)" : "var(--is-ordinary)")
      .attr("stroke-width", 1.5)
      .attr("opacity", 0.9);

    lg.append("g").attr("transform", `translate(0,${LINE_H})`)
      .call(d3.axisBottom(rx).ticks(5).tickSizeOuter(0)
        .tickFormat((d) => d3.format(".0f")(+d)))
      .selectAll("text").style("font-size", "9px");
    lg.append("g")
      .call(d3.axisLeft(ry).ticks(5).tickSizeOuter(0)
        .tickFormat((d) => d3.format(".2f")(+d)))
      .selectAll("text").style("font-size", "9px");

    // Infinite-variance regime label
    if (this.sigmaQ < THRESHOLD) {
      lg.append("text").attr("x", LINE_W / 2).attr("y", LINE_H - 8)
        .attr("text-anchor", "middle").attr("font-size", "10px")
        .attr("fill", "var(--is-explosion)").attr("font-weight", "600")
        .text("infinite-variance regime");
    }

    // Readouts
    this.readoutEl.innerHTML = `
      <span>σ<sub>q</sub> = <b>${this.sigmaQ.toFixed(2)}</b></span>
      <span>N = <b>${this.N}</b></span>
      <span>Ê[X²] = <b>${isEst.toFixed(4)}</b></span>
      <span>N<sub>eff</sub>/N = <b>${(essRatio * 100).toFixed(1)}%</b></span>`;

    setStatus(`σ_q=${this.sigmaQ.toFixed(2)}  est=${isEst.toFixed(4)}  ESS=${(essRatio*100).toFixed(0)}%`);
  }
}

customElements.define("variance-explorer", VarianceExplorer);
