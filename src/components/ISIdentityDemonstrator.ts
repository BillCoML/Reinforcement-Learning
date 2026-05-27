/**
 * V1 — IS Identity Demonstrator.
 * Side-by-side: direct samples from p = N(0,1) on the left vs IS-reweighted
 * samples from q = N(0,σ_q) on the right. Both estimate E_p[X²] = 1.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { normalPdf, sampleNormal, mulberry32 } from "../importance-sampling/gaussian";
import { ordinaryIS } from "../importance-sampling/estimators";

const W = 760, H = 360;
const MARGIN = { top: 28, right: 16, bottom: 36, left: 44 };
const PW = (W / 2) - MARGIN.left - MARGIN.right - 8;
const PH = H - MARGIN.top - MARGIN.bottom;
const BINS = 28;
const X_RANGE: [number, number] = [-4.2, 4.2];
const SEED = 7;

function makeRng() { return mulberry32(SEED); }

function computeData(N: number, sigmaQ: number) {
  const rng = makeRng();
  const pSamples = sampleNormal(N, 0, 1, rng);
  const qSamples = sampleNormal(N, 0, sigmaQ, rng);

  const wFn = (x: number) => normalPdf(x, 0, 1) / normalPdf(x, 0, sigmaQ);
  const fFn = (x: number) => x * x;

  const directEst = d3.mean(pSamples, fFn) ?? 0;
  const isEst = ordinaryIS(qSamples, fFn, wFn);

  // Bins for left (direct) histogram — density
  const binGen = d3.bin<number, number>()
    .value((d) => d)
    .domain(X_RANGE as [number, number])
    .thresholds(d3.range(X_RANGE[0], X_RANGE[1], (X_RANGE[1] - X_RANGE[0]) / BINS));

  const pBins = binGen(pSamples).map((b) => ({
    x0: b.x0 ?? 0,
    x1: b.x1 ?? 0,
    density: b.length / (N * ((b.x1 ?? 0) - (b.x0 ?? 0))),
  }));

  const qBins = binGen(qSamples).map((b) => {
    const bw = (b.x1 ?? 0) - (b.x0 ?? 0);
    const wSum = b.reduce((s, x) => s + wFn(x), 0);
    return {
      x0: b.x0 ?? 0,
      x1: b.x1 ?? 0,
      density: wSum / (N * (bw || 1)),
    };
  });

  // Weight histogram (inset in right panel)
  const weights = qSamples.map(wFn);
  const maxW = Math.min(d3.quantile(weights.slice().sort(d3.ascending), 0.99) ?? 5, 20);
  const wBinGen = d3.bin<number, number>()
    .value((d) => d)
    .domain([0, maxW])
    .thresholds(15);
  const wBins = wBinGen(weights.map((w) => Math.min(w, maxW))).map((b) => ({
    x0: b.x0 ?? 0, x1: b.x1 ?? 0, count: b.length,
  }));

  return { pBins, qBins, wBins, directEst, isEst, weights, maxW };
}

export class ISIdentityDemonstrator extends HTMLElement {
  private sigmaQ = 1.5;
  private N = 500;
  private svgEl!: SVGSVGElement;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "is-identity-demo" });

    // Controls
    const controls = document.createElement("div");
    controls.className = "rl-controls-row";
    controls.innerHTML = `
      <label class="rl-label">σ<sub>q</sub>
        <input type="range" min="0.3" max="3" step="0.05" value="${this.sigmaQ}" id="sq-slider">
        <span class="rl-mono" id="sq-val">${this.sigmaQ.toFixed(2)}</span>
      </label>
      <label class="rl-label">N
        <input type="range" min="50" max="5000" step="50" value="${this.N}" id="n-slider">
        <span class="rl-mono" id="n-val">${this.N}</span>
      </label>`;
    body.appendChild(controls);

    // SVG
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

    // Wire controls
    const sqSlider = controls.querySelector<HTMLInputElement>("#sq-slider")!;
    const sqVal = controls.querySelector<HTMLElement>("#sq-val")!;
    const nSlider = controls.querySelector<HTMLInputElement>("#n-slider")!;
    const nVal = controls.querySelector<HTMLElement>("#n-val")!;

    sqSlider.addEventListener("input", () => {
      this.sigmaQ = parseFloat(sqSlider.value);
      sqVal.textContent = this.sigmaQ.toFixed(2);
      this.render(setStatus);
    });
    nSlider.addEventListener("input", () => {
      this.N = parseInt(nSlider.value);
      nVal.textContent = String(this.N);
      this.render(setStatus);
    });

    this.render(setStatus);
  }

  private render(setStatus: (t: string) => void) {
    const data = computeData(this.N, this.sigmaQ);
    const svg = d3.select(this.svgEl as SVGSVGElement);
    svg.selectAll("*").remove();

    const yMax = Math.max(
      d3.max(data.pBins, (d) => d.density) ?? 0.6,
      d3.max(data.qBins, (d) => d.density) ?? 0.6,
      normalPdf(0, 0, 1) * 1.1,
    );

    const xScale = d3.scaleLinear().domain(X_RANGE).range([0, PW]);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([PH, 0]);

    // Normal density curves
    const curvePts = d3.range(X_RANGE[0], X_RANGE[1], 0.05);
    const pCurve = d3.line<number>()
      .x((x) => xScale(x))
      .y((x) => yScale(normalPdf(x, 0, 1)))(curvePts);

    const drawPanel = (
      offsetX: number,
      bins: typeof data.pBins,
      color: string,
      label: string,
      estVal: number,
    ) => {
      const g = svg.append("g").attr("transform",
        `translate(${offsetX + MARGIN.left},${MARGIN.top})`);

      // Axes
      g.append("g").attr("transform", `translate(0,${PH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0))
        .selectAll("text").style("font-size", "9px");
      g.append("g")
        .call(d3.axisLeft(yScale).ticks(4).tickSizeOuter(0))
        .selectAll("text").style("font-size", "9px");

      // Bars
      bins.forEach((b) => {
        g.append("rect")
          .attr("x", xScale(b.x0))
          .attr("y", yScale(b.density))
          .attr("width", Math.max(0, xScale(b.x1) - xScale(b.x0) - 1))
          .attr("height", PH - yScale(b.density))
          .attr("fill", color)
          .attr("opacity", 0.55);
      });

      // N(0,1) reference curve (truth)
      if (pCurve) {
        g.append("path").attr("d", pCurve)
          .attr("fill", "none")
          .attr("stroke", "var(--is-truth)")
          .attr("stroke-width", 1.5)
          .attr("stroke-dasharray", "4 2")
          .attr("opacity", 0.5);
      }

      // Reference line at 1.0 (truth)
      g.append("line")
        .attr("x1", 0).attr("x2", PW)
        .attr("y1", yScale(1.0 / ((X_RANGE[1] - X_RANGE[0]) / BINS)))
        .attr("y2", yScale(1.0 / ((X_RANGE[1] - X_RANGE[0]) / BINS)));
      // Estimator readout
      g.append("text")
        .attr("x", PW / 2).attr("y", -10)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("fill", "var(--rl-ink)")
        .text(`${label}  Ê[X²] = ${estVal.toFixed(4)}`);

      // Green ref line at x²=1 position? Actually show as text readout above histogram
    };

    drawPanel(0, data.pBins, "var(--is-target)", "Direct (p=N(0,1))", data.directEst);
    drawPanel(W / 2, data.qBins, "var(--is-proposal)", `IS (q=N(0,${this.sigmaQ.toFixed(2)}))`, data.isEst);


    // Separator
    svg.append("line")
      .attr("x1", W / 2).attr("x2", W / 2)
      .attr("y1", 0).attr("y2", H)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);

    setStatus(`σ_q=${this.sigmaQ.toFixed(2)}  N=${this.N}  IS_est=${data.isEst.toFixed(4)}`);
  }
}

customElements.define("is-identity-demonstrator", ISIdentityDemonstrator);
