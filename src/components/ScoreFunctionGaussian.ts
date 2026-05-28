/**
 * V2 — Score Function Estimator (Gaussian).
 * Live demo: θ slider → Gaussian density + sample dots → score function estimate vs true gradient.
 * Shows unbiasedness and Monte Carlo noise of the REINFORCE estimator.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { mulberry32 } from "../importance-sampling/gaussian";
import { gaussianTrueGradient } from "../pg/gaussian-score";

const W = 720;
const H = 400;
const MARGIN = { top: 20, right: 24, bottom: 40, left: 52 };
const PLOT_W = W - MARGIN.left - MARGIN.right;
const PLOT_H = H - MARGIN.top - MARGIN.bottom;

const X_MIN = -5, X_MAX = 5;
const N_SAMPLES = 50;
const N_CURVE_PTS = 200;

let seed = 42;

function sampleGaussian(theta: number, n: number): number[] {
  const rng = mulberry32(seed++);
  const samples: number[] = [];
  for (let i = 0; i < n; i++) {
    // Box-Muller
    const u1 = rng(), u2 = rng();
    samples.push(theta + Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
  }
  return samples;
}

function pdfGaussian(x: number, theta: number): number {
  return Math.exp(-0.5 * (x - theta) ** 2) / Math.sqrt(2 * Math.PI);
}

class ScoreFunctionGaussian extends HTMLElement {
  private _theta = 1.0;
  private _samples: number[] = [];
  private _svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private _xScale!: d3.ScaleLinear<number, number>;
  private _yScale!: d3.ScaleLinear<number, number>;
  private _estimateEl!: HTMLElement;
  private _trueEl!: HTMLElement;

  connectedCallback() {
    const { panel, body, setStatus } = createPanel({ id: "score-function-gaussian" });
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.gap = "12px";

    // Controls row
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;align-items:center;gap:16px;flex-wrap:wrap;font-size:13px;";

    const thetaLabel = document.createElement("label");
    thetaLabel.style.cssText = "display:flex;align-items:center;gap:8px;";
    thetaLabel.textContent = "θ = ";
    const thetaVal = document.createElement("span");
    thetaVal.style.cssText = "font-family:var(--font-mono);min-width:40px;font-weight:600;color:var(--pg-theta);";
    thetaVal.textContent = this._theta.toFixed(2);
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "-2"; slider.max = "2";
    slider.step = "0.05"; slider.value = String(this._theta);
    slider.style.width = "140px";
    thetaLabel.appendChild(slider);
    thetaLabel.appendChild(thetaVal);

    const resampleBtn = document.createElement("button");
    resampleBtn.className = "rl-btn";
    resampleBtn.textContent = "Resample";
    resampleBtn.style.fontSize = "12px";

    const nLabel = document.createElement("label");
    nLabel.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px;";
    nLabel.textContent = "N = ";
    const nSelect = document.createElement("select");
    nSelect.className = "rl-select";
    for (const v of [10, 50, 200, 1000]) {
      const opt = document.createElement("option");
      opt.value = String(v); opt.textContent = String(v);
      if (v === N_SAMPLES) opt.selected = true;
      nSelect.appendChild(opt);
    }
    nLabel.appendChild(nSelect);

    controls.append(thetaLabel, resampleBtn, nLabel);
    body.appendChild(controls);

    // SVG canvas
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);

    this._svg = d3.select(svgEl);
    const g = this._svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    this._xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, PLOT_W]);
    const yMax = pdfGaussian(0, 0) * 1.15;
    this._yScale = d3.scaleLinear().domain([0, yMax]).range([PLOT_H, 0]);

    // Axes
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${PLOT_H})`)
      .call(d3.axisBottom(this._xScale).ticks(10).tickSize(-PLOT_H))
      .call(ax => ax.select(".domain").remove())
      .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(this._yScale).ticks(4).tickSize(-PLOT_W))
      .call(ax => ax.select(".domain").remove())
      .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));

    // Axis labels
    g.append("text").attr("x", PLOT_W / 2).attr("y", PLOT_H + 32)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b").text("x");
    g.append("text").attr("transform", "rotate(-90)")
      .attr("x", -PLOT_H / 2).attr("y", -38)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b").text("p(x)  /  f(x) = x²");

    // Density curve (will be updated)
    g.append("path").attr("class", "pg-density-curve")
      .attr("fill", "none").attr("stroke", "var(--pg-softmax)").attr("stroke-width", 2);

    // f(x) = x² curve (secondary y-axis illusion — scaled to fit)
    g.append("path").attr("class", "pg-fx-curve")
      .attr("fill", "none").attr("stroke", "var(--pg-vanilla)")
      .attr("stroke-width", 1.5).attr("stroke-dasharray", "5,3");

    // Sample dots
    g.append("g").attr("class", "pg-sample-dots");

    // Vertical theta line
    g.append("line").attr("class", "pg-theta-line")
      .attr("y1", 0).attr("y2", PLOT_H)
      .attr("stroke", "var(--pg-theta)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

    // Legend
    const leg = g.append("g").attr("transform", `translate(${PLOT_W - 160}, 8)`);
    const entries = [
      { col: "var(--pg-softmax)", label: "N(θ, 1) density", dash: false },
      { col: "var(--pg-vanilla)", label: "f(x) = x²  (scaled)", dash: true },
      { col: "#64748b", label: "samples", dash: false, dot: true },
    ];
    entries.forEach((e, i) => {
      const row = leg.append("g").attr("transform", `translate(0,${i * 16})`);
      if (e.dot) {
        row.append("circle").attr("r", 3).attr("cx", 6).attr("cy", 0)
          .attr("fill", e.col).attr("opacity", 0.7);
      } else {
        row.append("line").attr("x1", 0).attr("x2", 14).attr("y1", 0).attr("y2", 0)
          .attr("stroke", e.col).attr("stroke-width", 2)
          .attr("stroke-dasharray", e.dash ? "4,2" : null);
      }
      row.append("text").attr("x", 18).attr("y", 4)
        .attr("font-size", "10px").attr("fill", "var(--rl-ink)").text(e.label);
    });

    // Stats panel
    const statsRow = document.createElement("div");
    statsRow.style.cssText = "display:flex;gap:24px;font-size:13px;flex-wrap:wrap;";

    const makeStatBox = (label: string, color: string) => {
      const box = document.createElement("div");
      box.style.cssText = `display:flex;align-items:baseline;gap:6px;`;
      const lbl = document.createElement("span");
      lbl.style.color = "#64748b"; lbl.textContent = label + " =";
      const val = document.createElement("span");
      val.style.cssText = `font-family:var(--font-mono);font-weight:600;color:${color};font-size:15px;`;
      val.textContent = "—";
      box.append(lbl, val);
      return { box, val };
    };

    const estBox = makeStatBox("∇̂J(θ) [estimate]", "var(--pg-vanilla)");
    const trueBox = makeStatBox("∇J(θ) [true]", "var(--pg-theta)");
    const biasBox = makeStatBox("error", "#64748b");

    this._estimateEl = estBox.val;
    this._trueEl = trueBox.val;

    statsRow.append(estBox.box, trueBox.box, biasBox.box);
    body.appendChild(statsRow);

    this.appendChild(panel);

    let currentN = N_SAMPLES;

    const update = (resample: boolean) => {
      if (resample || this._samples.length !== currentN) {
        this._samples = sampleGaussian(this._theta, currentN);
      }
      this._render(g, this._theta, this._samples);
      const est = this._samples.reduce((acc, x) => acc + x * x * (x - this._theta), 0) / this._samples.length;
      const trueGrad = gaussianTrueGradient(this._theta);
      this._estimateEl.textContent = est.toFixed(4);
      this._trueEl.textContent = trueGrad.toFixed(4);
      biasBox.val.textContent = (est - trueGrad).toFixed(4);
      biasBox.val.style.color = Math.abs(est - trueGrad) > Math.abs(trueGrad) * 0.5 ? "var(--pg-high-variance)" : "#64748b";
      setStatus(`θ=${this._theta.toFixed(2)} N=${currentN}`);
    };

    slider.addEventListener("input", () => {
      this._theta = parseFloat(slider.value);
      thetaVal.textContent = this._theta.toFixed(2);
      update(true);
    });

    resampleBtn.addEventListener("click", () => update(true));

    nSelect.addEventListener("change", () => {
      currentN = parseInt(nSelect.value);
      update(true);
    });

    update(true);
  }

  private _render(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    theta: number,
    samples: number[],
  ) {
    const x = this._xScale;
    const y = this._yScale;
    const yMax = (y.domain() as [number, number])[1];

    // Density curve
    const densityData = d3.range(N_CURVE_PTS).map(i => {
      const xv = X_MIN + (X_MAX - X_MIN) * i / (N_CURVE_PTS - 1);
      return [xv, pdfGaussian(xv, theta)] as [number, number];
    });
    const line = d3.line<[number,number]>().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveBasis);
    g.select(".pg-density-curve").attr("d", line(densityData));

    // f(x) = x² scaled to fit on same axes (scale so max visible value = 0.7 * yMax)
    const fxMax = Math.max(...d3.range(N_CURVE_PTS).map(i => {
      const xv = X_MIN + (X_MAX - X_MIN) * i / (N_CURVE_PTS - 1);
      return xv * xv;
    }));
    const fxScale = (0.7 * yMax) / fxMax;
    const fxData = d3.range(N_CURVE_PTS).map(i => {
      const xv = X_MIN + (X_MAX - X_MIN) * i / (N_CURVE_PTS - 1);
      return [xv, xv * xv * fxScale] as [number, number];
    });
    g.select(".pg-fx-curve").attr("d", line(fxData));

    // Sample dots (on density curve level)
    const dotsG = g.select(".pg-sample-dots");
    const dots = dotsG.selectAll<SVGCircleElement, number>("circle").data(samples);
    dots.join(
      enter => enter.append("circle")
        .attr("r", 3).attr("opacity", 0)
        .attr("fill", "#64748b"),
      update => update,
      exit => exit.remove(),
    )
      .attr("cx", xv => x(xv))
      .attr("cy", xv => y(pdfGaussian(xv, theta)))
      .attr("opacity", 0.55);

    // Theta line
    g.select(".pg-theta-line")
      .attr("x1", x(theta)).attr("x2", x(theta));
  }
}

customElements.define("pg-score-function-gaussian", ScoreFunctionGaussian);
