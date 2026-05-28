/**
 * V5 — Variance Reduction with Baseline.
 * Left: overlaid histogram of per-episode gradient norms (vanilla vs TD-baseline REINFORCE).
 * Right: 10-seed convergence bands (90% CI) for vanilla vs baseline.
 * Data loaded from public/data/pg/pg_variance_comparison.json.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 880;
const H = 400;
const MARGIN = { top: 28, right: 20, bottom: 48, left: 60 };
const HALF = W / 2 - 4;
const CHART_W = HALF - MARGIN.left - MARGIN.right;
const CHART_H = H - MARGIN.top - MARGIN.bottom;

const VANILLA_COLOR = "var(--pg-vanilla)";
const BASELINE_COLOR = "var(--pg-baseline)";

interface VarianceData {
  histNEpisodes: number;
  vanillaGradNorms: number[];
  baselineGradNorms: number[];
  bandNEpisodes: number;
  nSeeds: number;
  vanillaCurves: number[][];
  baselineCurves: number[][];
}

function buildHistogram(values: number[], nBins: number): { x0: number; x1: number; count: number }[] {
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / nBins;
  const bins = Array.from({ length: nBins }, (_, i) => ({
    x0: min + i * step, x1: min + (i + 1) * step, count: 0,
  }));
  for (const v of values) {
    const i = Math.min(Math.floor((v - min) / step), nBins - 1);
    bins[i].count++;
  }
  return bins;
}

function computeBand(curves: number[][]): { mean: number; lo: number; hi: number }[] {
  const N = curves[0].length;
  return Array.from({ length: N }, (_, i) => {
    const vals = curves.map(c => c[i]).sort((a, b) => a - b);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const lo = vals[Math.floor(vals.length * 0.05)];
    const hi = vals[Math.floor(vals.length * 0.95)];
    return { mean, lo, hi };
  });
}

function drawHistogram(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  ox: number,
  vanillaNorms: number[],
  baselineNorms: number[],
) {
  const g = svg.append("g").attr("transform", `translate(${ox + MARGIN.left}, ${MARGIN.top})`);

  g.append("text").attr("x", CHART_W / 2).attr("y", -14)
    .attr("text-anchor", "middle").attr("font-size", "12px").attr("font-weight", "600")
    .attr("fill", "var(--rl-ink)")
    .text("Gradient norm distribution (uniform policy, 300 episodes)");

  const allNorms = [...vanillaNorms, ...baselineNorms];
  const xMax = d3.quantile(allNorms.sort((a,b) => a-b), 0.97) ?? 1;
  const xScale = d3.scaleLinear().domain([0, xMax]).range([0, CHART_W]);

  const N_BINS = 30;
  const vanBins = buildHistogram(vanillaNorms.filter(v => v <= xMax), N_BINS);
  const baseBins = buildHistogram(baselineNorms.filter(v => v <= xMax), N_BINS);
  const yMax = Math.max(...vanBins.map(b => b.count), ...baseBins.map(b => b.count));
  const yScale = d3.scaleLinear().domain([0, yMax]).range([CHART_H, 0]);

  g.append("g").attr("transform", `translate(0,${CHART_H})`)
    .call(d3.axisBottom(xScale).ticks(5).tickSize(-CHART_H))
    .call(ax => ax.select(".domain").remove())
    .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));
  g.append("g")
    .call(d3.axisLeft(yScale).ticks(5).tickSize(-CHART_W))
    .call(ax => ax.select(".domain").remove())
    .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));

  g.append("text").attr("x", CHART_W / 2).attr("y", CHART_H + 36)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b")
    .text("‖∇J‖ per episode");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -CHART_H / 2).attr("y", -44)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b")
    .text("Count");

  const drawBars = (bins: typeof vanBins, color: string, opacity: number) => {
    for (const bin of bins) {
      const bw = Math.max(xScale(bin.x1) - xScale(bin.x0) - 1, 1);
      g.append("rect")
        .attr("x", xScale(bin.x0)).attr("y", yScale(bin.count))
        .attr("width", bw).attr("height", CHART_H - yScale(bin.count))
        .attr("fill", color).attr("opacity", opacity);
    }
  };

  drawBars(vanBins, VANILLA_COLOR, 0.5);
  drawBars(baseBins, BASELINE_COLOR, 0.5);

  // Legend
  const leg = g.append("g").attr("transform", `translate(${CHART_W - 140}, 4)`);
  for (const [label, color, i] of [
    ["Vanilla REINFORCE", VANILLA_COLOR, 0],
    ["+ TD baseline", BASELINE_COLOR, 1],
  ] as [string, string, number][]) {
    leg.append("rect").attr("x", 0).attr("y", i * 18).attr("width", 12).attr("height", 12)
      .attr("fill", color).attr("opacity", 0.7);
    leg.append("text").attr("x", 16).attr("y", i * 18 + 10)
      .attr("font-size", "10px").attr("fill", "var(--rl-ink)").text(label);
  }
}

function drawConvergenceBands(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  ox: number,
  vanillaCurves: number[][],
  baselineCurves: number[][],
  nEpisodes: number,
) {
  const g = svg.append("g").attr("transform", `translate(${ox + MARGIN.left}, ${MARGIN.top})`);

  g.append("text").attr("x", CHART_W / 2).attr("y", -14)
    .attr("text-anchor", "middle").attr("font-size", "12px").attr("font-weight", "600")
    .attr("fill", "var(--rl-ink)")
    .text(`V̂(0,0) over training  ·  10 seeds  ·  90% band`);

  const xScale = d3.scaleLinear().domain([0, nEpisodes - 1]).range([0, CHART_W]);
  const yScale = d3.scaleLinear().domain([0.5, 0.76]).range([CHART_H, 0]);

  g.append("g").attr("transform", `translate(0,${CHART_H})`)
    .call(d3.axisBottom(xScale).ticks(5).tickSize(-CHART_H))
    .call(ax => ax.select(".domain").remove())
    .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));
  g.append("g")
    .call(d3.axisLeft(yScale).ticks(5).tickSize(-CHART_W))
    .call(ax => ax.select(".domain").remove())
    .call(ax => ax.selectAll(".tick line").attr("stroke", "#e2e8f0"));

  g.append("text").attr("x", CHART_W / 2).attr("y", CHART_H + 36)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b").text("Episode");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -CHART_H / 2).attr("y", -44)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b").text("V̂(s₀=0)");

  // V* reference
  g.append("line").attr("x1", 0).attr("x2", CHART_W)
    .attr("y1", yScale(0.729)).attr("y2", yScale(0.729))
    .attr("stroke", "#15803d").attr("stroke-width", 1).attr("stroke-dasharray", "4,3").attr("opacity", 0.6);
  g.append("text").attr("x", CHART_W - 4).attr("y", yScale(0.729) - 3)
    .attr("text-anchor", "end").attr("font-size", "9px").attr("fill", "#15803d").text("V*(0,0)=0.729");

  const stride = Math.max(1, Math.floor(nEpisodes / CHART_W));
  const indices = d3.range(0, nEpisodes, stride);

  const drawBand = (curves: number[][], color: string) => {
    const band = computeBand(curves);
    const areaGen = d3.area<number>()
      .x(i => xScale(i))
      .y0(i => yScale(Math.max(band[i]?.lo ?? 0.5, 0.5)))
      .y1(i => yScale(Math.min(band[i]?.hi ?? 0.76, 0.76)))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(indices)
      .attr("d", areaGen).attr("fill", color).attr("opacity", 0.2);

    const lineGen = d3.line<number>()
      .x(i => xScale(i)).y(i => yScale(band[i]?.mean ?? 0.5)).curve(d3.curveMonotoneX);
    g.append("path").datum(indices)
      .attr("d", lineGen).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2);
  };

  drawBand(vanillaCurves, VANILLA_COLOR);
  drawBand(baselineCurves, BASELINE_COLOR);

  // Legend
  const leg = g.append("g").attr("transform", `translate(${CHART_W - 140}, 4)`);
  for (const [label, color, i] of [
    ["Vanilla", VANILLA_COLOR, 0],
    ["+ TD baseline", BASELINE_COLOR, 1],
  ] as [string, string, number][]) {
    leg.append("rect").attr("x", 0).attr("y", i * 18).attr("width", 22).attr("height", 3)
      .attr("fill", color);
    leg.append("text").attr("x", 26).attr("y", i * 18 + 5)
      .attr("font-size", "10px").attr("fill", "var(--rl-ink)").text(label);
  }
}

class VarianceReductionWithBaseline extends HTMLElement {
  connectedCallback() {
    const { panel, body, setStatus } = createPanel({ id: "variance-reduction-baseline" });
    body.style.overflowX = "auto";

    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    body.appendChild(svgEl);
    this.appendChild(panel);

    const svg = d3.select(svgEl);

    // Divider
    svg.append("line").attr("x1", W / 2).attr("x2", W / 2)
      .attr("y1", 10).attr("y2", H - 10)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

    fetch("/data/pg/pg_variance_comparison.json")
      .then(r => r.json())
      .then((data: VarianceData) => {
        drawHistogram(svg, 0, data.vanillaGradNorms, data.baselineGradNorms);
        drawConvergenceBands(svg, W / 2, data.vanillaCurves, data.baselineCurves, data.bandNEpisodes);

        const vanStd = d3.deviation(data.vanillaGradNorms) ?? 0;
        const baseStd = d3.deviation(data.baselineGradNorms) ?? 0;
        setStatus(`grad-norm std: vanilla=${vanStd.toFixed(3)} baseline=${baseStd.toFixed(3)}`);
      })
      .catch(() => {
        svg.append("text").attr("x", W / 2).attr("y", H / 2)
          .attr("text-anchor", "middle").attr("font-size", "13px").attr("fill", "#64748b")
          .text("Run scripts/pg_traces.py to generate training data.");
        setStatus("data not available");
      });
  }
}

customElements.define("pg-variance-reduction-with-baseline", VarianceReductionWithBaseline);
