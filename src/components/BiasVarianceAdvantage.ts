/**
 * V7 — Bias-Variance in Advantage Estimation.
 * Left: schematic of the advantage estimator family (MC ↔ TD(0) spectrum).
 *        Slider on n moves a marker through the family.
 * Right: empirical RMSE of n-step advantage estimates vs true A^π(0, RIGHT).
 *        Data loaded from public/data/pg/pg_advantage_rmse.json.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 720;
const H = 380;
const HALF = W / 2 - 4;
const MARGIN = { top: 28, right: 20, bottom: 48, left: 60 };
const CHART_W = HALF - MARGIN.left - MARGIN.right;
const CHART_H = H - MARGIN.top - MARGIN.bottom;

interface RMSEResult {
  n: number;
  rmse: number;
  biasSq: number;
  variance: number;
  meanEstimate: number;
}

interface AdvantageData {
  trueAdvantage: number;
  trueV0: number;
  trueQ0Right: number;
  nEpisodesPerN: number;
  results: RMSEResult[];
}

function drawSchematic(
  svg: d3.Selection<SVGGElement, unknown, null, undefined>,
  selectedN: number,
  nValues: number[],
) {
  const g = svg.append("g").attr("transform", `translate(10,${MARGIN.top})`);
  g.selectAll("*").remove();

  g.append("text").attr("x", HALF / 2 - 10).attr("y", -14)
    .attr("text-anchor", "middle").attr("font-size", "12px").attr("font-weight", "600")
    .attr("fill", "var(--rl-ink)").text("Advantage estimator family");

  const SX = 24, EX = HALF - 24;
  const MID_Y = CHART_H / 2;
  const LINE_Y = MID_Y + 20;

  // Bias-variance gradient bar
  const barGrad = svg.append("defs").append("linearGradient")
    .attr("id", "bv-gradient").attr("x1", "0%").attr("x2", "100%");
  barGrad.append("stop").attr("offset", "0%").attr("stop-color", "var(--pg-high-bias)");
  barGrad.append("stop").attr("offset", "100%").attr("stop-color", "var(--pg-high-variance)");

  g.append("rect").attr("x", SX).attr("y", LINE_Y)
    .attr("width", EX - SX).attr("height", 8).attr("rx", 4)
    .attr("fill", `url(#bv-gradient)`).attr("opacity", 0.5);

  // Endpoint labels
  g.append("text").attr("x", SX).attr("y", LINE_Y - 6)
    .attr("text-anchor", "middle").attr("font-size", "10px").attr("font-weight", "600")
    .attr("fill", "var(--pg-high-bias)").text("TD(0)  n=1");
  g.append("text").attr("x", EX).attr("y", LINE_Y - 6)
    .attr("text-anchor", "middle").attr("font-size", "10px").attr("font-weight", "600")
    .attr("fill", "var(--pg-high-variance)").text("MC  n=∞");

  // Bias label (left)
  g.append("text").attr("x", SX - 4).attr("y", LINE_Y + 20)
    .attr("font-size", "9px").attr("fill", "var(--pg-high-bias)").text("Low bias");
  g.append("text").attr("x", SX - 4).attr("y", LINE_Y + 30)
    .attr("font-size", "9px").attr("fill", "var(--pg-high-bias)").text("Low var");

  // Variance label (right)
  g.append("text").attr("x", EX + 4).attr("y", LINE_Y + 20)
    .attr("text-anchor", "end").attr("font-size", "9px").attr("fill", "var(--pg-high-variance)").text("Zero bias");
  g.append("text").attr("x", EX + 4).attr("y", LINE_Y + 30)
    .attr("text-anchor", "end").attr("font-size", "9px").attr("fill", "var(--pg-high-variance)").text("High var");

  // n-step tick marks
  const nMax = Math.max(...nValues);
  const nToX = (n: number) => SX + ((Math.log(n) / Math.log(nMax))) * (EX - SX);

  for (const n of nValues) {
    const x = nToX(n);
    g.append("line").attr("x1", x).attr("x2", x)
      .attr("y1", LINE_Y - 2).attr("y2", LINE_Y + 10)
      .attr("stroke", "#94a3b8").attr("stroke-width", 1);
    g.append("text").attr("x", x).attr("y", LINE_Y - 9)
      .attr("text-anchor", "middle").attr("font-size", "8px").attr("fill", "#94a3b8")
      .text(n === nValues[nValues.length - 1] ? "∞" : String(n));
  }

  // Selected marker
  const selX = nToX(selectedN);
  g.append("circle").attr("cx", selX).attr("cy", LINE_Y + 4).attr("r", 7)
    .attr("fill", "var(--pg-balanced)").attr("stroke", "white").attr("stroke-width", 2);
  g.append("text").attr("x", selX).attr("y", LINE_Y + 8)
    .attr("text-anchor", "middle").attr("font-size", "8px").attr("fill", "white").attr("font-weight", "600")
    .text(selectedN === nValues[nValues.length - 1] ? "∞" : String(selectedN));

  // Formula display
  const formulas: Record<string, string> = {
    "1": "Â_t = r_{t+1} + γV_φ(s_{t+1}) − V_φ(s_t)  [TD error]",
    "∞": "Â_t = G_t − V_φ(s_t)  [MC return minus baseline]",
    "default": `Â_t = Σₖ γᵏr_{t+k} + γⁿV_φ(s_{t+n}) − V_φ(s_t)  [${selectedN}-step]`,
  };
  const fKey = selectedN === 1 ? "1" : selectedN === nValues[nValues.length - 1] ? "∞" : "default";
  g.append("text").attr("x", (EX - SX) / 2 + SX).attr("y", LINE_Y + 55)
    .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "#334155")
    .text(formulas[fKey]);

  // GAE annotation
  g.append("rect").attr("x", SX + 20).attr("y", MID_Y - 60).attr("width", EX - SX - 40).attr("height", 38)
    .attr("rx", 6).attr("fill", "#f0fdf4").attr("stroke", "#86efac").attr("stroke-width", 1);
  g.append("text").attr("x", (EX - SX) / 2 + SX).attr("y", MID_Y - 45)
    .attr("text-anchor", "middle").attr("font-size", "10px").attr("font-weight", "600")
    .attr("fill", "#15803d").text("GAE (λ) — Lesson 11");
  g.append("text").attr("x", (EX - SX) / 2 + SX).attr("y", MID_Y - 32)
    .attr("text-anchor", "middle").attr("font-size", "9px").attr("fill", "#166534")
    .text("Exponentially-weighted mix: λ=0→TD, λ=1→MC");
}

function drawRMSEChart(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  data: RMSEResult[],
  selectedN: number,
) {
  const g = svg.select<SVGGElement>(".rmse-panel");
  g.selectAll("*").remove();

  g.append("text").attr("x", CHART_W / 2).attr("y", -14)
    .attr("text-anchor", "middle").attr("font-size", "12px").attr("font-weight", "600")
    .attr("fill", "var(--rl-ink)").text("RMSE of  Â^π(0, RIGHT)  vs true");

  const nValues = data.map(d => d.n);
  const xScale = d3.scaleBand().domain(nValues.map(String)).range([0, CHART_W]).padding(0.2);
  const yMax = Math.max(...data.map(d => d.rmse)) * 1.15;
  const yScale = d3.scaleLinear().domain([0, yMax]).range([CHART_H, 0]);

  g.append("g").attr("transform", `translate(0,${CHART_H})`)
    .call(d3.axisBottom(xScale).tickFormat(d => d === "200" ? "∞" : d).tickSize(-CHART_H))
    .call(ax => { ax.select(".domain").remove(); ax.selectAll(".tick line").attr("stroke", "#e2e8f0"); });
  g.append("g")
    .call(d3.axisLeft(yScale).ticks(5).tickSize(-CHART_W))
    .call(ax => { ax.select(".domain").remove(); ax.selectAll(".tick line").attr("stroke", "#e2e8f0"); });

  g.append("text").attr("x", CHART_W / 2).attr("y", CHART_H + 36)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b").text("n  (steps)");
  g.append("text").attr("transform", "rotate(-90)").attr("x", -CHART_H / 2).attr("y", -44)
    .attr("text-anchor", "middle").attr("font-size", "11px").attr("fill", "#64748b").text("RMSE");

  // Bias² + variance stacked bars
  for (const d of data) {
    const x = xScale(String(d.n))!;
    const bw = xScale.bandwidth();
    const isSelected = d.n === selectedN;
    const biasFill = "var(--pg-high-bias)";
    const varFill = "var(--pg-high-variance)";

    const biasH = CHART_H - yScale(Math.sqrt(d.biasSq));
    const varH = CHART_H - yScale(Math.sqrt(d.variance));

    g.append("rect").attr("x", x).attr("y", yScale(Math.sqrt(d.biasSq)))
      .attr("width", bw).attr("height", biasH)
      .attr("fill", biasFill).attr("opacity", isSelected ? 0.9 : 0.5);
    g.append("rect").attr("x", x).attr("y", yScale(Math.sqrt(d.biasSq + d.variance)))
      .attr("width", bw).attr("height", varH)
      .attr("fill", varFill).attr("opacity", isSelected ? 0.9 : 0.5);

    if (isSelected) {
      g.append("rect").attr("x", x - 1).attr("y", yScale(d.rmse) - 1)
        .attr("width", bw + 2).attr("height", CHART_H - yScale(d.rmse) + 1)
        .attr("fill", "none").attr("stroke", "var(--pg-balanced)").attr("stroke-width", 2);
    }
  }

  // RMSE line overlay
  const lineGen = d3.line<RMSEResult>()
    .x(d => xScale(String(d.n))! + xScale.bandwidth() / 2)
    .y(d => yScale(d.rmse)).curve(d3.curveCatmullRom);
  g.append("path").datum(data)
    .attr("fill", "none").attr("stroke", "#1c1e22").attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "4,2").attr("d", lineGen);

  // Legend
  const leg = g.append("g").attr("transform", `translate(${CHART_W - 100}, 4)`);
  leg.append("rect").attr("x", 0).attr("y", 0).attr("width", 10).attr("height", 10).attr("fill", "var(--pg-high-bias)").attr("opacity", 0.7);
  leg.append("text").attr("x", 14).attr("y", 9).attr("font-size", "9px").attr("fill", "var(--rl-ink)").text("|bias|");
  leg.append("rect").attr("x", 0).attr("y", 14).attr("width", 10).attr("height", 10).attr("fill", "var(--pg-high-variance)").attr("opacity", 0.7);
  leg.append("text").attr("x", 14).attr("y", 23).attr("font-size", "9px").attr("fill", "var(--rl-ink)").text("√variance");
}

class BiasVarianceAdvantage extends HTMLElement {
  connectedCallback() {
    const { panel, body, setStatus } = createPanel({ id: "bias-variance-advantage" });
    body.style.overflowX = "auto";

    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%";
    svgEl.style.maxWidth = `${W}px`;
    body.appendChild(svgEl);

    // Slider
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;align-items:center;gap:10px;font-size:12px;margin:8px 0;";
    const sliderLabel = document.createElement("label");
    sliderLabel.style.cssText = "display:flex;align-items:center;gap:8px;";
    sliderLabel.textContent = "n = ";
    const nValEl = document.createElement("span");
    nValEl.style.cssText = "font-family:var(--font-mono);font-weight:600;color:var(--pg-balanced);min-width:36px;";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "8"; slider.value = "0";
    slider.style.width = "140px";
    sliderLabel.append(slider, nValEl);
    controls.appendChild(sliderLabel);
    body.appendChild(controls);
    this.appendChild(panel);

    const svg = d3.select(svgEl);

    // Divider
    svg.append("line").attr("x1", HALF).attr("x2", HALF)
      .attr("y1", 10).attr("y2", H - 10)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

    // Static schematic group (will be re-drawn on slider change)
    const gSchematic = svg.append("g").attr("class", "schematic-group");

    // RMSE panel group
    svg.append("g").attr("class", "rmse-panel")
      .attr("transform", `translate(${HALF + MARGIN.left},${MARGIN.top})`);

    let data: RMSEResult[] = [];
    const nValues = [1, 2, 3, 5, 8, 13, 21, 50, 200];
    let selectedIdx = 0;

    const update = () => {
      const n = nValues[selectedIdx];
      nValEl.textContent = n === 200 ? "∞" : String(n);

      // Remove and redraw schematic
      gSchematic.selectAll("*").remove();
      drawSchematic(gSchematic.attr("transform", `translate(0,0)`), n, nValues);

      if (data.length > 0) {
        drawRMSEChart(svg, data, n);
        const d = data.find(r => r.n === n);
        if (d) setStatus(`n=${n} rmse=${d.rmse.toFixed(4)}`);
      }
    };

    slider.addEventListener("input", () => {
      selectedIdx = parseInt(slider.value);
      update();
    });

    fetch("/data/pg/pg_advantage_rmse.json")
      .then(r => r.json())
      .then((d: AdvantageData) => {
        data = d.results;
        slider.max = String(data.length - 1);
        drawRMSEChart(svg, data, nValues[0]);
        setStatus(`true A^π(0,R)=${d.trueAdvantage.toFixed(4)}`);
      })
      .catch(() => {
        svg.select(".rmse-panel").append("text")
          .attr("x", CHART_W / 2).attr("y", CHART_H / 2)
          .attr("text-anchor", "middle").attr("font-size", "12px").attr("fill", "#64748b")
          .text("Run scripts/pg_traces.py to generate data.");
      });

    update();
  }
}

customElements.define("pg-bias-variance-advantage", BiasVarianceAdvantage);
