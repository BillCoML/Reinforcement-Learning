/**
 * V3 — Banach Iteration in 2D. Multiple starting points converge to the same
 * fixed point under a 2D affine contraction T(x) = Ax + b. Shows trajectories
 * in the 2D plane plus a log-scale convergence plot.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { fixedPoint2D, opNormInfinity, iterate2D, spectralRadius2x2 } from "../contractions/ops";

const PLANE_W = 380, PLANE_H = 360;
const PLOT_W = 280, PLOT_H = 260;
const PAD = 40;
const MAX_ITERS = 30;

const COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#d97706"];

export class BanachIteration2D extends HTMLElement {
  private A = [[0.3, 0.2], [0.1, 0.4]];
  private b = [1.5, 0.8];
  private starts: Array<[number, number]> = [[6, 1], [-3, 5], [4, -4], [0, 6]];
  private step = 0;
  private playing = false;
  private playInterval = 0;
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private plotSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private statusEl!: HTMLElement;
  private normEl!: HTMLElement;
  private specEl!: HTMLElement;
  private validEl!: HTMLElement;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "v3-banach-iteration-2d" });
    this.appendChild(panel);

    const outerRow = document.createElement("div");
    outerRow.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start";
    body.appendChild(outerRow);

    // Left: matrix sliders + plane
    const leftCol = document.createElement("div");
    outerRow.appendChild(leftCol);

    leftCol.appendChild(this.buildMatrixControls());

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${PLANE_W} ${PLANE_H}`);
    svgEl.setAttribute("width", String(PLANE_W));
    svgEl.setAttribute("height", String(PLANE_H));
    svgEl.classList.add("rl-svg");
    svgEl.style.display = "block";
    leftCol.appendChild(svgEl);
    this.svg = d3.select(svgEl);

    // Right: convergence plot + info
    const rightCol = document.createElement("div");
    rightCol.style.minWidth = "280px";
    outerRow.appendChild(rightCol);

    const plotEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    plotEl.setAttribute("viewBox", `0 0 ${PLOT_W} ${PLOT_H}`);
    plotEl.setAttribute("width", String(PLOT_W));
    plotEl.setAttribute("height", String(PLOT_H));
    plotEl.classList.add("rl-svg");
    plotEl.style.display = "block";
    rightCol.appendChild(plotEl);
    this.plotSvg = d3.select(plotEl);

    const info = document.createElement("div");
    info.style.cssText = "margin-top:12px;font-size:12px";
    rightCol.appendChild(info);

    this.normEl = Object.assign(document.createElement("div"), { style: "margin-bottom:4px" });
    this.specEl = Object.assign(document.createElement("div"), { style: "margin-bottom:4px" });
    this.validEl = Object.assign(document.createElement("div"), { style: "margin-bottom:8px;font-weight:600" });
    info.appendChild(this.normEl);
    info.appendChild(this.specEl);
    info.appendChild(this.validEl);

    // Controls
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;margin-top:8px;flex-wrap:wrap";
    rightCol.appendChild(btnRow);

    const stepBtn = document.createElement("button");
    stepBtn.className = "rl-btn"; stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => { if (this.step < MAX_ITERS) { this.step++; this.redraw(); } });
    btnRow.appendChild(stepBtn);

    const playBtn = document.createElement("button");
    playBtn.className = "rl-btn"; playBtn.textContent = "Play";
    playBtn.addEventListener("click", () => {
      if (this.playing) { clearInterval(this.playInterval); this.playing = false; playBtn.textContent = "Play"; }
      else { this.playing = true; playBtn.textContent = "Pause";
        this.playInterval = window.setInterval(() => {
          if (this.step >= MAX_ITERS) { clearInterval(this.playInterval); this.playing = false; playBtn.textContent = "Play"; return; }
          this.step++; this.redraw();
        }, 300); }
    });
    btnRow.appendChild(playBtn);

    const resetBtn = document.createElement("button");
    resetBtn.className = "rl-btn"; resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => { clearInterval(this.playInterval); this.playing = false; playBtn.textContent = "Play"; this.step = 0; this.redraw(); });
    btnRow.appendChild(resetBtn);

    const addBtn = document.createElement("button");
    addBtn.className = "rl-btn"; addBtn.textContent = "+ Point";
    addBtn.addEventListener("click", () => {
      const px = (Math.random() - 0.5) * 14;
      const py = (Math.random() - 0.5) * 14;
      this.starts.push([px, py]);
      this.redraw();
    });
    btnRow.appendChild(addBtn);

    this.statusEl = Object.assign(document.createElement("div"), {
      style: "margin-top:6px;font-size:11px;color:var(--rl-ink-faint)",
    });
    rightCol.appendChild(this.statusEl);

    // Click on plane to add start point
    this.svg.on("click", (ev: MouseEvent) => {
      const rect = svgEl.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const { xSc, ySc } = this.getScales();
      this.starts.push([xSc.invert(sx), ySc.invert(sy)]);
      this.redraw();
    });

    this.redraw();
  }

  private buildMatrixControls(): HTMLElement {
    const div = document.createElement("div");
    div.style.cssText = "display:grid;grid-template-columns:auto auto auto auto;gap:4px 8px;align-items:center;margin-bottom:10px;font-size:11px";

    const label = document.createElement("div");
    label.style.cssText = "grid-column:1/-1;color:var(--rl-ink-faint);font-size:11px;margin-bottom:2px";
    label.innerHTML = "Matrix <em>A</em> (must have ‖A‖<sub>∞</sub> &lt; 1 for contraction):";
    div.appendChild(label);

    const entries = [[0, 0], [0, 1], [1, 0], [1, 1]];
    entries.forEach(([i, j]) => {
      const lbl = document.createElement("span");
      lbl.style.color = "var(--rl-ink-faint)";
      lbl.textContent = `A[${i}][${j}]:`;
      const sl = document.createElement("input");
      sl.type = "range"; sl.min = "-0.9"; sl.max = "0.9"; sl.step = "0.05";
      sl.value = String(this.A[i][j]);
      sl.style.width = "80px";
      const vl = document.createElement("span");
      vl.className = "rl-mono";
      vl.style.fontSize = "11px";
      vl.textContent = this.A[i][j].toFixed(2);
      sl.addEventListener("input", () => {
        this.A[i][j] = parseFloat(sl.value);
        vl.textContent = this.A[i][j].toFixed(2);
        this.step = 0;
        this.redraw();
      });
      div.appendChild(lbl);
      div.appendChild(sl);
      div.appendChild(vl);
      if ((i + j) % 2 === 1) div.appendChild(document.createElement("div"));
    });
    return div;
  }

  private getScales() {
    const xMin = -8, xMax = 8, yMin = -8, yMax = 8;
    const xSc = d3.scaleLinear([xMin, xMax], [PAD, PLANE_W - PAD]);
    const ySc = d3.scaleLinear([yMin, yMax], [PLANE_H - PAD, PAD]);
    return { xSc, ySc, xMin, xMax, yMin, yMax };
  }

  private redraw() {
    const fp = fixedPoint2D(this.A, this.b);
    const norm = opNormInfinity(this.A);
    const spec = spectralRadius2x2(this.A);
    const isContr = norm < 1;

    this.normEl.innerHTML = `<span style="color:var(--rl-ink-faint)" class="rl-mono">‖A‖<sub>∞</sub> = ${norm.toFixed(3)}</span>`;
    this.specEl.innerHTML = `<span style="color:var(--rl-ink-faint)" class="rl-mono">ρ(A) = ${spec.toFixed(3)}</span>`;
    this.validEl.innerHTML = isContr
      ? `<span style="color:var(--contr-ok)">✓ contraction (c = ${norm.toFixed(3)})</span>`
      : `<span style="color:var(--contr-warn)">⚠ not a contraction (‖A‖<sub>∞</sub> ≥ 1)</span>`;
    this.statusEl.textContent = `k = ${this.step} | x* = (${fp[0].toFixed(3)}, ${fp[1].toFixed(3)})`;

    this.drawPlane(fp);
    this.drawPlot(fp);
  }

  private drawPlane(fp: number[]) {
    const { xSc, ySc } = this.getScales();
    this.svg.selectAll("*").remove();

    // Grid lines
    [-6, -4, -2, 0, 2, 4, 6].forEach(v => {
      this.svg.append("line").attr("x1", xSc(v)).attr("y1", PAD).attr("x2", xSc(v)).attr("y2", PLANE_H - PAD)
        .attr("stroke", "var(--rl-border)").attr("stroke-width", 0.5).attr("opacity", 0.5);
      this.svg.append("line").attr("x1", PAD).attr("y1", ySc(v)).attr("x2", PLANE_W - PAD).attr("y2", ySc(v))
        .attr("stroke", "var(--rl-border)").attr("stroke-width", 0.5).attr("opacity", 0.5);
    });

    // Axes
    this.svg.append("line").attr("x1", PAD).attr("y1", ySc(0)).attr("x2", PLANE_W - PAD).attr("y2", ySc(0))
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
    this.svg.append("line").attr("x1", xSc(0)).attr("y1", PAD).attr("x2", xSc(0)).attr("y2", PLANE_H - PAD)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);

    // Fixed point
    this.svg.append("circle").attr("cx", xSc(fp[0])).attr("cy", ySc(fp[1])).attr("r", 6)
      .attr("fill", "none").attr("stroke", "var(--contr-fixed-point)").attr("stroke-width", 2.5);
    this.svg.append("text").attr("x", xSc(fp[0]) + 8).attr("y", ySc(fp[1]) - 7)
      .attr("class", "annot").attr("fill", "var(--contr-fixed-point)").text("x*");

    // Trajectories
    this.starts.forEach((s0, ci) => {
      const color = COLORS[ci % COLORS.length];
      const traj = iterate2D(this.A, this.b, s0, this.step);
      const shown = traj;

      // Trail path
      if (shown.length > 1) {
        const pathPts = shown.map(p => [xSc(p[0]), ySc(p[1])]);
        const d = "M" + pathPts.map(p => p.join(",")).join("L");
        this.svg.append("path").attr("d", d).attr("fill", "none")
          .attr("stroke", color).attr("stroke-width", 1.2).attr("opacity", 0.6);
      }

      // Dots
      shown.forEach((p, ki) => {
        const t = ki / Math.max(shown.length - 1, 1);
        const op = 0.25 + 0.75 * t;
        this.svg.append("circle").attr("cx", xSc(p[0])).attr("cy", ySc(p[1]))
          .attr("r", ki === shown.length - 1 ? 5 : 3)
          .attr("fill", color).attr("opacity", op);
      });

      // Starting label
      this.svg.append("text").attr("x", xSc(s0[0]) + 6).attr("y", ySc(s0[1]) - 5)
        .attr("class", "annot").attr("fill", color).text(`p${ci + 1}`);
    });
  }

  private drawPlot(fp: number[]) {
    const PP = { l: 44, r: 12, t: 20, b: 36 };
    const pW = PLOT_W - PP.l - PP.r;
    const pH = PLOT_H - PP.t - PP.b;
    this.plotSvg.selectAll("*").remove();

    if (this.step < 2) {
      this.plotSvg.append("text").attr("x", PLOT_W / 2).attr("y", PLOT_H / 2)
        .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--rl-ink-faint)")
        .text("Step to see convergence");
      return;
    }

    const allDists: number[][] = this.starts.map(s0 => {
      const traj = iterate2D(this.A, this.b, s0, this.step);
      return traj.map(p => Math.max(1e-9, Math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2)));
    });

    const allVals = allDists.flat();
    const yMax = Math.max(...allVals);
    const yMin = Math.min(...allVals);

    const xSc = d3.scaleLinear([0, this.step], [PP.l, PP.l + pW]);
    const ySc = d3.scaleLog([Math.max(yMin * 0.5, 1e-8), yMax * 1.5], [PP.t + pH, PP.t]);

    this.plotSvg.append("g").attr("transform", `translate(0,${PP.t + pH})`).call(d3.axisBottom(xSc).ticks(5).tickFormat(d3.format("d")))
      .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
    this.plotSvg.append("g").attr("transform", `translate(${PP.l},0)`)
      .call(d3.axisLeft(ySc).ticks(4, ".0e"))
      .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "9px");
    this.plotSvg.selectAll("line,path").attr("stroke", "var(--rl-border)");

    this.plotSvg.append("text").attr("x", PP.l + pW / 2).attr("y", PLOT_H - 4)
      .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").text("iteration k");
    this.plotSvg.append("text").attr("x", 10).attr("y", PP.t + pH / 2)
      .attr("text-anchor", "middle").attr("transform", `rotate(-90,10,${PP.t + pH / 2})`)
      .attr("class", "annot").attr("fill", "var(--rl-ink-faint)").text("‖xₖ − x*‖");

    this.plotSvg.append("text").attr("x", PP.l + pW - 4).attr("y", PP.t + 12)
      .attr("text-anchor", "end").attr("class", "annot")
      .attr("fill", "var(--rl-ink-faint)").text("log scale");

    allDists.forEach((dists, ci) => {
      const color = COLORS[ci % COLORS.length];
      const pts = dists.map((d, i) => [i, d] as [number, number]);
      const pathData = pts.map(([k, d]) => [xSc(k), (() => { try { return ySc(d); } catch { return PP.t + pH; } })()]);
      if (pathData.length > 1) {
        this.plotSvg.append("path")
          .attr("d", "M" + pathData.map(p => p.join(",")).join("L"))
          .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("opacity", 0.85);
      }
    });
  }
}

customElements.define("banach-iteration-2d", BanachIteration2D);
