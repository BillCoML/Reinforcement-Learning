/**
 * V2 — Contraction Iterator 1D. A cobweb-plot visualization of 1D iteration.
 * Two panels: top shows the function graph T(x) vs y=x with the cobweb overlaid;
 * bottom shows the iteration trail on a number line.
 *
 * Cobweb convention: each step is two axis-aligned segments —
 *   horizontal: (x_k, x_k) → (x_k, T(x_k))
 *   vertical:   (x_k, T(x_k)) → (T(x_k), T(x_k))
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { iterate1D } from "../contractions/ops";

const W = 720, GRAPH_H = 280, LINE_H = 80;
const H = GRAPH_H + LINE_H + 8;
const PAD = { l: 50, r: 20, t: 20, b: 40 };

interface MapDef {
  label: string;
  fn: (x: number) => number;
  domain: [number, number];
  fixedPoint: number | null;
  c: number | null;
  note: string;
}

const MAPS: MapDef[] = [
  { label: "T(x) = 0.5x + 1  (c = 0.5, x* = 2)", fn: x => 0.5 * x + 1, domain: [-1, 12], fixedPoint: 2, c: 0.5, note: "Fast convergence: c = 0.5" },
  { label: "T(x) = 0.9x + 0.5  (c = 0.9, x* = 5)", fn: x => 0.9 * x + 0.5, domain: [-2, 12], fixedPoint: 5, c: 0.9, note: "Slow convergence: c = 0.9" },
  { label: "T(x) = −0.6x + 4  (c = 0.6, x* = 2.5)", fn: x => -0.6 * x + 4, domain: [-1, 10], fixedPoint: 2.5, c: 0.6, note: "Alternating convergence: negative slope" },
  { label: "T(x) = x + 1  (c = 1, no fixed point)", fn: x => x + 1, domain: [-1, 12], fixedPoint: null, c: 1, note: "No convergence: c = 1 (non-expansive only)" },
];

export class ContractionIterator1D extends HTMLElement {
  private mapIdx = 0;
  private x0 = 10;
  private iterates: number[] = [];
  private step = 0;          // how many cobweb hops drawn
  private playing = false;
  private playInterval = 0;
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private cobwebG!: d3.Selection<SVGGElement, unknown, null, undefined>;
  private trailG!: d3.Selection<SVGGElement, unknown, null, undefined>;
  private statusEl!: HTMLElement;
  private stepBtn!: HTMLButtonElement;
  private playBtn!: HTMLButtonElement;
  private resetBtn!: HTMLButtonElement;
  private infoEl!: HTMLElement;

  connectedCallback() { this.build(); }

  private get map(): MapDef { return MAPS[this.mapIdx]; }

  private build() {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "v2-contraction-iterator-1d" });
    this.appendChild(panel);

    // Controls row
    const ctrlRow = document.createElement("div");
    ctrlRow.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px";
    body.appendChild(ctrlRow);

    const mapSel = document.createElement("select");
    mapSel.className = "rl-select";
    MAPS.forEach((m, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = m.label;
      mapSel.appendChild(opt);
    });
    mapSel.addEventListener("change", () => {
      this.mapIdx = parseInt(mapSel.value);
      this.reset();
    });
    ctrlRow.appendChild(mapSel);

    const x0Label = document.createElement("label");
    x0Label.style.cssText = "font-size:12px;color:var(--rl-ink-faint);display:flex;align-items:center;gap:6px";
    x0Label.textContent = "x₀:";
    const x0Input = document.createElement("input");
    x0Input.type = "range";
    x0Input.min = "-1"; x0Input.max = "12"; x0Input.step = "0.5";
    x0Input.value = String(this.x0);
    x0Input.style.width = "100px";
    const x0Val = document.createElement("span");
    x0Val.className = "rl-mono";
    x0Val.style.fontSize = "12px";
    x0Val.textContent = String(this.x0);
    x0Input.addEventListener("input", () => {
      this.x0 = parseFloat(x0Input.value);
      x0Val.textContent = this.x0.toFixed(1);
      this.reset();
    });
    x0Label.appendChild(x0Input);
    x0Label.appendChild(x0Val);
    ctrlRow.appendChild(x0Label);

    // Button row
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;margin-bottom:12px";
    body.appendChild(btnRow);

    this.stepBtn = document.createElement("button");
    this.stepBtn.className = "rl-btn";
    this.stepBtn.textContent = "Step";
    this.stepBtn.addEventListener("click", () => this.doStep());
    btnRow.appendChild(this.stepBtn);

    this.playBtn = document.createElement("button");
    this.playBtn.className = "rl-btn";
    this.playBtn.textContent = "Play";
    this.playBtn.addEventListener("click", () => this.togglePlay());
    btnRow.appendChild(this.playBtn);

    this.resetBtn = document.createElement("button");
    this.resetBtn.className = "rl-btn";
    this.resetBtn.textContent = "Reset";
    this.resetBtn.addEventListener("click", () => this.reset());
    btnRow.appendChild(this.resetBtn);

    this.statusEl = document.createElement("span");
    this.statusEl.className = "rl-mono";
    this.statusEl.style.cssText = "font-size:12px;color:var(--rl-ink-faint);margin-left:8px";
    btnRow.appendChild(this.statusEl);

    // SVG
    const svgWrap = document.createElement("div");
    body.appendChild(svgWrap);
    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    svgWrap.appendChild(svgEl);
    this.svg = d3.select(svgEl);

    this.infoEl = document.createElement("div");
    this.infoEl.style.cssText = "font-size:12px;color:var(--rl-ink-muted);margin-top:8px";
    body.appendChild(this.infoEl);

    this.reset();
  }

  private reset() {
    this.stopPlay();
    this.iterates = iterate1D(this.map.fn, this.x0, 30);
    this.step = 0;
    this.renderAll();
    this.updateStatus();
  }

  private updateStatus() {
    if (this.step === 0) {
      this.statusEl.textContent = `x₀ = ${this.x0.toFixed(2)}`;
    } else {
      const xk = this.iterates[this.step];
      this.statusEl.textContent = `x${this.step} = ${xk.toFixed(4)}`;
    }
    this.infoEl.textContent = this.map.note;
  }

  private doStep() {
    if (this.step >= 25) return;
    this.step++;
    this.addCobwebHop(this.step - 1, this.step);
    this.addTrailDot(this.step);
    this.updateStatus();
  }

  private togglePlay() {
    if (this.playing) {
      this.stopPlay();
    } else {
      this.playing = true;
      this.playBtn.textContent = "Pause";
      this.playInterval = window.setInterval(() => {
        if (this.step >= 25) { this.stopPlay(); return; }
        this.doStep();
      }, 600);
    }
  }

  private stopPlay() {
    this.playing = false;
    this.playBtn.textContent = "Play";
    clearInterval(this.playInterval);
  }

  private renderAll() {
    this.svg.selectAll("*").remove();
    const m = this.map;
    const [xMin, xMax] = m.domain;
    const yMin = xMin, yMax = xMax;

    const gW = W - PAD.l - PAD.r;
    const gH = GRAPH_H - PAD.t - PAD.b;

    const xSc = d3.scaleLinear([xMin, xMax], [PAD.l, PAD.l + gW]);
    const ySc = d3.scaleLinear([yMin, yMax], [PAD.t + gH, PAD.t]);

    // Axes
    const gAxis = this.svg.append("g");
    gAxis.append("g").attr("transform", `translate(0,${PAD.t + gH})`).call(d3.axisBottom(xSc).ticks(5));
    gAxis.append("g").attr("transform", `translate(${PAD.l},0)`).call(d3.axisLeft(ySc).ticks(5));
    gAxis.selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
    gAxis.selectAll("line,path").attr("stroke", "var(--rl-border)");

    // Axis labels
    this.svg.append("text").attr("x", PAD.l + gW / 2).attr("y", GRAPH_H - 4)
      .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").text("x");
    this.svg.append("text").attr("x", 12).attr("y", PAD.t + gH / 2)
      .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--rl-ink-faint)")
      .attr("transform", `rotate(-90, 12, ${PAD.t + gH / 2})`).text("T(x)");

    // Clip path for graph area
    this.svg.append("defs").append("clipPath").attr("id", "cobweb-clip")
      .append("rect").attr("x", PAD.l).attr("y", PAD.t)
      .attr("width", gW).attr("height", gH);

    // y = x identity line
    const xs = [xMin, xMax];
    this.svg.append("line")
      .attr("x1", xSc(xs[0])).attr("y1", ySc(xs[0]))
      .attr("x2", xSc(xs[1])).attr("y2", ySc(xs[1]))
      .attr("stroke", "var(--rl-ink-faint)").attr("stroke-width", 1.2)
      .attr("stroke-dasharray", "4 3").attr("clip-path", "url(#cobweb-clip)");

    // T(x) curve
    const pts = d3.range(xMin, xMax, (xMax - xMin) / 200)
      .map(x => ({ x, y: m.fn(x) }))
      .filter(p => p.y >= yMin && p.y <= yMax);
    const line = d3.line<{ x: number; y: number }>().x(p => xSc(p.x)).y(p => ySc(p.y));
    this.svg.append("path").datum(pts)
      .attr("d", line).attr("fill", "none")
      .attr("stroke", "var(--contr-output)").attr("stroke-width", 2.5)
      .attr("clip-path", "url(#cobweb-clip)");

    // Fixed point marker
    if (m.fixedPoint !== null) {
      const fp = m.fixedPoint;
      if (fp >= xMin && fp <= xMax) {
        this.svg.append("circle")
          .attr("cx", xSc(fp)).attr("cy", ySc(fp))
          .attr("r", 5).attr("fill", "none")
          .attr("stroke", "var(--contr-fixed-point)").attr("stroke-width", 2);
        this.svg.append("text").attr("x", xSc(fp) + 8).attr("y", ySc(fp) - 6)
          .attr("class", "annot").attr("fill", "var(--contr-fixed-point)")
          .text(`x* = ${fp}`);
      }
    }

    // T(x) label
    this.svg.append("text").attr("x", PAD.l + gW - 4).attr("y", PAD.t + 14)
      .attr("text-anchor", "end").attr("class", "annot").attr("fill", "var(--contr-output)")
      .text("T(x)");

    // Starting x₀ dot
    const sx0 = this.x0;
    if (sx0 >= xMin && sx0 <= xMax) {
      this.svg.append("circle")
        .attr("cx", xSc(sx0)).attr("cy", ySc(sx0)).attr("r", 5)
        .attr("fill", "var(--contr-input)").attr("clip-path", "url(#cobweb-clip)");
      this.svg.append("text").attr("x", xSc(sx0) + 8).attr("y", ySc(sx0) - 6)
        .attr("class", "annot").attr("fill", "var(--contr-input)").text("x₀");
    }

    // Groups for cobweb and number-line trail (so we can add to them)
    this.cobwebG = this.svg.append("g").attr("clip-path", "url(#cobweb-clip)");
    this.trailG = this.svg.append("g");

    // Store scales for later use
    (this as any)._xSc = xSc;
    (this as any)._ySc = ySc;

    // --- Number line ---
    const lineY = GRAPH_H + LINE_H / 2;
    this.svg.append("line")
      .attr("x1", PAD.l).attr("y1", lineY)
      .attr("x2", PAD.l + gW).attr("y2", lineY)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1.5);
    // Tick marks
    xSc.ticks(5).forEach(t => {
      this.svg.append("line").attr("x1", xSc(t)).attr("y1", lineY - 4).attr("x2", xSc(t)).attr("y2", lineY + 4)
        .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
      this.svg.append("text").attr("x", xSc(t)).attr("y", lineY + 14)
        .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--rl-ink-faint)").text(t);
    });
    // Fixed point on number line
    if (m.fixedPoint !== null && m.fixedPoint >= xMin && m.fixedPoint <= xMax) {
      this.svg.append("circle").attr("cx", xSc(m.fixedPoint)).attr("cy", lineY).attr("r", 4)
        .attr("fill", "none").attr("stroke", "var(--contr-fixed-point)").attr("stroke-width", 2);
    }
    // x0 on number line
    if (sx0 >= xMin && sx0 <= xMax) {
      this.trailG.append("circle").attr("cx", xSc(sx0)).attr("cy", lineY).attr("r", 5)
        .attr("fill", "var(--contr-input)").attr("opacity", 1);
    }
  }

  private addCobwebHop(from: number, to: number) {
    const xSc = (this as any)._xSc as d3.ScaleLinear<number, number>;
    const ySc = (this as any)._ySc as d3.ScaleLinear<number, number>;
    const xk = this.iterates[from];
    const xk1 = this.iterates[to];
    const m = this.map;
    const [xMin, xMax] = m.domain;
    const yMin = xMin, yMax = xMax;

    const opacity = Math.max(0.25, 1 - from * 0.05);
    const color = "var(--contr-trajectory)";

    // Horizontal segment: (x_k, x_k) → (x_k, T(x_k))
    const txk = m.fn(xk);
    if (xk >= xMin && xk <= xMax && txk >= yMin && txk <= yMax) {
      this.cobwebG.append("line")
        .attr("x1", xSc(xk)).attr("y1", ySc(xk))
        .attr("x2", xSc(xk)).attr("y2", ySc(txk))
        .attr("stroke", color).attr("stroke-width", 1.5).attr("opacity", opacity);
    }
    // Vertical segment: (x_k, T(x_k)) → (T(x_k), T(x_k)) i.e. (x_{k+1}, x_{k+1})
    if (xk >= xMin && xk <= xMax && xk1 >= xMin && xk1 <= xMax && txk >= yMin && txk <= yMax) {
      this.cobwebG.append("line")
        .attr("x1", xSc(xk)).attr("y1", ySc(txk))
        .attr("x2", xSc(xk1)).attr("y2", ySc(xk1))
        .attr("stroke", color).attr("stroke-width", 1.5).attr("opacity", opacity);
    }
  }

  private addTrailDot(k: number) {
    const xSc = (this as any)._xSc as d3.ScaleLinear<number, number>;
    const xk = this.iterates[k];
    const [xMin, xMax] = this.map.domain;
    if (xk < xMin || xk > xMax) return;
    const lineY = GRAPH_H + LINE_H / 2;
    const opacity = Math.max(0.3, 1 - (k - 1) * 0.05);
    this.trailG.append("circle")
      .attr("cx", xSc(xk)).attr("cy", lineY).attr("r", 4)
      .attr("fill", "var(--contr-trajectory)").attr("opacity", opacity);
    // Current point — larger, brighter
    this.trailG.selectAll(".cur-pt").remove();
    this.trailG.append("circle").attr("class", "cur-pt")
      .attr("cx", xSc(xk)).attr("cy", lineY).attr("r", 6)
      .attr("fill", "var(--contr-input)").attr("opacity", 1)
      .attr("stroke", "white").attr("stroke-width", 1.5);
  }
}

customElements.define("contraction-iterator-1d", ContractionIterator1D);
