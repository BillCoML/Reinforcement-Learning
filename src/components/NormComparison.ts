/**
 * V1 — Norm Comparison. A 2D plane with a draggable point; shows ℓ1, ℓ2, ℓ∞
 * norms live, with unit balls of all three drawn as outlines. A toggle
 * highlights the dominant coordinate in the ℓ∞ reading.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 600, H = 360;
const CX = 210, CY = 180;          // origin in SVG coords
const SCALE = 80;                   // pixels per unit

export class NormComparison extends HTMLElement {
  private ptX = 1.2;
  private ptY = 0.8;
  private highlight = false;
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private dragCircle!: d3.Selection<SVGCircleElement, unknown, null, undefined>;
  private l1El!: HTMLElement;
  private l2El!: HTMLElement;
  private linfEl!: HTMLElement;
  private domCoordEl!: HTMLElement;

  connectedCallback() {
    this.build();
  }

  private build() {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "v1-norm-comparison" });
    this.appendChild(panel);

    const wrap = document.createElement("div");
    wrap.style.display = "flex";
    wrap.style.gap = "24px";
    wrap.style.alignItems = "flex-start";
    wrap.style.flexWrap = "wrap";
    body.appendChild(wrap);

    // SVG side
    const svgWrap = document.createElement("div");
    wrap.appendChild(svgWrap);
    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", String(W));
    svgEl.setAttribute("height", String(H));
    svgEl.classList.add("rl-svg");
    svgEl.style.maxWidth = "100%";
    svgWrap.appendChild(svgEl);
    this.svg = d3.select(svgEl);

    // Draw unit balls
    this.drawBalls();

    // Axes
    this.svg.append("line").attr("x1", 10).attr("y1", CY).attr("x2", W - 10).attr("y2", CY)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
    this.svg.append("line").attr("x1", CX).attr("y1", 10).attr("x2", CX).attr("y2", H - 10)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1);
    this.svg.append("text").attr("x", W - 14).attr("y", CY - 6).attr("class", "annot")
      .attr("fill", "var(--rl-ink-faint)").text("x₁");
    this.svg.append("text").attr("x", CX + 6).attr("y", 18).attr("class", "annot")
      .attr("fill", "var(--rl-ink-faint)").text("x₂");

    // Draggable point and vector line
    this.svg.append("line").attr("class", "pt-line")
      .attr("stroke", "var(--contr-input)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    this.dragCircle = this.svg.append("circle")
      .attr("class", "pt")
      .attr("r", 7)
      .attr("fill", "var(--contr-input)")
      .attr("cursor", "grab")
      .attr("stroke", "white")
      .attr("stroke-width", 2) as d3.Selection<SVGCircleElement, unknown, null, undefined>;

    // Legend labels on SVG
    const legendY = H - 24;
    this.svg.append("circle").attr("cx", 20).attr("cy", legendY).attr("r", 5).attr("fill", "none")
      .attr("stroke", "#2563eb").attr("stroke-width", 1.5);
    this.svg.append("text").attr("x", 30).attr("y", legendY + 4).attr("class", "annot")
      .attr("fill", "var(--rl-ink-muted)").text("ℓ₂");
    this.svg.append("rect").attr("x", 54).attr("y", legendY - 5).attr("width", 10).attr("height", 10)
      .attr("fill", "none").attr("stroke", "#16a34a").attr("stroke-width", 1.5);
    this.svg.append("text").attr("x", 68).attr("y", legendY + 4).attr("class", "annot")
      .attr("fill", "var(--rl-ink-muted)").text("ℓ∞");
    // ℓ1 diamond
    this.svg.append("polygon").attr("points", `${96},${legendY} ${101},${legendY-5} ${106},${legendY} ${101},${legendY+5}`)
      .attr("fill", "none").attr("stroke", "#9333ea").attr("stroke-width", 1.5);
    this.svg.append("text").attr("x", 112).attr("y", legendY + 4).attr("class", "annot")
      .attr("fill", "var(--rl-ink-muted)").text("ℓ₁");

    // Right panel: readouts + toggle
    const info = document.createElement("div");
    info.style.minWidth = "180px";
    info.style.paddingTop = "8px";
    wrap.appendChild(info);

    const makeRow = (label: string) => {
      const row = document.createElement("div");
      row.style.marginBottom = "10px";
      const lbl = document.createElement("div");
      lbl.className = "rl-mono";
      lbl.style.fontSize = "11px";
      lbl.style.color = "var(--rl-ink-faint)";
      lbl.textContent = label;
      const val = document.createElement("div");
      val.style.fontSize = "22px";
      val.style.fontFamily = "var(--rl-font-mono)";
      val.style.color = "var(--rl-ink)";
      val.style.fontWeight = "600";
      row.appendChild(lbl);
      row.appendChild(val);
      info.appendChild(row);
      return val;
    };

    this.l2El = makeRow("‖x‖₂  Euclidean");
    this.linfEl = makeRow("‖x‖∞  sup-norm");
    this.l1El = makeRow("‖x‖₁  taxicab");

    const domRow = document.createElement("div");
    domRow.style.marginTop = "4px";
    domRow.style.fontSize = "12px";
    domRow.style.color = "var(--rl-ink-muted)";
    this.domCoordEl = document.createElement("span");
    domRow.appendChild(this.domCoordEl);
    info.appendChild(domRow);

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "rl-btn";
    toggleBtn.style.marginTop = "16px";
    toggleBtn.style.fontSize = "12px";
    toggleBtn.textContent = "Highlight max coord: OFF";
    toggleBtn.addEventListener("click", () => {
      this.highlight = !this.highlight;
      toggleBtn.textContent = `Highlight max coord: ${this.highlight ? "ON" : "OFF"}`;
      this.updateReadouts();
    });
    info.appendChild(toggleBtn);

    info.appendChild(Object.assign(document.createElement("p"), {
      style: "font-size:11px;color:var(--rl-ink-faint);margin-top:12px;line-height:1.5",
      textContent: "Drag the blue point. The ℓ∞ norm picks whichever coordinate is largest — that's the 'sup'.",
    }));

    // Drag behavior
    const drag = d3.drag<SVGCircleElement, unknown>()
      .on("drag", (event) => {
        this.ptX = (event.x - CX) / SCALE;
        this.ptY = -(event.y - CY) / SCALE;
        // clamp to visible area
        const maxR = 2.8;
        this.ptX = Math.max(-maxR, Math.min(maxR, this.ptX));
        this.ptY = Math.max(-maxR, Math.min(maxR, this.ptY));
        this.updatePoint();
      });
    this.dragCircle.call(drag as any);

    this.updatePoint();
  }

  private drawBalls() {
    // ℓ∞ unit ball — square [-1,1]²
    const sq = SCALE;
    this.svg.append("rect")
      .attr("x", CX - sq).attr("y", CY - sq).attr("width", 2 * sq).attr("height", 2 * sq)
      .attr("fill", "rgba(22,163,74,0.07)").attr("stroke", "#16a34a").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5 3");
    // ℓ2 unit ball — circle
    this.svg.append("circle")
      .attr("cx", CX).attr("cy", CY).attr("r", SCALE)
      .attr("fill", "rgba(37,99,235,0.06)").attr("stroke", "#2563eb").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5 3");
    // ℓ1 unit ball — diamond
    const d = SCALE;
    this.svg.append("polygon")
      .attr("points", `${CX},${CY - d} ${CX + d},${CY} ${CX},${CY + d} ${CX - d},${CY}`)
      .attr("fill", "rgba(147,51,234,0.05)").attr("stroke", "#9333ea").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5 3");
  }

  private toSVG(x: number, y: number): [number, number] {
    return [CX + x * SCALE, CY - y * SCALE];
  }

  private updatePoint() {
    const [sx, sy] = this.toSVG(this.ptX, this.ptY);
    this.dragCircle.attr("cx", sx).attr("cy", sy);
    this.svg.select("line.pt-line").attr("x1", CX).attr("y1", CY).attr("x2", sx).attr("y2", sy);
    this.updateReadouts();
  }

  private updateReadouts() {
    const x = this.ptX, y = this.ptY;
    const l2 = Math.sqrt(x * x + y * y);
    const linf = Math.max(Math.abs(x), Math.abs(y));
    const l1 = Math.abs(x) + Math.abs(y);
    this.l2El.textContent = l2.toFixed(4);
    this.l1El.textContent = l1.toFixed(4);

    if (this.highlight) {
      const domIsX = Math.abs(x) >= Math.abs(y);
      const domVal = domIsX ? x : y;
      const domName = domIsX ? "x₁" : "x₂";
      this.linfEl.innerHTML = `<span style="color:var(--contr-fixed-point)">${linf.toFixed(4)}</span>`;
      this.domCoordEl.textContent = `ℓ∞ picks ${domName} = ${domVal.toFixed(3)}`;
    } else {
      this.linfEl.textContent = linf.toFixed(4);
      this.domCoordEl.textContent = "";
    }
  }
}

customElements.define("norm-comparison", NormComparison);
