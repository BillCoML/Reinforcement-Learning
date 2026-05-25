/**
 * V4 — Counterexample Gallery. Three tabs showing the three ways the Banach
 * theorem can fail: c=1 translation, incomplete space (Newton for √2), and
 * multiple fixed points (x²).
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { iterate1D } from "../contractions/ops";
import { newtonIterates, fractionToString, fractionToDecimal } from "../contractions/newton";

const W = 720, H = 340;
const PAD = { l: 50, r: 20, t: 20, b: 36 };

export class CounterexampleGallery extends HTMLElement {
  private tab = 0;

  connectedCallback() { this.build(); }

  private build() {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "v4-counterexample-gallery" });
    this.appendChild(panel);

    // Tab bar
    const tabBar = document.createElement("div");
    tabBar.style.cssText = "display:flex;gap:2px;margin-bottom:12px;border-bottom:1px solid var(--rl-border);padding-bottom:0";
    body.appendChild(tabBar);

    const tabs = [
      { label: "c = 1 (translation)", id: 0 },
      { label: "Incomplete space (rationals)", id: 1 },
      { label: "Multiple fixed points (x²)", id: 2 },
    ];

    const content = document.createElement("div");
    body.appendChild(content);

    const tabBtns = tabs.map(t => {
      const btn = document.createElement("button");
      btn.className = "rl-btn";
      btn.style.cssText = "border-radius:6px 6px 0 0;border-bottom:none;margin-bottom:-1px;font-size:12px;padding:6px 12px";
      btn.textContent = t.label;
      btn.addEventListener("click", () => {
        this.tab = t.id;
        tabBtns.forEach((b, i) => {
          b.style.background = i === this.tab ? "var(--rl-surface)" : "";
          b.style.borderBottom = i === this.tab ? "1px solid var(--rl-surface)" : "";
          b.style.fontWeight = i === this.tab ? "600" : "400";
          b.style.color = i === this.tab ? "var(--rl-ink)" : "var(--rl-ink-muted)";
        });
        this.renderTab(content);
      });
      tabBar.appendChild(btn);
      return btn;
    });

    // Activate first tab
    tabBtns[0].click();
  }

  private renderTab(container: HTMLElement) {
    container.innerHTML = "";
    if (this.tab === 0) this.renderTab0(container);
    else if (this.tab === 1) this.renderTab1(container);
    else this.renderTab2(container);
  }

  // Tab 0: T(x) = x + 1, no fixed point
  private renderTab0(container: HTMLElement) {
    const desc = Object.assign(document.createElement("p"), {
      style: "font-size:13px;color:var(--rl-ink-muted);margin-bottom:8px",
      innerHTML: "<strong>T(x) = x + 1</strong> preserves distances (c = 1). The iteration <em>x, x+1, x+2, …</em> diverges — no fixed point exists because x + 1 = x has no solution.",
    });
    container.appendChild(desc);

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);
    const svg = d3.select(svgEl);

    const traj = iterate1D(x => x + 1, 0, 12);
    const xMin = -1, xMax = 14;
    const gW = W - PAD.l - PAD.r, gH = H - PAD.t - PAD.b;
    const xSc = d3.scaleLinear([xMin, xMax], [PAD.l, PAD.l + gW]);
    const ySc = d3.scaleLinear([xMin - 0.5, xMax + 0.5], [PAD.t + gH, PAD.t]);

    svg.append("g").attr("transform", `translate(0,${PAD.t + gH})`).call(d3.axisBottom(xSc).ticks(7))
      .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
    svg.append("g").attr("transform", `translate(${PAD.l},0)`).call(d3.axisLeft(ySc).ticks(5))
      .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
    svg.selectAll("line,path").attr("stroke", "var(--rl-border)");

    // y = x
    svg.append("line").attr("x1", xSc(xMin)).attr("y1", ySc(xMin)).attr("x2", xSc(xMax)).attr("y2", ySc(xMax))
      .attr("stroke", "var(--rl-ink-faint)").attr("stroke-width", 1).attr("stroke-dasharray", "4 3");
    // T(x) = x+1
    svg.append("line").attr("x1", xSc(xMin)).attr("y1", ySc(xMin + 1)).attr("x2", xSc(xMax - 1)).attr("y2", ySc(xMax))
      .attr("stroke", "var(--contr-output)").attr("stroke-width", 2.5);
    svg.append("text").attr("x", xSc(8)).attr("y", ySc(9.5)).attr("class", "annot")
      .attr("fill", "var(--contr-output)").text("T(x) = x+1");
    svg.append("text").attr("x", xSc(6)).attr("y", ySc(5.5)).attr("class", "annot")
      .attr("fill", "var(--rl-ink-faint)").text("y = x");

    // "No fixed point" annotation (lines are parallel)
    svg.append("text").attr("x", PAD.l + 10).attr("y", PAD.t + 16)
      .attr("class", "annot").attr("fill", "var(--contr-warn)")
      .text("Lines are parallel — no intersection → no fixed point");

    // Cobweb
    for (let k = 0; k < 10; k++) {
      const xk = traj[k], xk1 = traj[k + 1];
      const op = Math.max(0.2, 1 - k * 0.07);
      svg.append("line").attr("x1", xSc(xk)).attr("y1", ySc(xk)).attr("x2", xSc(xk)).attr("y2", ySc(xk1))
        .attr("stroke", "var(--contr-trajectory)").attr("stroke-width", 1.3).attr("opacity", op);
      svg.append("line").attr("x1", xSc(xk)).attr("y1", ySc(xk1)).attr("x2", xSc(xk1)).attr("y2", ySc(xk1))
        .attr("stroke", "var(--contr-trajectory)").attr("stroke-width", 1.3).attr("opacity", op);
    }
    // Starting point
    svg.append("circle").attr("cx", xSc(0)).attr("cy", ySc(0)).attr("r", 5)
      .attr("fill", "var(--contr-input)").attr("stroke", "white").attr("stroke-width", 1.5);
    svg.append("text").attr("x", xSc(0.2)).attr("y", ySc(0) - 6).attr("class", "annot")
      .attr("fill", "var(--contr-input)").text("x₀ = 0");
    // Arrow indicating divergence
    svg.append("text").attr("x", xSc(11)).attr("y", ySc(11) + 16).attr("class", "annot")
      .attr("fill", "var(--contr-warn)").text("→ ∞");
  }

  // Tab 1: Newton's method for √2 on ℚ — rational iterates as fractions
  private renderTab1(container: HTMLElement) {
    const desc = Object.assign(document.createElement("p"), {
      style: "font-size:13px;color:var(--rl-ink-muted);margin-bottom:8px",
      innerHTML: "<strong>T(x) = (x + 2/x)/2</strong> on <strong>ℚ</strong> converges to √2 ≈ 1.41421…, but √2 ∉ ℚ. The iteration \"escapes\" the rational number line.",
    });
    container.appendChild(desc);

    const fracs = newtonIterates(8);
    const fracStrs = fracs.map(fractionToString);
    const fracVals = fracs.map(fractionToDecimal);
    const sqrt2 = Math.sqrt(2);

    // Fraction display table
    const tbl = document.createElement("table");
    tbl.style.cssText = "margin-bottom:12px;font-size:12px;border-collapse:collapse";
    const hdr = tbl.insertRow();
    ["k", "xₖ (fraction)", "xₖ (decimal)", "|xₖ − √2|"].forEach(h => {
      const th = document.createElement("th");
      th.textContent = h;
      th.style.cssText = "padding:3px 10px;color:var(--rl-ink-faint);font-weight:400;text-align:right;border-bottom:1px solid var(--rl-border)";
      hdr.appendChild(th);
    });
    fracs.slice(0, 7).forEach((_f, k) => {
      const row = tbl.insertRow();
      row.style.borderBottom = "1px solid var(--rl-border)";
      const vals = [String(k), fracStrs[k], fracVals[k].toFixed(8), Math.abs(fracVals[k] - sqrt2).toExponential(3)];
      vals.forEach(v => {
        const td = row.insertCell();
        td.className = "rl-mono";
        td.textContent = v;
        td.style.cssText = "padding:2px 10px;text-align:right";
      });
    });
    container.appendChild(tbl);

    // Number line viz
    const lineW = W - 80, lineX0 = 40, lineY = 80;
    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} 120`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);
    const svg = d3.select(svgEl);

    const xSc = d3.scaleLinear([1, 2], [lineX0, lineX0 + lineW]);
    svg.append("line").attr("x1", lineX0).attr("y1", lineY).attr("x2", lineX0 + lineW).attr("y2", lineY)
      .attr("stroke", "var(--rl-border)").attr("stroke-width", 1.5);
    svg.append("text").attr("x", xSc(1)).attr("y", lineY + 14).attr("class", "annot")
      .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-faint)").text("1");
    svg.append("text").attr("x", xSc(1.5)).attr("y", lineY + 14).attr("class", "annot")
      .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-faint)").text("1.5");
    svg.append("text").attr("x", xSc(2)).attr("y", lineY + 14).attr("class", "annot")
      .attr("text-anchor", "middle").attr("fill", "var(--rl-ink-faint)").text("2");

    // √2 — irrational, shown as dashed marker
    svg.append("line").attr("x1", xSc(sqrt2)).attr("y1", lineY - 30).attr("x2", xSc(sqrt2)).attr("y2", lineY + 4)
      .attr("stroke", "var(--contr-fixed-point)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3");
    svg.append("text").attr("x", xSc(sqrt2)).attr("y", lineY - 35).attr("class", "annot")
      .attr("text-anchor", "middle").attr("fill", "var(--contr-fixed-point)").text("√2 (irrational — not in ℚ)");

    // Rational iterates
    fracVals.slice(0, 7).forEach((v, k) => {
      if (v < 1 || v > 2) return;
      const op = 0.4 + 0.1 * k;
      svg.append("circle").attr("cx", xSc(v)).attr("cy", lineY).attr("r", 5)
        .attr("fill", "var(--contr-input)").attr("opacity", op);
      if (k < 4) {
        svg.append("text").attr("x", xSc(v)).attr("y", lineY + 22)
          .attr("text-anchor", "middle").attr("class", "annot")
          .attr("fill", "var(--contr-input)").attr("opacity", op)
          .text(fracStrs[k].length < 10 ? fracStrs[k] : `x${k}`);
      }
    });

    svg.append("text").attr("x", W / 2).attr("y", lineY - 50).attr("text-anchor", "middle")
      .attr("class", "annot").attr("fill", "var(--rl-ink-faint)")
      .text("Rational iterates (blue dots) converge toward √2 but never reach it — it's not in ℚ");
  }

  // Tab 2: T(x) = x², multiple fixed points
  private renderTab2(container: HTMLElement) {
    const desc = Object.assign(document.createElement("p"), {
      style: "font-size:13px;color:var(--rl-ink-muted);margin-bottom:8px",
      innerHTML: "<strong>T(x) = x²</strong> on [0,1] has two fixed points: 0 and 1. |T'(x)| = 2x can be ≥ 1 near x = 1, so it's <em>not</em> a contraction on the full interval. Different starting points converge to different fixed points.",
    });
    container.appendChild(desc);

    let x0 = 0.9;
    const gW = W - PAD.l - PAD.r, gH = H - PAD.t - PAD.b - 20;
    const xSc = d3.scaleLinear([0, 1], [PAD.l, PAD.l + gW]);
    const ySc = d3.scaleLinear([0, 1], [PAD.t + gH, PAD.t]);

    const sliderRow = document.createElement("div");
    sliderRow.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:12px";
    const lbl = Object.assign(document.createElement("label"), { textContent: "x₀:" });
    const sl = document.createElement("input");
    sl.type = "range"; sl.min = "0"; sl.max = "0.99"; sl.step = "0.01"; sl.value = String(x0);
    sl.style.width = "140px";
    const valEl = Object.assign(document.createElement("span"), { className: "rl-mono", style: "font-size:12px" });
    sliderRow.appendChild(lbl); sliderRow.appendChild(sl); sliderRow.appendChild(valEl);
    container.appendChild(sliderRow);

    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H - 20}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);

    const draw = () => {
      const traj = iterate1D(x => x * x, x0, 20);
      const dest = Math.abs(traj[20]) < 0.01 ? 0 : 1;
      valEl.textContent = `${x0.toFixed(2)} → ${dest === 0 ? "0" : "1"} (${dest === 0 ? "stable" : "stays at 1"})`;
      const svg = d3.select(svgEl);
      svg.selectAll("*").remove();

      svg.append("g").attr("transform", `translate(0,${PAD.t + gH})`).call(d3.axisBottom(xSc).ticks(5))
        .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
      svg.append("g").attr("transform", `translate(${PAD.l},0)`).call(d3.axisLeft(ySc).ticks(5))
        .selectAll("text").attr("fill", "var(--rl-ink-faint)").style("font-size", "10px");
      svg.selectAll("line,path").attr("stroke", "var(--rl-border)");

      // y = x
      svg.append("line").attr("x1", xSc(0)).attr("y1", ySc(0)).attr("x2", xSc(1)).attr("y2", ySc(1))
        .attr("stroke", "var(--rl-ink-faint)").attr("stroke-width", 1).attr("stroke-dasharray", "4 3");

      // T(x) = x²
      const pts = d3.range(0, 1.01, 0.005).map(x => ({ x, y: x * x }));
      const line = d3.line<{ x: number; y: number }>().x(p => xSc(p.x)).y(p => ySc(p.y));
      svg.append("path").datum(pts).attr("d", line).attr("fill", "none")
        .attr("stroke", "var(--contr-output)").attr("stroke-width", 2.5);
      svg.append("text").attr("x", xSc(0.85)).attr("y", ySc(0.6))
        .attr("class", "annot").attr("fill", "var(--contr-output)").text("T(x) = x²");

      // Fixed points: 0 and 1
      [0, 1].forEach(fp => {
        svg.append("circle").attr("cx", xSc(fp)).attr("cy", ySc(fp)).attr("r", 6)
          .attr("fill", "none").attr("stroke", "var(--contr-fixed-point)").attr("stroke-width", 2.5);
        svg.append("text").attr("x", xSc(fp) + (fp === 0 ? 8 : -32)).attr("y", ySc(fp) - 6)
          .attr("class", "annot").attr("fill", "var(--contr-fixed-point)").text(`x* = ${fp}`);
      });

      // Unstable FP annotation
      svg.append("text").attr("x", PAD.l + gW / 2).attr("y", PAD.t + 12)
        .attr("text-anchor", "middle").attr("class", "annot").attr("fill", "var(--contr-warn)")
        .text("Not a global contraction: |T'(x)| = 2x ≥ 1 near x = 1");

      // Cobweb
      const shown = traj.slice(0, 15);
      for (let k = 0; k < shown.length - 1; k++) {
        const xk = shown[k], xk1 = shown[k + 1];
        const op = Math.max(0.2, 1 - k * 0.05);
        svg.append("line").attr("x1", xSc(xk)).attr("y1", ySc(xk)).attr("x2", xSc(xk)).attr("y2", ySc(xk1))
          .attr("stroke", "var(--contr-trajectory)").attr("stroke-width", 1.3).attr("opacity", op);
        svg.append("line").attr("x1", xSc(xk)).attr("y1", ySc(xk1)).attr("x2", xSc(xk1)).attr("y2", ySc(xk1))
          .attr("stroke", "var(--contr-trajectory)").attr("stroke-width", 1.3).attr("opacity", op);
      }
      svg.append("circle").attr("cx", xSc(x0)).attr("cy", ySc(x0)).attr("r", 5)
        .attr("fill", "var(--contr-input)").attr("stroke", "white").attr("stroke-width", 1.5);
    };

    sl.addEventListener("input", () => { x0 = parseFloat(sl.value); draw(); });
    draw();
  }
}

customElements.define("counterexample-gallery", CounterexampleGallery);
