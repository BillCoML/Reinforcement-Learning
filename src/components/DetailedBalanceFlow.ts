/**
 * V6 — Detailed Balance Flow. Nodes in a row; between each pair, two arrows
 * i→j and j→i whose thickness is the probability flow πᵢPᵢⱼ. "Compute π" finds
 * the stationary distribution and sizes the arrows; "Check detailed balance"
 * marks each pair ✓ when the opposing flows match (reversible) or ⚠ when they
 * differ. Presets: a reversible birth-death chain and an irreversible one.
 */
import * as d3 from "d3";
import { createPanel, type PanelHandle } from "./PanelChrome";
import { MarkovChain } from "../markov/chain";
import { birthDeath, asymmetric } from "../markov/presets";

const NS = "http://www.w3.org/2000/svg";
const STATE_VARS = ["--mc-state-1", "--mc-state-2", "--mc-state-3", "--mc-state-4"];
const W = 720;
const H = 320;

export class DetailedBalanceFlow extends HTMLElement {
  private P: number[][] = birthDeath.P.map((r) => r.slice());
  private pi: number[] | null = null;
  private showCheck = false;

  private panel!: PanelHandle;
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private readoutEl!: HTMLElement;

  connectedCallback(): void {
    this.render();
  }

  private get K(): number {
    return this.P.length;
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({ id: "detailed-balance-flow" });
    this.panel.body.append(this.buildControls());

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    this.svg = d3.select(svgEl as SVGSVGElement);
    this.svg.append("defs").append("marker")
      .attr("id", "db-arrow").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "var(--mc-edge)");

    this.readoutEl = document.createElement("div");
    this.readoutEl.className = "mc-readout";

    this.panel.body.append(wrap, this.readoutEl);
    this.appendChild(this.panel.panel);
    this.draw();
  }

  private buildControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";
    c.style.marginBottom = "10px";

    c.append("preset ");
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", "preset chain");
    for (const p of [birthDeath, asymmetric]) {
      const o = document.createElement("option");
      o.value = p.key;
      o.textContent = p.label;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      this.P = (sel.value === "asymmetric" ? asymmetric : birthDeath).P.map((r) => r.slice());
      this.pi = null;
      this.showCheck = false;
      this.draw();
    });
    c.appendChild(sel);

    const computeBtn = document.createElement("button");
    computeBtn.className = "primary";
    computeBtn.textContent = "Compute π";
    computeBtn.addEventListener("click", () => {
      this.pi = new MarkovChain(this.P).stationary();
      this.draw();
    });

    const checkBtn = document.createElement("button");
    checkBtn.textContent = "Check detailed balance";
    checkBtn.addEventListener("click", () => {
      if (!this.pi) this.pi = new MarkovChain(this.P).stationary();
      this.showCheck = true;
      this.draw();
    });

    c.append(computeBtn, checkBtn);
    return c;
  }

  private nodeX(i: number): number {
    const margin = 110;
    return margin + (i * (W - 2 * margin)) / (this.K - 1);
  }

  private draw(): void {
    const K = this.K;
    const y = H / 2;
    const r = 24;
    this.svg.selectAll("g.db-layer").remove();
    const root = this.svg.append("g").attr("class", "db-layer");

    const flow = (i: number, j: number) => (this.pi ? this.pi[i] * this.P[i][j] : null);
    const widthOf = (f: number | null) => (f == null ? 2 : Math.max(1.5, 1.5 + 26 * f));

    const pairResults: string[] = [];

    for (let i = 0; i < K; i++) {
      for (let j = i + 1; j < K; j++) {
        if (this.P[i][j] <= 1e-9 && this.P[j][i] <= 1e-9) continue;
        const adj = j - i;
        const off = 14 + (adj - 1) * 26;
        this.drawFlow(root, i, j, +1, -off, widthOf(flow(i, j))); // i→j, bow up
        this.drawFlow(root, j, i, -1, +off, widthOf(flow(j, i))); // j→i, bow down

        if (this.showCheck && this.pi) {
          const fij = flow(i, j)!;
          const fji = flow(j, i)!;
          const ok = Math.abs(fij - fji) < 1e-9;
          const mx = (this.nodeX(i) + this.nodeX(j)) / 2;
          const badge = root.append("g").attr("transform", `translate(${mx},${y - off - 16})`);
          badge.append("circle").attr("r", 9)
            .attr("fill", ok ? "var(--mc-tint-3)" : "rgba(185,28,28,0.16)")
            .attr("stroke", ok ? "var(--mc-stationary)" : "var(--mc-periodic)");
          badge.append("text").attr("text-anchor", "middle").attr("dominant-baseline", "central")
            .style("font-size", "11px").style("fill", ok ? "var(--mc-stationary)" : "var(--mc-periodic)")
            .text(ok ? "✓" : "⚠");
          pairResults.push(
            `(${i},${j}): π${i}P${i}${j} = ${fij.toFixed(4)} ${ok ? "=" : "≠"} π${j}P${j}${i} = ${fji.toFixed(4)} ${ok ? "✓" : "⚠"}`,
          );
        }
      }
    }

    // nodes on top
    for (let i = 0; i < K; i++) {
      const g = root.append("g").attr("transform", `translate(${this.nodeX(i)},${y})`);
      g.append("circle").attr("r", r).attr("fill", `var(${STATE_VARS[i % STATE_VARS.length]})`)
        .attr("stroke", "rgba(0,0,0,0.18)");
      g.append("text").attr("text-anchor", "middle").attr("dominant-baseline", "central")
        .style("font-family", "var(--rl-font-ui)").style("font-weight", "600").style("fill", "#fff")
        .text(String(i));
      if (this.pi)
        g.append("text").attr("y", r + 16).attr("text-anchor", "middle").attr("class", "annot")
          .style("font-family", "var(--rl-font-mono)").style("font-size", "11px")
          .style("fill", "var(--mc-stationary)").text(`π=${this.pi[i].toFixed(3)}`);
    }

    // readout
    if (!this.pi) {
      this.readoutEl.innerHTML = `Click <strong>Compute π</strong> to solve πP = π and size each arrow by the flow πᵢPᵢⱼ.`;
    } else if (!this.showCheck) {
      this.readoutEl.innerHTML =
        `π = (${this.pi.map((x) => x.toFixed(4)).join(", ")}). Arrow thickness = flow πᵢPᵢⱼ. ` +
        `Click <strong>Check detailed balance</strong> to compare opposing flows pairwise.`;
    } else {
      const resid = new MarkovChain(this.P).detailedBalanceResidual(this.pi);
      const reversible = resid < 1e-9;
      this.readoutEl.innerHTML =
        `<strong>${reversible ? "Reversible" : "Not reversible"}</strong> — max flow imbalance = ` +
        `<span class="rl-mono">${resid.toExponential(2)}</span>.<br>` +
        pairResults.join("<br>");
    }
    this.panel.setStatus(this.pi ? (this.showCheck ? "checked" : "π computed") : "ready");
  }

  private drawFlow(
    root: d3.Selection<SVGGElement, unknown, null, undefined>,
    from: number,
    to: number,
    _dir: number,
    bow: number,
    strokeW: number,
  ): void {
    const y = H / 2;
    const r = 24;
    const xa = this.nodeX(from);
    const xb = this.nodeX(to);
    const dx = Math.sign(xb - xa);
    const a = { x: xa + dx * r, y };
    const b = { x: xb - dx * r, y };
    const mx = (a.x + b.x) / 2;
    const path = `M${a.x},${a.y} Q${mx},${y + bow} ${b.x},${b.y}`;
    root.append("path").attr("d", path).attr("fill", "none")
      .attr("stroke", "var(--mc-edge)").attr("stroke-width", strokeW).attr("opacity", 0.8)
      .attr("marker-end", "url(#db-arrow)");
  }
}

customElements.define("detailed-balance-flow", DetailedBalanceFlow);
