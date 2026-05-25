/**
 * V2 — Power-of-P Animator. A heatmap of Pⁿ with n on a slider (and an
 * "animate 1→30" button), beside the distribution (Pⁿ)_{i,·} from a selectable
 * start state i, drawn as bars with an optional π overlay. Two tabs: the
 * ergodic weather chain (rows collapse to π) and the periodic chain (Pⁿ flips
 * between I and the swap matrix forever).
 */
import { createPanel, type PanelHandle } from "./PanelChrome";
import { DistributionBars } from "./markov-bars";
import { MarkovChain } from "../markov/chain";
import { weather, periodic2 } from "../markov/presets";
import { prefersReducedMotion } from "./base";
import * as d3 from "d3";

type TabKey = "weather" | "periodic";

const NS = "http://www.w3.org/2000/svg";

export class PowerOfPAnimator extends HTMLElement {
  private tab: TabKey = "weather";
  private n = 1;
  private startState = 0;
  private showOverlay = true;
  private timer = 0;

  private chain!: MarkovChain;
  private pi!: number[];
  private names?: string[];

  private panel!: PanelHandle;
  private heatEl!: SVGSVGElement;
  private bars!: DistributionBars;
  private barWrap!: HTMLElement;
  private slider!: HTMLInputElement;
  private nLabel!: HTMLElement;
  private startWrap!: HTMLElement;

  connectedCallback(): void {
    this.loadTab("weather");
    this.render();
  }

  disconnectedCallback(): void {
    if (this.timer) window.clearInterval(this.timer);
  }

  private loadTab(tab: TabKey): void {
    this.tab = tab;
    const preset = tab === "weather" ? weather : periodic2;
    this.chain = new MarkovChain(preset.P);
    this.pi = this.chain.stationary();
    this.names = preset.stateNames;
    this.startState = 0;
    this.n = 1;
  }

  private get K(): number {
    return this.chain.K;
  }

  private label(i: number): string {
    return this.names?.[i] ?? String(i);
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({ id: "power-of-p-animator" });
    this.panel.body.append(this.buildTabs(), this.buildControls());

    const row = document.createElement("div");
    row.className = "mc-row";

    const heatCol = document.createElement("div");
    heatCol.className = "mc-col";
    const heatTitle = this.smallLabel("Pⁿ  (white = 0, dark = 1)");
    this.heatEl = document.createElementNS(NS, "svg");
    this.heatEl.setAttribute("width", "100%");
    this.heatEl.classList.add("rl-svg");
    heatCol.append(heatTitle, this.heatEl);

    this.barWrap = document.createElement("div");
    this.barWrap.className = "mc-col";

    row.append(heatCol, this.barWrap);
    this.panel.body.append(row);
    this.appendChild(this.panel.panel);

    this.rebuildBars();
    this.draw();
  }

  private smallLabel(text: string): HTMLElement {
    const d = document.createElement("div");
    d.className = "axis-label";
    d.style.cssText =
      "font-family:var(--rl-font-ui);font-size:11px;color:var(--rl-ink-muted);margin-bottom:4px";
    d.textContent = text;
    return d;
  }

  private rebuildBars(): void {
    this.barWrap.innerHTML = "";
    this.barWrap.append(this.smallLabel(`distribution from start state — (Pⁿ)_{i,·}`));
    this.bars = new DistributionBars(this.barWrap, {
      labels: this.names ?? d3.range(this.K).map(String),
      width: 340,
    });
    // start-state selector
    this.startWrap = document.createElement("div");
    this.startWrap.className = "rl-controls";
    this.startWrap.style.marginTop = "8px";
    this.startWrap.append("start i ");
    const sel = document.createElement("select");
    for (let i = 0; i < this.K; i++) {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = this.label(i);
      if (i === this.startState) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      this.startState = +sel.value;
      this.draw();
    });
    this.startWrap.appendChild(sel);
    this.barWrap.append(this.startWrap);
  }

  private buildTabs(): HTMLElement {
    const tabs = document.createElement("div");
    tabs.className = "rl-controls";
    tabs.style.marginBottom = "10px";
    const mk = (key: TabKey, text: string) => {
      const b = document.createElement("button");
      b.textContent = text;
      b.setAttribute("aria-pressed", String(this.tab === key));
      b.addEventListener("click", () => {
        if (this.tab === key) return;
        this.stopAnim();
        this.loadTab(key);
        this.render();
      });
      return b;
    };
    tabs.append(
      mk("weather", "weather (ergodic)"),
      mk("periodic", "periodic (oscillates)"),
    );
    return tabs;
  }

  private buildControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";
    c.style.marginBottom = "12px";

    const animBtn = document.createElement("button");
    animBtn.className = "primary";
    animBtn.textContent = "Animate n=1→30";
    animBtn.addEventListener("click", () => this.toggleAnim(animBtn));

    const nWrap = document.createElement("label");
    this.slider = document.createElement("input");
    this.slider.type = "range";
    this.slider.min = "0";
    this.slider.max = "30";
    this.slider.step = "1";
    this.slider.value = String(this.n);
    this.slider.addEventListener("input", () => {
      this.stopAnim();
      this.n = +this.slider.value;
      this.draw();
    });
    this.nLabel = document.createElement("span");
    this.nLabel.className = "rl-mono";
    this.nLabel.textContent = `n=${this.n}`;
    nWrap.append("n ", this.slider, this.nLabel);

    const overlayWrap = document.createElement("label");
    const ov = document.createElement("input");
    ov.type = "checkbox";
    ov.checked = this.showOverlay;
    ov.addEventListener("change", () => {
      this.showOverlay = ov.checked;
      this.draw();
    });
    overlayWrap.append(ov, " show π overlay");

    c.append(animBtn, nWrap, overlayWrap);
    return c;
  }

  private toggleAnim(btn: HTMLButtonElement): void {
    if (this.timer) {
      this.stopAnim();
      btn.textContent = "Animate n=1→30";
      return;
    }
    btn.textContent = "Stop";
    this.n = 1;
    const interval = prefersReducedMotion() ? 400 : 150;
    this.timer = window.setInterval(() => {
      this.n = this.n >= 30 ? 1 : this.n + 1;
      this.draw();
      if (this.n === 30) {
        this.stopAnim();
        btn.textContent = "Animate n=1→30";
      }
    }, interval);
  }

  private stopAnim(): void {
    if (this.timer) {
      window.clearInterval(this.timer);
      this.timer = 0;
    }
  }

  private draw(): void {
    this.slider.value = String(this.n);
    this.nLabel.textContent = `n=${this.n}`;
    const Pn = this.chain.pPower(this.n);
    this.drawHeatmap(Pn);
    const rowDist = Array.from({ length: this.K }, (_, j) => Pn.get(this.startState, j));
    this.bars.update(rowDist, this.showOverlay ? this.pi : undefined);
    this.panel.setStatus(`${this.tab} · n=${this.n}`);
  }

  private drawHeatmap(Pn: { get(i: number, j: number): number }): void {
    const K = this.K;
    const cell = K <= 3 ? 64 : 52;
    const leftPad = 50; // room for row labels (e.g. "cloudy")
    const topPad = 26;
    const W = leftPad + K * cell + 8;
    const H = topPad + K * cell + 8;
    const svg = d3.select(this.heatEl);
    svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
    const interp = d3.interpolateRgb("#ffffff", "#1c1e22");

    // column headers
    for (let j = 0; j < K; j++) {
      svg.append("text")
        .attr("x", leftPad + j * cell + cell / 2)
        .attr("y", topPad - 10)
        .attr("text-anchor", "middle")
        .attr("class", "annot")
        .style("font-family", "var(--rl-font-mono)")
        .style("font-size", "10px")
        .style("fill", "var(--rl-ink-faint)")
        .text(this.label(j));
    }
    for (let i = 0; i < K; i++) {
      svg.append("text")
        .attr("x", leftPad - 8)
        .attr("y", topPad + i * cell + cell / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "central")
        .attr("class", "annot")
        .style("font-family", "var(--rl-font-mono)")
        .style("font-size", "10px")
        .style("fill", "var(--rl-ink-faint)")
        .text(this.label(i));
      for (let j = 0; j < K; j++) {
        const v = Pn.get(i, j);
        const g = svg.append("g");
        g.append("rect")
          .attr("x", leftPad + j * cell)
          .attr("y", topPad + i * cell)
          .attr("width", cell - 3)
          .attr("height", cell - 3)
          .attr("rx", 3)
          .attr("fill", interp(v))
          .attr("stroke", "var(--rl-border)")
          .attr("stroke-width", 0.5);
        g.append("text")
          .attr("x", leftPad + j * cell + (cell - 3) / 2)
          .attr("y", topPad + i * cell + (cell - 3) / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .style("font-family", "var(--rl-font-mono)")
          .style("font-size", "11px")
          .style("fill", v > 0.55 ? "#fff" : "var(--rl-ink)")
          .text(v.toFixed(2));
      }
    }
  }
}

customElements.define("power-of-p-animator", PowerOfPAnimator);
