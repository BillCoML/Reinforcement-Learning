/**
 * V1 — Weather Chain Explorer. The lesson's warm-up: a literal weather chain
 * drawn as three colored nodes, a current-state marker that hops on "Step"
 * (with a ~250ms tween), running visit tallies, and an empirical-frequency bar
 * chart that converges to the stationary distribution π over many steps.
 */
import { createPanel, type PanelHandle } from "./PanelChrome";
import { TransitionGraph } from "./TransitionGraph";
import { DistributionBars } from "./markov-bars";
import { MarkovChain } from "../markov/chain";
import { weather } from "../markov/presets";
import { mulberry32 } from "../bandits/stats";
import { prefersReducedMotion } from "./base";

const NAMES = weather.stateNames!; // sunny / cloudy / rainy
const COLORS = ["var(--mc-sunny)", "var(--mc-cloudy)", "var(--mc-rainy)"];

export class WeatherChainExplorer extends HTMLElement {
  private chain = new MarkovChain(weather.P);
  private pi = this.chain.stationary();
  private rng = mulberry32(7);
  private state = 0;
  private counts = [0, 0, 0];
  private steps = 0;
  private busy = false;

  private panel!: PanelHandle;
  private graph!: TransitionGraph;
  private bars!: DistributionBars;
  private tallyEl!: HTMLElement;

  connectedCallback(): void {
    this.counts[this.state] = 1; // X_0 counts as a visit
    this.steps = 1;
    this.render();
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({ id: "weather-chain-explorer" });

    const row = document.createElement("div");
    row.className = "mc-row";

    const graphCol = document.createElement("div");
    graphCol.className = "mc-col";
    this.graph = new TransitionGraph(graphCol, {
      P: weather.P,
      stateNames: NAMES,
      stateColors: COLORS,
      currentState: this.state,
      width: 420,
      height: 320,
    });

    const barCol = document.createElement("div");
    barCol.className = "mc-col";
    const barTitle = document.createElement("div");
    barTitle.className = "axis-label";
    barTitle.style.cssText =
      "font-family:var(--rl-font-ui);font-size:11px;color:var(--rl-ink-muted);margin-bottom:4px";
    barTitle.textContent = "empirical visit frequency (green tick = π)";
    barCol.append(barTitle);
    // DistributionBars appends its own SVG into the container it's given.
    this.bars = new DistributionBars(barCol, { labels: NAMES, colors: COLORS, width: 360 });
    this.tallyEl = document.createElement("div");
    this.tallyEl.className = "mc-readout";
    barCol.append(this.tallyEl);

    row.append(graphCol, barCol);
    this.panel.body.append(this.buildControls(), row);
    this.appendChild(this.panel.panel);

    this.draw();
  }

  private buildControls(): HTMLElement {
    const c = document.createElement("div");
    c.className = "rl-controls";
    c.style.marginBottom = "12px";

    const stepBtn = document.createElement("button");
    stepBtn.className = "primary";
    stepBtn.textContent = "Step";
    stepBtn.addEventListener("click", () => this.step());

    const run100 = document.createElement("button");
    run100.textContent = "Run 100 steps";
    run100.addEventListener("click", () => this.runMany(100));

    const run1000 = document.createElement("button");
    run1000.textContent = "Run 1000 steps";
    run1000.addEventListener("click", () => this.runMany(1000));

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => this.reset());

    c.append(stepBtn, run100, run1000, resetBtn);
    return c;
  }

  private async step(): Promise<void> {
    if (this.busy) return;
    const from = this.state;
    const to = this.chain.sampleNext(from, this.rng);
    this.counts[to]++;
    this.steps++;
    this.state = to;
    if (!prefersReducedMotion()) {
      this.busy = true;
      await this.graph.animate(from, to, 250);
      this.busy = false;
    } else {
      this.graph.setCurrentState(to);
    }
    this.draw();
  }

  private runMany(n: number): void {
    if (this.busy) return;
    for (let i = 0; i < n; i++) {
      const to = this.chain.sampleNext(this.state, this.rng);
      this.counts[to]++;
      this.steps++;
      this.state = to;
    }
    this.graph.setCurrentState(this.state);
    this.draw();
  }

  private reset(): void {
    this.rng = mulberry32(7);
    this.state = 0;
    this.counts = [0, 0, 0];
    this.counts[0] = 1;
    this.steps = 1;
    this.graph.setCurrentState(0);
    this.draw();
  }

  private draw(): void {
    const freq = this.counts.map((c) => c / this.steps);
    this.bars.update(freq, this.pi);
    const tally = NAMES.map((n, i) => `${n}: ${this.counts[i]}`).join(" · ");
    this.tallyEl.innerHTML =
      `current: <strong style="color:${COLORS[this.state]}">${NAMES[this.state]}</strong><br>` +
      `${tally}<br>π = (${this.pi.map((x) => x.toFixed(3)).join(", ")})`;
    this.panel.setStatus(`t=${this.steps - 1}`);
  }
}

customElements.define("weather-chain-explorer", WeatherChainExplorer);
