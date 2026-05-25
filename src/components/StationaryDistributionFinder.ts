/**
 * V4 — Stationary Distribution Finder. Edit a chain on the left; on the right,
 * compute π three ways via tabs: (1) linear algebra — the balance system πⱼ =
 * Σᵢ πᵢ Pᵢⱼ rendered with the live coefficients; (2) power iteration — animate
 * μₜ = μ₀ Pᵗ toward π (or watch it oscillate); (3) eigendecomposition — the
 * spectrum of Pᵀ with the unit eigenvalue highlighted. A verify panel shows πP
 * and the residual ‖πP − π‖.
 */
import { createPanel, type PanelHandle } from "./PanelChrome";
import { ChainEditor } from "./ChainEditor";
import { DistributionBars } from "./markov-bars";
import { mathToHTML } from "./MathBlock";
import { MarkovChain, complexAbs, totalVariation } from "../markov/chain";
import { weather } from "../markov/presets";
import { prefersReducedMotion } from "./base";

type Mode = "linalg" | "power" | "eigen";

export class StationaryDistributionFinder extends HTMLElement {
  private P: number[][] = weather.P.map((r) => r.slice());
  private mode: Mode = "linalg";
  private t = 0;
  private timer = 0;

  private panel!: PanelHandle;
  private bars!: DistributionBars;
  private barTitle!: HTMLElement;
  private explainEl!: HTMLElement;
  private verifyEl!: HTMLElement;
  private modeBar!: HTMLElement;

  connectedCallback(): void {
    this.render();
  }

  disconnectedCallback(): void {
    this.stop();
  }

  private chain(): MarkovChain {
    return new MarkovChain(this.P);
  }

  private get K(): number {
    return this.P.length;
  }

  private render(): void {
    this.innerHTML = "";
    this.panel = createPanel({
      id: "stationary-distribution-finder",
      heavy: true,
      mobileNotice: "The stationary-distribution finder is interactive — view on a wider screen.",
    });

    const row = document.createElement("div");
    row.className = "mc-row";

    const left = document.createElement("div");
    left.className = "mc-col";
    new ChainEditor(left, {
      P: this.P,
      maxStates: 5,
      minStates: 2,
      onChange: (P) => {
        this.P = P;
        this.t = 0;
        this.refresh();
      },
    });

    const right = document.createElement("div");
    right.className = "mc-col";
    this.modeBar = this.buildModeBar();
    this.barTitle = document.createElement("div");
    this.barTitle.className = "axis-label";
    this.barTitle.style.cssText =
      "font-family:var(--rl-font-ui);font-size:11px;color:var(--rl-ink-muted);margin:6px 0 4px";
    right.append(this.modeBar, this.barTitle);
    this.bars = new DistributionBars(right, { width: 340 });
    this.explainEl = document.createElement("div");
    this.explainEl.className = "mc-readout";
    right.append(this.explainEl);

    row.append(left, right);

    this.verifyEl = document.createElement("div");
    this.verifyEl.className = "mc-readout";
    this.verifyEl.style.borderTop = "1px solid var(--rl-border)";
    this.verifyEl.style.marginTop = "14px";
    this.verifyEl.style.paddingTop = "12px";

    this.panel.body.append(row, this.verifyEl);
    this.appendChild(this.panel.panel);

    this.refresh();
  }

  private buildModeBar(): HTMLElement {
    const bar = document.createElement("div");
    bar.className = "rl-controls";
    const mk = (m: Mode, label: string) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.dataset.mode = m;
      b.setAttribute("aria-pressed", String(this.mode === m));
      b.addEventListener("click", () => {
        this.stop();
        this.mode = m;
        this.t = 0;
        for (const btn of bar.querySelectorAll("button"))
          btn.setAttribute("aria-pressed", String(btn.getAttribute("data-mode") === m));
        this.refresh();
      });
      return b;
    };
    bar.append(
      mk("linalg", "linear algebra"),
      mk("power", "power iteration"),
      mk("eigen", "eigendecomposition"),
    );
    return bar;
  }

  private refresh(): void {
    let chain: MarkovChain;
    try {
      chain = this.chain();
    } catch {
      this.explainEl.textContent = "Invalid matrix.";
      return;
    }
    const pi = chain.stationary();

    if (this.mode === "linalg") this.refreshLinalg(chain, pi);
    else if (this.mode === "power") this.refreshPower(chain, pi);
    else this.refreshEigen(chain, pi);

    // verify πP = π
    const piP = this.applyPi(pi);
    const resid = Math.max(...piP.map((v, i) => Math.abs(v - pi[i])));
    this.verifyEl.innerHTML =
      `<strong>Verify πP = π.</strong> ` +
      `πP = (${piP.map((x) => x.toFixed(4)).join(", ")}) · ` +
      `residual ‖πP − π‖∞ = <span class="rl-mono">${resid.toExponential(2)}</span> ` +
      (resid < 1e-6 ? `<span class="mc-badge" style="background:var(--mc-tint-3);color:var(--mc-stationary)">✓ stationary</span>` : "");
    this.panel.setStatus(`${this.mode} · ${this.K} states`);
  }

  private applyPi(pi: number[]): number[] {
    const out = new Array<number>(this.K).fill(0);
    for (let j = 0; j < this.K; j++)
      for (let i = 0; i < this.K; i++) out[j] += pi[i] * this.P[i][j];
    return out;
  }

  private refreshLinalg(_chain: MarkovChain, pi: number[]): void {
    this.barTitle.textContent = "computed π (green tick = π)";
    this.bars.update(pi, pi);
    // Build the balance system with live coefficients: π_j = Σ_i P_{ij} π_i.
    const rows: string[] = [];
    for (let j = 0; j < this.K; j++) {
      const terms = [];
      for (let i = 0; i < this.K; i++) {
        const c = this.P[i][j];
        if (c <= 1e-9) continue;
        terms.push(`${c.toFixed(2)}\\,\\pi_${i}`);
      }
      rows.push(`\\pi_${j} &= ${terms.join(" + ") || "0"}`);
    }
    rows.push(`1 &= ${Array.from({ length: this.K }, (_, i) => `\\pi_${i}`).join(" + ")}`);
    const tex = `\\begin{aligned}${rows.join(" \\\\ ")}\\end{aligned}`;
    this.explainEl.innerHTML =
      `<p>Solve the balance equations (one is redundant — replace it with the normalization):</p>` +
      `<div style="overflow-x:auto">${mathToHTML(tex, true)}</div>` +
      `<p>Solution: π = (${pi.map((x) => x.toFixed(4)).join(", ")}).</p>`;
  }

  private refreshPower(chain: MarkovChain, pi: number[]): void {
    const mu0 = new Array<number>(this.K).fill(1 / this.K);
    const mu = chain.distributionAfter(mu0, this.t);
    const tv = totalVariation(mu, pi);
    this.barTitle.textContent = `power iteration: μₜ = μ₀Pᵗ at t=${this.t} (green tick = π)`;
    this.bars.update(mu, pi);
    this.explainEl.innerHTML =
      `<div class="rl-controls" style="margin-bottom:8px">` +
      `<button data-act="play" class="primary">${this.timer ? "Pause" : "Play"}</button>` +
      `<button data-act="step">Step</button>` +
      `<button data-act="reset">Reset</button>` +
      `<label>t <input type="range" min="0" max="60" step="1" value="${this.t}" data-act="slider"><span class="rl-mono">t=${this.t}</span></label>` +
      `</div>` +
      `<p>‖μₜ − π‖<sub>TV</sub> = <span class="rl-mono">${tv.toExponential(2)}</span>. ` +
      `Starting from uniform μ₀; for an ergodic chain this drives to π, for a periodic chain it oscillates.</p>`;
    // wire controls
    const play = this.explainEl.querySelector<HTMLButtonElement>('[data-act="play"]')!;
    play.onclick = () => this.togglePlay();
    this.explainEl.querySelector<HTMLButtonElement>('[data-act="step"]')!.onclick = () => {
      this.stop();
      this.t = Math.min(60, this.t + 1);
      this.refresh();
    };
    this.explainEl.querySelector<HTMLButtonElement>('[data-act="reset"]')!.onclick = () => {
      this.stop();
      this.t = 0;
      this.refresh();
    };
    const slider = this.explainEl.querySelector<HTMLInputElement>('[data-act="slider"]')!;
    slider.oninput = () => {
      this.stop();
      this.t = +slider.value;
      this.refresh();
    };
  }

  private togglePlay(): void {
    if (this.timer) {
      this.stop();
      this.refresh();
      return;
    }
    const interval = prefersReducedMotion() ? 450 : 220;
    this.timer = window.setInterval(() => {
      this.t = this.t >= 60 ? 0 : this.t + 1;
      this.refresh();
    }, interval);
    this.refresh();
  }

  private stop(): void {
    if (this.timer) {
      window.clearInterval(this.timer);
      this.timer = 0;
    }
  }

  private refreshEigen(chain: MarkovChain, pi: number[]): void {
    this.barTitle.textContent = "π from the unit left-eigenvector (green tick = π)";
    this.bars.update(pi, pi);
    const eigs = chain
      .eigenvalues()
      .map((c) => ({ c, mag: complexAbs(c) }))
      .sort((a, b) => b.mag - a.mag);
    const list = eigs
      .map((e, k) => {
        const isUnit = Math.abs(e.mag - 1) < 1e-6 && k === 0;
        const val = Math.abs(e.c.im) < 1e-9
          ? e.c.re.toFixed(4)
          : `${e.c.re.toFixed(3)} ${e.c.im >= 0 ? "+" : "−"} ${Math.abs(e.c.im).toFixed(3)}i`;
        const style = isUnit
          ? `background:var(--mc-tint-3);color:var(--mc-stationary)`
          : `background:var(--rl-surface-2);color:var(--rl-ink-muted)`;
        return `<span class="mc-badge" style="${style}">λ=${val}</span>`;
      })
      .join(" ");
    this.explainEl.innerHTML =
      `<p>Eigenvalues of P (sorted by magnitude). The unit eigenvalue carries the ` +
      `stationary left-eigenvector, normalized to π:</p><p>${list}</p>` +
      `<p>π = (${pi.map((x) => x.toFixed(4)).join(", ")}).</p>`;
  }
}

customElements.define("stationary-distribution-finder", StationaryDistributionFinder);
