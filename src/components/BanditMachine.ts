/**
 * V1 — Bandit Machine. A tactile warm-up: three levers with hidden Bernoulli
 * means, pull them and feel the variance. Running empirical means update under
 * each lever; a log records the last pulls. "Show hidden means" reveals μ_i.
 */
import { createPanel } from "./PanelChrome";
import { prefersReducedMotion } from "./base";

const DEFAULT_MEANS = [0.3, 0.5, 0.7];
const COIN_GOLD = "#d4a017";

export class BanditMachine extends HTMLElement {
  private means: number[] = DEFAULT_MEANS;
  private N: number[] = [];
  private S: number[] = [];
  private log: { t: number; a: number; r: number }[] = [];
  private t = 0;
  private revealed = false;

  private armEls: HTMLElement[] = [];
  private meanEls: HTMLElement[] = [];
  private badgeEls: HTMLElement[] = [];
  private knobEls: SVGGElement[] = [];
  private coinLayers: SVGGElement[] = [];
  private logEl!: HTMLElement;
  private setStatus: (s: string) => void = () => {};

  connectedCallback(): void {
    const attr = this.getAttribute("means");
    if (attr) this.means = attr.split(",").map(Number);
    this.reset();
    this.render();
  }

  private reset(): void {
    const K = this.means.length;
    this.N = new Array(K).fill(0);
    this.S = new Array(K).fill(0);
    this.log = [];
    this.t = 0;
  }

  private render(): void {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "bandit-machine" });
    this.setStatus = setStatus;

    const stage = document.createElement("div");
    stage.className = "bandit-stage";

    const arms = document.createElement("div");
    arms.className = "bandit-arms";

    this.armEls = [];
    this.meanEls = [];
    this.badgeEls = [];
    this.knobEls = [];
    this.coinLayers = [];

    for (let i = 0; i < this.means.length; i++) {
      const card = document.createElement("div");
      card.className = "bandit-arm";

      const name = document.createElement("div");
      name.className = "arm-name";
      name.textContent = `arm ${i + 1}`;

      const lever = this.makeLever(i);

      const mean = document.createElement("div");
      mean.className = "running-mean";
      this.meanEls[i] = mean;

      const badge = document.createElement("div");
      badge.className = "truth-badge";
      this.badgeEls[i] = badge;

      // Explicit, keyboard-focusable control with a clear accessible name.
      const pullBtn = document.createElement("button");
      pullBtn.className = "arm-pull-btn";
      pullBtn.textContent = "Pull";
      pullBtn.setAttribute("aria-label", `Pull arm ${i + 1}`);

      card.append(name, lever, mean, badge, pullBtn);
      const pull = () => this.pull(i);
      // The whole card is a convenient mouse target; the button drives keyboard/AT.
      card.addEventListener("click", pull);
      pullBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        pull();
      });

      this.armEls[i] = card;
      arms.appendChild(card);
    }

    const logBox = document.createElement("div");
    logBox.className = "bandit-log";
    const logTitle = document.createElement("div");
    logTitle.className = "log-title";
    logTitle.textContent = "last pulls";
    this.logEl = document.createElement("div");
    logBox.append(logTitle, this.logEl);

    stage.append(arms, logBox);

    const controls = document.createElement("div");
    controls.className = "rl-controls";
    controls.style.marginTop = "16px";

    const revealBtn = document.createElement("button");
    revealBtn.textContent = "Show hidden means";
    revealBtn.setAttribute("aria-pressed", "false");
    revealBtn.addEventListener("click", () => {
      this.revealed = !this.revealed;
      revealBtn.setAttribute("aria-pressed", String(this.revealed));
      revealBtn.textContent = this.revealed ? "Hide means" : "Show hidden means";
      this.updateArms();
    });

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => {
      this.reset();
      this.updateArms();
      this.updateLog();
    });

    const hint = document.createElement("span");
    hint.style.color = "var(--rl-ink-muted)";
    hint.textContent = "click a lever (or press Enter) to pull";

    controls.append(revealBtn, resetBtn, hint);

    body.append(stage, controls);
    this.appendChild(panel);

    this.updateArms();
    this.updateLog();
  }

  private makeLever(i: number): SVGSVGElement {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", "70");
    svg.setAttribute("height", "118");
    svg.setAttribute("viewBox", "0 0 70 118");
    svg.classList.add("rl-svg");

    // base box
    const base = document.createElementNS(NS, "rect");
    base.setAttribute("x", "6");
    base.setAttribute("y", "70");
    base.setAttribute("width", "58");
    base.setAttribute("height", "42");
    base.setAttribute("rx", "6");
    base.setAttribute("fill", "var(--rl-surface-2)");
    base.setAttribute("stroke", "var(--rl-border)");

    // slot
    const slot = document.createElementNS(NS, "rect");
    slot.setAttribute("x", "20");
    slot.setAttribute("y", "92");
    slot.setAttribute("width", "30");
    slot.setAttribute("height", "5");
    slot.setAttribute("rx", "2.5");
    slot.setAttribute("fill", "var(--rl-ink-faint)");

    // lever track
    const track = document.createElementNS(NS, "rect");
    track.setAttribute("x", "31");
    track.setAttribute("y", "16");
    track.setAttribute("width", "8");
    track.setAttribute("height", "58");
    track.setAttribute("rx", "4");
    track.setAttribute("fill", "var(--rl-border)");

    // knob group (animated)
    const knob = document.createElementNS(NS, "g") as SVGGElement;
    const knobCircle = document.createElementNS(NS, "circle");
    knobCircle.setAttribute("cx", "35");
    knobCircle.setAttribute("cy", "16");
    knobCircle.setAttribute("r", "11");
    knobCircle.setAttribute("fill", "var(--rl-algo-ucb)");
    knob.appendChild(knobCircle);
    this.knobEls[i] = knob;

    // coin layer (animated coins appended here)
    const coinLayer = document.createElementNS(NS, "g") as SVGGElement;
    this.coinLayers[i] = coinLayer;

    svg.append(base, slot, track, knob, coinLayer);
    return svg;
  }

  private pull(i: number): void {
    const reward = Math.random() < this.means[i] ? 1 : 0;
    this.t += 1;
    this.N[i] += 1;
    this.S[i] += reward;
    this.log.unshift({ t: this.t, a: i + 1, r: reward });
    this.log = this.log.slice(0, 8);

    this.animatePull(i, reward);
    this.updateArms();
    this.updateLog();
  }

  private animatePull(i: number, reward: number): void {
    const card = this.armEls[i];
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 320);

    if (prefersReducedMotion()) {
      if (reward === 1) this.placeCoin(i, true);
      return;
    }

    const knob = this.knobEls[i];
    knob.animate(
      [
        { transform: "translateY(0px)" },
        { transform: "translateY(40px)" },
        { transform: "translateY(0px)" },
      ],
      { duration: 380, easing: "cubic-bezier(.34,1.4,.64,1)" },
    );
    if (reward === 1) {
      setTimeout(() => this.placeCoin(i, false), 160);
    }
  }

  private placeCoin(i: number, instant: boolean): void {
    const NS = "http://www.w3.org/2000/svg";
    const coin = document.createElementNS(NS, "circle");
    coin.setAttribute("cx", "35");
    coin.setAttribute("cy", "88");
    coin.setAttribute("r", "6");
    coin.setAttribute("fill", COIN_GOLD);
    coin.setAttribute("stroke", "#a87c10");
    this.coinLayers[i].appendChild(coin);
    if (instant) {
      setTimeout(() => coin.remove(), 600);
      return;
    }
    const anim = coin.animate(
      [
        { transform: "translateY(-6px)", opacity: 1 },
        { transform: "translateY(14px)", opacity: 0 },
      ],
      { duration: 600, easing: "ease-in" },
    );
    anim.onfinish = () => coin.remove();
  }

  private updateArms(): void {
    for (let i = 0; i < this.means.length; i++) {
      const n = this.N[i];
      const mu = n > 0 ? this.S[i] / n : NaN;
      this.meanEls[i].innerHTML = n
        ? `μ̂ = ${mu.toFixed(2)} <span class="sub">(n=${n})</span>`
        : `μ̂ = — <span class="sub">(n=0)</span>`;
      const badge = this.badgeEls[i];
      if (this.revealed) {
        badge.textContent = `μ = ${this.means[i].toFixed(2)}`;
        badge.classList.add("revealed");
      } else {
        badge.textContent = "μ = ?";
        badge.classList.remove("revealed");
      }
    }
    const total = this.t;
    this.setStatus(`pulls=${total}`);
  }

  private updateLog(): void {
    if (this.log.length === 0) {
      this.logEl.innerHTML = `<div class="log-row">— no pulls yet —</div>`;
      return;
    }
    this.logEl.innerHTML = this.log
      .map(
        (e) =>
          `<div class="log-row">[t=${String(e.t).padStart(2)}, a=${e.a}, r=<span class="r${e.r}">${e.r}</span>]</div>`,
      )
      .join("");
  }
}

customElements.define("bandit-machine", BanditMachine);
