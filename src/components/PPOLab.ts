/**
 * V6 — PPO Lab (centerpiece).
 * Five synchronized panels: gridworld, learning curves, clip fraction, KL trace, ratio histogram.
 * Controls: lr, batch, epochs, clip ε, GAE λ, algorithm toggle, seed, play/pause/step/reset.
 * Default config: aggressive regime (lr=2.0, batch=5, epochs=10) to show clipping in action.
 * Keyboard: Space = play/pause, ArrowRight = step, r = reset.
 */
import { buildGridworld } from "../mdp/gridworld";
import { mulberry32 } from "../td/helpers";
import { ppoUpdate, collectBatch } from "../ppo/ppo";
import { vanillaPGUpdate } from "../ppo/vanilla";
import type { PPOState, PPOIterationLog, PPOConfig, Trajectory } from "../ppo/types";
import "./PPOLab/PanelGridworld";
import "./PPOLab/PanelLearningCurves";
import "./PPOLab/PanelClipFraction";
import "./PPOLab/PanelKLTrace";
import "./PPOLab/PanelRatioHistogram";

import type { PanelGridworld } from "./PPOLab/PanelGridworld";
import type { PanelLearningCurves } from "./PPOLab/PanelLearningCurves";
import type { PanelClipFraction } from "./PPOLab/PanelClipFraction";
import type { PanelKLTrace } from "./PPOLab/PanelKLTrace";
import type { PanelRatioHistogram } from "./PPOLab/PanelRatioHistogram";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

const DEFAULTS = {
  lrPolicy: 2.0,
  lrValue: 1.0,
  clipEps: 0.2,
  gaeLambda: 0.95,
  epochs: 10,
  batchEpisodes: 5,
  entropyCoef: 0.01,
  valueCoef: 0.5,
  normalizeAdvantages: true,
  seed: 0,
  algo: "ppo" as "ppo" | "vanilla",
};

customElements.define(
  "ppo-lab",
  class extends HTMLElement {
    private state: PPOState = { theta: new Float64Array(mdp.nS * mdp.nA), V: new Float64Array(mdp.nS) };
    private logs: PPOIterationLog[] = [];
    private latestRatios: Float64Array = new Float64Array(0);
    private iter = 0;
    private playing = false;
    private rafId = 0;
    private lastStepTime = 0;
    private rng!: () => number;
    private currentConfig = { ...DEFAULTS };

    private panelGW!: PanelGridworld;
    private panelLC!: PanelLearningCurves;
    private panelCF!: PanelClipFraction;
    private panelKL!: PanelKLTrace;
    private panelRH!: PanelRatioHistogram;
    private iterLabel!: HTMLSpanElement;
    private playBtn!: HTMLButtonElement;
    private lrInput!: HTMLInputElement;
    private batchSel!: HTMLSelectElement;
    private epochsSel!: HTMLSelectElement;
    private clipInput!: HTMLInputElement;
    private lambdaInput!: HTMLInputElement;
    private algoSel!: HTMLSelectElement;
    private seedInput!: HTMLInputElement;
    private statusBar!: HTMLDivElement;

    connectedCallback() {
      this.render();
      this.reset();
      this.setupKeyboard();
    }

    disconnectedCallback() {
      this.playing = false;
      cancelAnimationFrame(this.rafId);
    }

    private setupKeyboard() {
      document.addEventListener("keydown", (e: KeyboardEvent) => {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
        if (e.key === " ") { e.preventDefault(); this.togglePlay(); }
        else if (e.key === "ArrowRight") { e.preventDefault(); this.step(); }
        else if (e.key === "r" || e.key === "R") { this.reset(); }
      });
    }

    private render() {
      this.innerHTML = "";
      this.style.cssText = "display:block;";

      const wrapper = document.createElement("div");
      wrapper.style.cssText = "max-width:960px;margin:24px auto;background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:10px;padding:16px;";

      // Title.
      const titleRow = document.createElement("div");
      titleRow.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;";
      const titleEl = document.createElement("p");
      titleEl.style.cssText = "font-weight:700;font-size:16px;margin:0;";
      titleEl.textContent = "V6 — PPO Lab";

      this.iterLabel = document.createElement("span");
      this.iterLabel.style.cssText = "font-size:13px;color:var(--rl-ink-muted);font-variant-numeric:tabular-nums;";
      this.iterLabel.textContent = "Iter: 0";
      titleRow.appendChild(titleEl);
      titleRow.appendChild(this.iterLabel);
      wrapper.appendChild(titleRow);

      // Main grid.
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:280px 1fr;grid-template-rows:auto auto auto;gap:12px;";

      // Panel A — spans 2 rows on left.
      this.panelGW = document.createElement("ppo-panel-gridworld") as PanelGridworld;
      this.panelGW.style.cssText = "grid-row:1/3;";
      grid.appendChild(this.panelGW);

      // Panel B — learning curves.
      this.panelLC = document.createElement("ppo-panel-learning-curves") as PanelLearningCurves;
      grid.appendChild(this.panelLC);

      // Panel C — clip fraction.
      this.panelCF = document.createElement("ppo-panel-clip-fraction") as PanelClipFraction;
      grid.appendChild(this.panelCF);

      // Row 2: KL trace + ratio histogram.
      const row2 = document.createElement("div");
      row2.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:12px;";

      this.panelKL = document.createElement("ppo-panel-kl-trace") as PanelKLTrace;
      row2.appendChild(this.panelKL);

      this.panelRH = document.createElement("ppo-panel-ratio-histogram") as PanelRatioHistogram;
      row2.appendChild(this.panelRH);
      grid.appendChild(row2);

      wrapper.appendChild(grid);

      // Controls.
      const controls = this.buildControls();
      wrapper.appendChild(controls);

      // Status bar.
      this.statusBar = document.createElement("div");
      this.statusBar.style.cssText = "margin-top:8px;font-size:12px;color:var(--rl-ink-muted);min-height:18px;";
      wrapper.appendChild(this.statusBar);

      this.appendChild(wrapper);
    }

    private buildControls(): HTMLElement {
      const ctrl = document.createElement("div");
      ctrl.style.cssText = "margin-top:14px;padding:10px;background:#f8f6f0;border-radius:8px;border:1px solid #e5e1d5;";

      const row1 = document.createElement("div");
      row1.style.cssText = "display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:8px;";

      // Algorithm.
      this.algoSel = this.makeSelect("Algorithm:", ["ppo", "vanilla"], ["PPO (clipped)", "Vanilla PG (no clip)"], row1);
      this.algoSel.value = DEFAULTS.algo;

      // Learning rate.
      this.lrInput = this.makeRangeInput("LR:", 0.05, 5.0, 0.05, DEFAULTS.lrPolicy, row1);

      // Batch size.
      this.batchSel = this.makeSelect("Batch:", ["5", "10", "20", "50"], ["5 ep", "10 ep", "20 ep", "50 ep"], row1);
      this.batchSel.value = String(DEFAULTS.batchEpisodes);

      // Epochs.
      this.epochsSel = this.makeSelect("Epochs:", ["1", "2", "4", "8", "10", "16"], ["1", "2", "4", "8", "10", "16"], row1);
      this.epochsSel.value = String(DEFAULTS.epochs);

      const row2 = document.createElement("div");
      row2.style.cssText = "display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:8px;";

      // Clip epsilon.
      this.clipInput = this.makeRangeInput("Clip ε:", 0.05, 0.5, 0.05, DEFAULTS.clipEps, row2);

      // GAE lambda.
      this.lambdaInput = this.makeRangeInput("GAE λ:", 0.0, 1.0, 0.05, DEFAULTS.gaeLambda, row2);

      // Seed.
      this.seedInput = this.makeRangeInput("Seed:", 0, 9, 1, DEFAULTS.seed, row2);
      this.seedInput.step = "1";

      const row3 = document.createElement("div");
      row3.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;align-items:center;";

      // Playback buttons.
      this.playBtn = document.createElement("button");
      this.playBtn.textContent = "▶ Play";
      this.playBtn.style.cssText = this.btnStyle("#15803d");
      this.playBtn.addEventListener("click", () => this.togglePlay());

      const stepBtn = document.createElement("button");
      stepBtn.textContent = "⏭ Step";
      stepBtn.style.cssText = this.btnStyle("#0e7490");
      stepBtn.addEventListener("click", () => this.step());

      const resetBtn = document.createElement("button");
      resetBtn.textContent = "↺ Reset";
      resetBtn.style.cssText = this.btnStyle("#b45309");
      resetBtn.addEventListener("click", () => this.reset());

      const keyHint = document.createElement("span");
      keyHint.style.cssText = "font-size:11px;color:var(--rl-ink-muted);margin-left:6px;";
      keyHint.textContent = "Space=play, →=step, r=reset";

      row3.appendChild(this.playBtn);
      row3.appendChild(stepBtn);
      row3.appendChild(resetBtn);
      row3.appendChild(keyHint);

      ctrl.appendChild(row1);
      ctrl.appendChild(row2);
      ctrl.appendChild(row3);
      return ctrl;
    }

    private btnStyle(color: string): string {
      return `background:${color};color:#fff;border:none;padding:6px 14px;border-radius:5px;cursor:pointer;font-size:13px;font-weight:600;`;
    }

    private makeSelect(label: string, values: string[], labels: string[], container: HTMLElement): HTMLSelectElement {
      const wrap = document.createElement("label");
      wrap.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;";
      wrap.appendChild(Object.assign(document.createElement("span"), { textContent: label }));
      const sel = document.createElement("select");
      sel.style.cssText = "font-size:12px;padding:2px 5px;";
      values.forEach((v, i) => {
        const opt = document.createElement("option");
        opt.value = v; opt.textContent = labels[i];
        sel.appendChild(opt);
      });
      wrap.appendChild(sel);
      container.appendChild(wrap);
      return sel;
    }

    private makeRangeInput(label: string, min: number, max: number, step: number, val: number, container: HTMLElement): HTMLInputElement {
      const wrap = document.createElement("label");
      wrap.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;";
      wrap.appendChild(Object.assign(document.createElement("span"), { textContent: label }));
      const inp = document.createElement("input");
      inp.type = "range";
      inp.min = String(min); inp.max = String(max); inp.step = String(step); inp.value = String(val);
      inp.style.cssText = "width:80px;";
      const valSpan = document.createElement("span");
      valSpan.style.cssText = "font-variant-numeric:tabular-nums;min-width:34px;font-size:12px;";
      valSpan.textContent = String(val);
      inp.addEventListener("input", () => { valSpan.textContent = inp.value; });
      wrap.appendChild(inp); wrap.appendChild(valSpan);
      container.appendChild(wrap);
      return inp;
    }

    private getConfig(): PPOConfig & { seed: number; algo: "ppo" | "vanilla" } {
      return {
        lrPolicy: parseFloat(this.lrInput.value),
        lrValue: Math.min(parseFloat(this.lrInput.value) * 0.5, 1.0),
        clipEps: parseFloat(this.clipInput.value),
        gaeLambda: parseFloat(this.lambdaInput.value),
        epochs: parseInt(this.epochsSel.value, 10),
        batchEpisodes: parseInt(this.batchSel.value, 10),
        entropyCoef: 0.01,
        valueCoef: 0.5,
        normalizeAdvantages: true,
        seed: parseInt(this.seedInput.value, 10),
        algo: this.algoSel.value as "ppo" | "vanilla",
      };
    }

    private reset() {
      this.playing = false;
      cancelAnimationFrame(this.rafId);
      this.playBtn.textContent = "▶ Play";
      this.state = { theta: new Float64Array(mdp.nS * mdp.nA), V: new Float64Array(mdp.nS) };
      this.logs = [];
      this.latestRatios = new Float64Array(0);
      this.iter = 0;
      this.currentConfig = this.getConfig();
      this.rng = mulberry32(this.currentConfig.seed);
      this.updateAllPanels();
      this.statusBar.textContent = "Reset. Press Play or Step to begin.";
    }

    private step() {
      const config = this.currentConfig;
      const batch: Trajectory[] = collectBatch(this.state.theta, mdp, config.batchEpisodes, this.rng);

      if (config.algo === "vanilla") {
        const { state: newState, log } = vanillaPGUpdate(
          this.state,
          batch,
          {
            lrPolicy: config.lrPolicy,
            lrValue: config.lrValue,
            gaeLambda: config.gaeLambda,
            normalizeAdvantages: config.normalizeAdvantages,
            batchEpisodes: config.batchEpisodes,
          },
          mdp,
        );
        this.state = newState;
        // Create a PPOIterationLog-compatible object for vanilla.
        const fakeLog: PPOIterationLog = {
          iter: this.iter,
          vStart: log.vStart,
          meanKL: 0,
          clipFraction: 0,
          meanRatio: 1,
          maxRatio: 1,
          minRatio: 1,
          surrogateValue: log.vStart,
          entropyMean: 0,
          batchSize: log.batchSize,
        };
        this.logs.push(fakeLog);
        this.latestRatios = new Float64Array(0);
      } else {
        const { state: newState, log, ratios } = ppoUpdate(this.state, batch, config, mdp);
        this.state = newState;
        this.logs.push({ ...log, iter: this.iter });
        this.latestRatios = ratios;
      }

      this.iter++;
      this.updateAllPanels();

      const last = this.logs[this.logs.length - 1];
      this.statusBar.textContent =
        `Iter ${this.iter}: V(s₀)=${last.vStart.toFixed(4)}, ` +
        `clip=${(last.clipFraction * 100).toFixed(1)}%, ` +
        `KL=${last.meanKL.toFixed(5)}, batch=${last.batchSize} transitions`;
    }

    private updateAllPanels() {
      this.iterLabel.textContent = `Iter: ${this.iter}`;
      this.panelGW.update(this.state.theta, this.state.V, mdp.nA);
      this.panelLC.update(this.logs);
      this.panelCF.update(this.logs);
      this.panelKL.update(this.logs);
      this.panelRH.update(this.latestRatios, this.currentConfig.clipEps);
    }

    private togglePlay() {
      this.playing = !this.playing;
      this.playBtn.textContent = this.playing ? "⏸ Pause" : "▶ Play";
      if (this.playing) {
        this.lastStepTime = performance.now();
        this.playLoop();
      } else {
        cancelAnimationFrame(this.rafId);
      }
    }

    private playLoop() {
      if (!this.playing) return;
      const now = performance.now();
      if (now - this.lastStepTime >= 200) {
        this.step();
        this.lastStepTime = now;
      }
      this.rafId = requestAnimationFrame(() => this.playLoop());
    }
  },
);
