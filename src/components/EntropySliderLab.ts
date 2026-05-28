/**
 * V6 — Entropy Slider Lab (centerpiece). Six synchronized panels driven by one α slider.
 * Loads three JSON files at page load; slider snaps to nearest pre-computed α.
 * All panels update synchronously when slider moves.
 */
import "./EntropySliderLab/PanelPolicyHeatmap";
import "./EntropySliderLab/PanelValueBifurcation";
import "./EntropySliderLab/PanelEntropyKL";
import "./EntropySliderLab/PanelGoalDiagnostic";
import "./EntropySliderLab/PanelLengthHistogram";
import "./EntropySliderLab/PanelAutoCaption";

import type { PanelPolicyHeatmap } from "./EntropySliderLab/PanelPolicyHeatmap";
import type { PanelValueBifurcation } from "./EntropySliderLab/PanelValueBifurcation";
import type { PanelEntropyKL } from "./EntropySliderLab/PanelEntropyKL";
import type { PanelGoalDiagnostic } from "./EntropySliderLab/PanelGoalDiagnostic";
import type { PanelLengthHistogram } from "./EntropySliderLab/PanelLengthHistogram";
import type { PanelAutoCaption } from "./EntropySliderLab/PanelAutoCaption";

interface SweepEntry {
  alpha: number;
  Q: number[][];
  V_soft: number[];
  pi: number[][];
  V_pi: number[];
  meanEntropy: number;
  klToUniform: number;
}

interface RolloutEntry {
  alpha: number;
  goalReachProb: number;
  pitReachProb: number;
  timeoutProb: number;
  meanStepsToTerminal: number;
  lengthHistogram: number[];
}

const MAX_STEPS = 500;

customElements.define(
  "entropy-slider-lab",
  class extends HTMLElement {
    private sweepData: SweepEntry[] = [];
    private rolloutData: RolloutEntry[] = [];
    private alphaIdx = 20;

    // Panel refs
    private panelA!: PanelPolicyHeatmap;
    private panelB!: PanelValueBifurcation;
    private panelC!: PanelEntropyKL;
    private panelD!: PanelGoalDiagnostic;
    private panelE!: PanelLengthHistogram;
    private panelF!: PanelAutoCaption;
    private slider!: HTMLInputElement;
    private alphaLabel!: HTMLElement;

    async connectedCallback() {
      this.innerHTML = "";
      this.style.display = "block";

      // Outer chrome
      const wrapper = document.createElement("div");
      wrapper.className = "panel-chrome";
      wrapper.style.cssText = "background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:10px;padding:20px;";
      this.appendChild(wrapper);

      const title = document.createElement("h3");
      title.className = "panel-title";
      title.style.cssText = "margin:0 0 16px;font-size:1em;color:var(--rl-ink);";
      title.textContent = "Entropy Slider Lab";
      wrapper.appendChild(title);

      // Global α slider
      const sliderRow = document.createElement("div");
      sliderRow.className = "controls-row";
      sliderRow.style.cssText = "margin-bottom:20px;";
      sliderRow.innerHTML = `
        <div style="width:100%">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <label style="font-weight:600;font-family:var(--rl-font-ui);font-size:0.9em;">Temperature α</label>
            <span id="esl-alpha-val" style="font-family:var(--rl-font-mono);font-size:0.95em;color:var(--maxent-soft);font-weight:600;">—</span>
          </div>
          <input type="range" id="esl-slider" min="0" max="60" step="1" value="20"
            style="width:100%;accent-color:var(--maxent-soft);"
            aria-label="Temperature alpha slider">
          <div style="display:flex;justify-content:space-between;font-size:0.75em;color:var(--rl-ink-muted);margin-top:4px;">
            <span>←Tie-breaking</span><span>Useful</span><span>Trade-off</span><span>Failure→</span>
          </div>
        </div>
      `;
      wrapper.appendChild(sliderRow);

      this.slider = wrapper.querySelector("#esl-slider") as HTMLInputElement;
      this.alphaLabel = wrapper.querySelector("#esl-alpha-val") as HTMLElement;

      // Two-row layout: top row (A, B, C), bottom row (D, E, F)
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;";
      wrapper.appendChild(grid);

      const makeCell = (title: string) => {
        const cell = document.createElement("div");
        cell.style.cssText = "background:var(--rl-surface-2);border:1px solid var(--rl-border);border-radius:8px;padding:10px;";
        const h = document.createElement("div");
        h.style.cssText = "font-size:0.8em;color:var(--rl-ink-muted);font-family:var(--rl-font-ui);font-weight:600;margin-bottom:6px;";
        h.textContent = title;
        cell.appendChild(h);
        return cell;
      };

      const cellA = makeCell("A — Policy heatmap");
      const cellB = makeCell("B — Value bifurcation");
      const cellC = makeCell("C — Entropy & KL");
      const cellD = makeCell("D — Goal-reach");
      const cellE = makeCell("E — Step lengths");
      const cellF = makeCell("F — What is happening");

      grid.append(cellA, cellB, cellC, cellD, cellE, cellF);

      this.panelA = document.createElement("esl-panel-policy-heatmap") as PanelPolicyHeatmap;
      this.panelB = document.createElement("esl-panel-value-bifurcation") as PanelValueBifurcation;
      this.panelC = document.createElement("esl-panel-entropy-kl") as PanelEntropyKL;
      this.panelD = document.createElement("esl-panel-goal-diagnostic") as PanelGoalDiagnostic;
      this.panelE = document.createElement("esl-panel-length-histogram") as PanelLengthHistogram;
      this.panelF = document.createElement("esl-panel-auto-caption") as PanelAutoCaption;

      cellA.appendChild(this.panelA);
      cellB.appendChild(this.panelB);
      cellC.appendChild(this.panelC);
      cellD.appendChild(this.panelD);
      cellE.appendChild(this.panelE);
      cellF.appendChild(this.panelF);

      // Load data
      const loading = document.createElement("p");
      loading.style.cssText = "text-align:center;color:var(--rl-ink-muted);";
      loading.textContent = "Loading data…";
      wrapper.appendChild(loading);

      try {
        const [sweep, rollouts] = await Promise.all([
          fetch("/data/maxent/alpha_sweep.json").then(r => r.json()),
          fetch("/data/maxent/maxent_rollouts.json").then(r => r.json()),
        ]);
        this.sweepData = sweep;
        this.rolloutData = rollouts;
        loading.remove();
      } catch {
        loading.textContent = "Data unavailable — run scripts/maxent_traces.py first.";
        return;
      }

      // Update slider max
      this.slider.max = String(this.sweepData.length - 1);

      // Pre-compute full-range data for curves
      const allAlphas = this.sweepData.map(d => d.alpha);
      const allVsoft = this.sweepData.map(d => d.V_soft[0]);
      const allVpi = this.sweepData.map(d => d.V_pi[0]);
      const allEntropy = this.sweepData.map(d => d.meanEntropy);
      const allKL = this.sweepData.map(d => d.klToUniform);

      this.panelB.setFullData(allAlphas, allVsoft, allVpi);
      this.panelC.setFullData(allAlphas, allEntropy, allKL);

      // Slider handler
      this.slider.addEventListener("input", () => {
        this.alphaIdx = +this.slider.value;
        this.updateAll();
      });

      // Keyboard support
      this.slider.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft" && this.alphaIdx > 0) {
          this.alphaIdx--;
          this.slider.value = String(this.alphaIdx);
          this.updateAll();
        } else if (e.key === "ArrowRight" && this.alphaIdx < this.sweepData.length - 1) {
          this.alphaIdx++;
          this.slider.value = String(this.alphaIdx);
          this.updateAll();
        }
      });

      this.updateAll();
    }

    private updateAll() {
      const si = Math.min(this.alphaIdx, this.sweepData.length - 1);
      const ri = Math.min(this.alphaIdx, this.rolloutData.length - 1);
      const s = this.sweepData[si];
      const r = this.rolloutData[ri];

      this.alphaLabel.textContent = `α = ${s.alpha.toFixed(4)}`;

      try { this.panelA.update(s.pi, s.V_soft); }
      catch(e) { console.error("Panel A error:", e); }
      this.panelB.updateCursor(s.alpha);
      this.panelC.updateCursor(s.alpha);
      try { this.panelD.update(r.goalReachProb, r.timeoutProb, r.pitReachProb); }
      catch(e) { console.error("Panel D error:", e); }
      try { this.panelE.update(r.lengthHistogram, MAX_STEPS); }
      catch(e) { console.error("Panel E error:", e); }
      this.panelF.update(s.alpha);
    }
  },
);
