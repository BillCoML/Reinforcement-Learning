/**
 * V4 — V^π Heatmap. The gridworld shaded by V^π(s) (diverging green↔red, the
 * shared value scale) with the numeric value in each cell. Pick a policy and
 * "Compute" runs the exact linear solve (I − γP^π)V = R^π; a sorted bar chart
 * and the ‖V‖∞ / max / min / mean summary appear alongside.
 */
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { buildGridworld } from "../mdp/gridworld";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { buildPolicySelect, resolvePolicy, savedPolicy, type PolicyKey } from "./mdp-shared";
import { makeValueColorScale, textColorOn, fmtV } from "./value-scale";
import { rc, type MDP } from "../mdp/types";

export class ValueHeatmap extends HTMLElement {
  private mdp: MDP = buildGridworld({ slippery: false, gamma: 0.9 });
  private gr!: GridworldRenderer;
  private key: PolicyKey = "uniform";
  private V: number[] = [];
  private bars!: HTMLElement;
  private summary!: HTMLElement;

  connectedCallback(): void {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "value-heatmap" });

    const controls = document.createElement("div");
    controls.className = "mdp-controls";
    const label = document.createElement("label");
    label.append("policy ");
    label.appendChild(
      buildPolicySelect((k) => {
        this.key = k;
        this.compute();
      }),
    );
    controls.appendChild(label);
    const compute = document.createElement("button");
    compute.type = "button";
    compute.textContent = "Compute V^π";
    compute.addEventListener("click", () => this.compute());
    controls.appendChild(compute);
    body.appendChild(controls);

    const row = document.createElement("div");
    row.className = "mdp-row";
    const gridCol = document.createElement("div");
    const sideCol = document.createElement("div");
    sideCol.className = "mdp-col";
    row.append(gridCol, sideCol);
    body.appendChild(row);

    this.V = policyEvaluationExact(this.mdp, resolvePolicy(this.mdp, this.key));
    this.gr = new GridworldRenderer(gridCol, { mdp: this.mdp, valueFn: this.V });

    this.bars = document.createElement("div");
    this.summary = document.createElement("div");
    this.summary.className = "mdp-readout";
    sideCol.append(this.bars, this.summary);

    // re-evaluate if the user saves a new policy in V2 while "custom" is selected
    savedPolicy.subscribe(() => {
      if (this.key === "saved") this.compute();
    });

    this.appendChild(panel);
    this.compute();
  }

  private compute(): void {
    this.V = policyEvaluationExact(this.mdp, resolvePolicy(this.mdp, this.key));
    this.gr.update({ mdp: this.mdp, valueFn: this.V }, { animate: true });
    this.renderBars();
  }

  private renderBars(): void {
    const scale = makeValueColorScale();
    const order = this.V.map((v, s) => ({ v, s }))
      .filter((d) => !this.mdp.terminals[d.s])
      .sort((a, b) => b.v - a.v);
    const max = Math.max(...this.V);
    const min = Math.min(...this.V);
    const span = Math.max(Math.abs(max), Math.abs(min), 1e-6);

    this.bars.innerHTML = `<p class="mdp-readout" style="margin:0 0 6px">V^π by state (sorted)</p>`;
    for (const { v, s } of order) {
      const { r, c } = rc(s);
      const line = document.createElement("div");
      line.style.cssText = "display:flex;align-items:center;gap:8px;margin:2px 0;font-family:var(--rl-font-ui);font-size:12px";
      const tag = document.createElement("span");
      tag.className = "mdp-stat";
      tag.style.width = "40px";
      tag.textContent = `(${r},${c})`;
      const track = document.createElement("div");
      track.style.cssText = "flex:1;height:14px;background:var(--rl-surface-2);border-radius:3px;position:relative;overflow:hidden";
      const fill = document.createElement("div");
      fill.style.cssText = `position:absolute;top:0;bottom:0;width:${((Math.abs(v) / span) * 100).toFixed(1)}%;background:${scale(v)};${v < 0 ? "right:50%" : "left:50%"}`;
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "mdp-stat";
      val.style.cssText = "width:52px;text-align:right";
      val.textContent = fmtV(v);
      val.style.color = textColorOn(scale(v)) === "#ffffff" ? scale(v) : "var(--rl-ink)";
      line.append(tag, track, val);
      this.bars.appendChild(line);
    }

    const mean = this.V.reduce((a, b) => a + b, 0) / this.V.length;
    const inf = Math.max(...this.V.map((x) => Math.abs(x)));
    this.summary.innerHTML =
      `<span class="mdp-stat">‖V^π‖∞ = ${inf.toFixed(3)}</span> · ` +
      `max ${max.toFixed(3)} · min ${min.toFixed(3)} · mean ${mean.toFixed(3)}. ` +
      `The center bar line is 0; green bars extend right (positive), red left (negative).`;
  }
}

customElements.define("value-heatmap", ValueHeatmap);
