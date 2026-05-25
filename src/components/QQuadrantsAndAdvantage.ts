/**
 * V5 — Q-Quadrants and Advantage. Each cell is split into four action triangles
 * shaded by Q^π(s,a) (or A^π = Q^π − V^π). The greenest quadrant is the best
 * action; the deep-red quadrants pointing into the pit show how costly a wrong
 * move is. Defaults to the optimal policy so the §5 worked example —
 * Q*((1,0),Right) = −1.000, A*((1,0),Right) = −1.810 — is reproducible on hover.
 */
import { createPanel } from "./PanelChrome";
import { GridworldRenderer } from "./GridworldRenderer";
import { buildGridworld } from "../mdp/gridworld";
import { policyEvaluationExact } from "../mdp/policy-evaluation";
import { qFromV, advantage } from "../mdp/q-and-advantage";
import { buildPolicySelect, resolvePolicy, savedPolicy, type PolicyKey } from "./mdp-shared";
import { type MDP } from "../mdp/types";

export class QQuadrantsAndAdvantage extends HTMLElement {
  private mdp: MDP = buildGridworld({ slippery: false, gamma: 0.9 });
  private gr!: GridworldRenderer;
  private key: PolicyKey = "optimal";
  private showAdvantage = false;
  private readout!: HTMLElement;

  connectedCallback(): void {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "q-quadrants-advantage" });

    const controls = document.createElement("div");
    controls.className = "mdp-controls";

    const polLabel = document.createElement("label");
    polLabel.append("policy ");
    const sel = buildPolicySelect((k) => {
      this.key = k;
      this.compute();
    });
    sel.value = "optimal";
    polLabel.appendChild(sel);
    controls.appendChild(polLabel);

    const seg = document.createElement("div");
    seg.className = "mdp-seg";
    const mk = (text: string, adv: boolean) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.setAttribute("aria-label", text);
      if (this.showAdvantage === adv) b.classList.add("is-active");
      b.addEventListener("click", () => {
        this.showAdvantage = adv;
        seg.querySelectorAll("button").forEach((x) => x.classList.remove("is-active"));
        b.classList.add("is-active");
        this.compute();
      });
      return b;
    };
    seg.append(mk("Q^π", false), mk("A^π = Q−V", true));
    const segLabel = document.createElement("label");
    segLabel.append("show ");
    segLabel.appendChild(seg);
    controls.appendChild(segLabel);
    body.appendChild(controls);

    const gridCol = document.createElement("div");
    body.appendChild(gridCol);
    this.gr = new GridworldRenderer(gridCol, { mdp: this.mdp, showQuadrants: true, qValues: this.computeQ() });

    this.readout = document.createElement("div");
    this.readout.className = "mdp-readout";
    body.appendChild(this.readout);

    savedPolicy.subscribe(() => {
      if (this.key === "saved") this.compute();
    });

    this.appendChild(panel);
    this.compute();
  }

  private computeQ(): number[][] {
    const V = policyEvaluationExact(this.mdp, resolvePolicy(this.mdp, this.key));
    const Q = qFromV(this.mdp, V);
    return this.showAdvantage ? advantage(Q, V) : Q;
  }

  private compute(): void {
    this.gr.update(
      { qValues: this.computeQ(), showQuadrants: true, advantageMode: this.showAdvantage },
      { animate: true },
    );
    const what = this.showAdvantage ? "A^π(s,a) = Q^π − V^π" : "Q^π(s,a)";
    this.readout.innerHTML =
      `Each quadrant shows <span class="mdp-stat">${what}</span> for one action; hover a quadrant for its value and next-state distribution. ` +
      (this.showAdvantage
        ? `The optimal action sits at ≈0 (neutral); negative (red) quadrants quantify how much each wrong move costs. Under the optimal policy, hover the <strong>Right</strong> quadrant of (1,0): A = −1.810.`
        : `The greenest quadrant is the best action. Under the optimal policy, hover the <strong>Right</strong> quadrant of (1,0) — it steps into the pit: Q = −1.000.`);
  }
}

customElements.define("q-quadrants-advantage", QQuadrantsAndAdvantage);
