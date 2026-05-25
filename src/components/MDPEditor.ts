/**
 * MDPEditor — a small side-panel for tweaking the gridworld's parameters
 * (discount γ, deterministic vs slippery transitions). A plain class embedded
 * by V1, V6, V7; emits the new GridworldOpts via onChange so the parent can
 * rebuild the MDP. Pit/goal stay fixed (the running example is locked).
 */
import type { GridworldOpts } from "../mdp/gridworld";

export interface MDPEditorProps {
  opts: GridworldOpts;
  showGamma?: boolean;
  showSlippery?: boolean;
  /** γ slider bounds (default [0.5, 0.99]). */
  gammaRange?: [number, number];
  onChange: (opts: GridworldOpts) => void;
}

export class MDPEditor {
  private opts: GridworldOpts;
  private props: MDPEditorProps;
  private gammaOut?: HTMLElement;

  constructor(container: HTMLElement, props: MDPEditorProps) {
    this.props = props;
    this.opts = { slippery: false, gamma: 0.9, ...props.opts };
    container.appendChild(this.build());
  }

  private emit(): void {
    this.props.onChange({ ...this.opts });
  }

  private build(): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "mdp-controls";

    if (this.props.showGamma ?? true) {
      const [lo, hi] = this.props.gammaRange ?? [0.5, 0.99];
      const label = document.createElement("label");
      label.append("γ ");
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = String(lo);
      slider.max = String(hi);
      slider.step = "0.01";
      slider.value = String(this.opts.gamma ?? 0.9);
      slider.setAttribute("aria-label", "discount factor gamma");
      const out = document.createElement("span");
      out.className = "mdp-stat";
      out.textContent = (this.opts.gamma ?? 0.9).toFixed(2);
      this.gammaOut = out;
      slider.addEventListener("input", () => {
        this.opts.gamma = Number(slider.value);
        out.textContent = this.opts.gamma.toFixed(2);
        this.emit();
      });
      label.append(slider, " ", out);
      wrap.appendChild(label);
    }

    if (this.props.showSlippery ?? true) {
      const seg = document.createElement("div");
      seg.className = "mdp-seg";
      const mk = (text: string, slippery: boolean) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = text;
        b.setAttribute("aria-label", `${text} transitions`);
        if (!!this.opts.slippery === slippery) b.classList.add("is-active");
        b.addEventListener("click", () => {
          this.opts.slippery = slippery;
          seg.querySelectorAll("button").forEach((x) => x.classList.remove("is-active"));
          b.classList.add("is-active");
          this.emit();
        });
        return b;
      };
      seg.append(mk("deterministic", false), mk("slippery", true));
      const label = document.createElement("label");
      label.append("transitions ");
      label.appendChild(seg);
      wrap.appendChild(label);
    }

    return wrap;
  }

  /** Programmatically reflect external opts (e.g. when a parent resets). */
  setGamma(gamma: number): void {
    this.opts.gamma = gamma;
    if (this.gammaOut) this.gammaOut.textContent = gamma.toFixed(2);
  }
}
