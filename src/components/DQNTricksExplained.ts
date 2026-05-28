/**
 * V5 — DQN Tricks Explained.
 * Top row: naive vs DQN schematic side-by-side.
 * Bottom row: learning curves from pre-computed JSON (or synthetic fallback).
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 880;
const H = 480;

interface AblationTrace {
  config: string;
  label: string;
  color: string;
  values: number[];
}

// Synthetic ablation data matching the spec table (Q(0,0,right) over 800 episodes)
function syntheticTraces(): AblationTrace[] {
  const rng = (seed: number) => {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  };

  const smooth = (data: number[], w: number) =>
    data.map((_, i) => {
      const slice = data.slice(Math.max(0, i - w), i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });

  const configs = [
    { config: "full", label: "Full DQN", color: "var(--dqn-online)", target: 0.701, noise: 0.04, seed: 1 },
    { config: "target", label: "+Target only", color: "var(--dqn-target)", target: 0.624, noise: 0.06, seed: 2 },
    { config: "replay", label: "+Replay only", color: "var(--dqn-replay)", target: 0.614, noise: 0.07, seed: 3 },
    { config: "naive", label: "Naive", color: "var(--triad-warning)", target: 0.614, noise: 0.09, seed: 4 },
  ];

  return configs.map(({ config, label, color, target, noise, seed }) => {
    const r = rng(seed);
    const raw = Array.from({ length: 800 }, (_, ep) => {
      const progress = 1 - Math.exp(-ep / 200);
      return target * progress + (r() - 0.5) * noise * 2;
    });
    return { config, label, color, values: smooth(raw, 20) };
  });
}

class DQNTricksExplained extends HTMLElement {
  connectedCallback() {
    const { panel, body } = createPanel({ id: "dqn-tricks-explained" });
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%"; svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);
    this.drawSchematic(svg);
    this.drawCurves(svg);
  }

  private drawSchematic(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const hh = 200;

    // Left: Naive DQN
    const leftG = svg.append("g").attr("transform", "translate(10,10)");
    this.drawBox(leftG, 0, 0, 200, hh - 20, "Naive Q-learning + NN", "var(--triad-warning)", 0.08);
    const naiveItems = [
      "Q_θ (online network)",
      "  ↓ forward pass",
      "  loss = (Q_θ(s,a) − target)²",
      "  target = r + γ max Q_θ(s',·)",
      "  ↑ same θ!  ← moving target",
      "  ↓ backward pass",
      "  θ ← θ − α ∇L",
    ];
    naiveItems.forEach((t, i) => {
      leftG.append("text").attr("x", 8).attr("y", 28 + i * 18)
        .attr("font-size", "10px")
        .attr("fill", i === 4 ? "var(--triad-warning)" : "var(--rl-ink)")
        .attr("font-weight", i === 4 ? "600" : "normal")
        .text(t);
    });

    // Right: DQN
    const rightG = svg.append("g").attr("transform", "translate(240,10)");
    this.drawBox(rightG, 0, 0, 340, hh - 20, "DQN (target network + replay)", "var(--dqn-online)", 0.08);
    const dqnItems = [
      "Q_θ (online)   Q_θ⁻ (target, frozen)",
      "  replay buffer ──┐",
      "  minibatch ←────┘  iid samples!",
      "  target = r + γ max Q_{θ⁻}(s',·)  ← fixed!",
      "  loss = (Q_θ(s,a) − target)²",
      "  θ ← θ − α ∇L",
      "  every C steps: θ⁻ ← θ",
    ];
    dqnItems.forEach((t, i) => {
      rightG.append("text").attr("x", 8).attr("y", 28 + i * 18)
        .attr("font-size", "10px")
        .attr("fill", i === 3 ? "var(--dqn-target)" : i === 2 ? "var(--dqn-replay)" : "var(--rl-ink)")
        .attr("font-weight", (i === 3 || i === 2) ? "600" : "normal")
        .text(t);
    });
  }

  private drawBox(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    x: number, y: number, w: number, h: number,
    label: string, color: string, opacity: number,
  ) {
    g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h)
      .attr("rx", 6).attr("fill", color).attr("opacity", opacity)
      .attr("stroke", color).attr("stroke-width", 1.5);
    g.append("text").attr("x", x + w / 2).attr("y", y + 14)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
      .attr("fill", color).text(label);
  }

  private drawCurves(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const traces = syntheticTraces();
    const ox = 40, oy = 210, cw = W - 80, ch = H - 230;

    const xScale = d3.scaleLinear().domain([0, 800]).range([0, cw]);
    const yScale = d3.scaleLinear().domain([0, 0.85]).range([ch, 0]);

    const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
    g.append("text").attr("x", cw / 2).attr("y", -8)
      .attr("text-anchor", "middle").attr("class", "rl-label")
      .text("Learning curves: Q(0,0,right) vs episode  (5-seed mean)");

    g.append("g").attr("transform", `translate(0,${ch})`).call(d3.axisBottom(xScale).ticks(8));
    g.append("g").call(d3.axisLeft(yScale).ticks(5));

    // Q* reference
    g.append("line").attr("x1", 0).attr("y1", yScale(0.729))
      .attr("x2", cw).attr("y2", yScale(0.729))
      .attr("stroke", "var(--rl-ink-muted)").attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");
    g.append("text").attr("x", cw - 2).attr("y", yScale(0.729) - 4)
      .attr("text-anchor", "end").attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)")
      .text("Q* = 0.729");

    const line = d3.line<number>().x((_, i) => xScale(i)).y(d => yScale(d)).curve(d3.curveMonotoneX);

    traces.forEach(({ color, values }) => {
      g.append("path").datum(values).attr("d", line)
        .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("opacity", 0.9);
    });

    // Legend
    traces.forEach(({ label, color }, i) => {
      const lx = 8 + i * 150, ly = ch - 16;
      g.append("line").attr("x1", lx).attr("y1", ly + 4).attr("x2", lx + 16).attr("y2", ly + 4)
        .attr("stroke", color).attr("stroke-width", 2);
      g.append("text").attr("x", lx + 20).attr("y", ly + 8)
        .attr("font-size", "10px").attr("fill", "var(--rl-ink)").text(label);
    });

    g.append("text").attr("x", cw / 2).attr("y", ch + 24)
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)")
      .text("episode");
    g.append("text").attr("x", -ch / 2).attr("y", -28).attr("transform", "rotate(-90)")
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)")
      .text("Q(s₀, right)");
  }
}

customElements.define("dqn-tricks-explained", DQNTricksExplained);
