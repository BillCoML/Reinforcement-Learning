/**
 * V4 — Q-Network Architecture schematic.
 * Static diagram: input (one-hot state), hidden layers, output (Q-values).
 * Backprop flow arrows and self-reference of target computation highlighted.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

const W = 720;
const H = 360;

class QNetworkArchitecture extends HTMLElement {
  connectedCallback() {
    const { panel, body } = createPanel({ id: "q-network-architecture" });
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%"; svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);
    this.draw(svg);
  }

  private draw(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const defs = svg.append("defs");
    // Arrow markers
    const addMarker = (id: string, color: string) => {
      defs.append("marker").attr("id", id)
        .attr("viewBox", "0 0 10 10").attr("refX", 5).attr("refY", 5)
        .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
        .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", color);
    };
    addMarker("arr-fwd", "var(--fa-linear)");
    addMarker("arr-bwd", "var(--triad-warning)");
    addMarker("arr-target", "var(--dqn-target)");

    // Layers config
    const layers = [
      { label: "Input", sublabel: "one-hot state\n(49 dims)", nodes: 5, color: "var(--fa-tabular)" },
      { label: "Hidden 1", sublabel: "linear(49→64)\nReLU", nodes: 5, color: "var(--fa-linear)" },
      { label: "Hidden 2", sublabel: "linear(64→64)\nReLU", nodes: 5, color: "var(--fa-linear)" },
      { label: "Output", sublabel: "Q(s,·)\n(4 actions)", nodes: 4, color: "var(--fa-neural)" },
    ];

    const lx = [60, 220, 380, 540];
    const nodeR = 12;
    const totalH = H - 60;

    // Draw connections (forward pass)
    layers.forEach((layer, li) => {
      if (li === layers.length - 1) return;
      const nextLayer = layers[li + 1];
      const n1 = layer.nodes, n2 = nextLayer.nodes;
      for (let i = 0; i < n1; i++) {
        for (let j = 0; j < n2; j++) {
          const y1 = 40 + ((i + 1) / (n1 + 1)) * totalH;
          const y2 = 40 + ((j + 1) / (n2 + 1)) * totalH;
          svg.append("line")
            .attr("x1", lx[li] + nodeR).attr("y1", y1)
            .attr("x2", lx[li + 1] - nodeR).attr("y2", y2)
            .attr("stroke", "var(--fa-linear)").attr("stroke-width", 0.5).attr("opacity", 0.2);
        }
      }
    });

    // Draw nodes
    layers.forEach((layer, li) => {
      const n = layer.nodes;
      for (let i = 0; i < n; i++) {
        const y = 40 + ((i + 1) / (n + 1)) * totalH;
        svg.append("circle").attr("cx", lx[li]).attr("cy", y).attr("r", nodeR)
          .attr("fill", layer.color).attr("stroke", "white").attr("stroke-width", 1.5);
      }

      // Layer label
      svg.append("text").attr("x", lx[li]).attr("y", H - 8)
        .attr("text-anchor", "middle").attr("font-size", "11px")
        .attr("font-weight", "600").attr("fill", "var(--rl-ink)")
        .text(layer.label);
      layer.sublabel.split("\n").forEach((line, si) => {
        svg.append("text").attr("x", lx[li]).attr("y", H - 8 + 14 * (si + 1))
          .attr("text-anchor", "middle").attr("font-size", "9px")
          .attr("fill", "var(--rl-ink-muted)").text(line);
      });
    });

    // Backprop arrow (right to left, red dashed)
    const bpy = 20;
    svg.append("path")
      .attr("d", `M ${lx[3] + nodeR + 8} ${bpy} L ${lx[0] - nodeR - 8} ${bpy}`)
      .attr("fill", "none").attr("stroke", "var(--triad-warning)")
      .attr("stroke-width", 2).attr("stroke-dasharray", "6,3")
      .attr("marker-end", "url(#arr-bwd)");
    svg.append("text").attr("x", (lx[0] + lx[3]) / 2).attr("y", bpy - 5)
      .attr("text-anchor", "middle").attr("font-size", "10px")
      .attr("fill", "var(--triad-warning)").text("backprop: ∇_θ L");

    // Target computation inset (right side)
    const tx = lx[3] + 60, ty = 30, tw = W - tx - 10;
    const tbox = svg.append("g").attr("transform", `translate(${tx},${ty})`);
    tbox.append("rect").attr("width", tw).attr("height", 140)
      .attr("rx", 6).attr("fill", "var(--dqn-target)").attr("opacity", 0.08)
      .attr("stroke", "var(--dqn-target)").attr("stroke-width", 1.5);
    [
      "Target computation:",
      "feed s' through same",
      "network → Q(s', ·)",
      "target = r + γ·max Q(s',·)",
      "",
      "⚠ same θ used for both",
      "prediction & target!",
    ].forEach((line, i) => {
      tbox.append("text").attr("x", tw / 2).attr("y", 20 + i * 17)
        .attr("text-anchor", "middle").attr("font-size", "10px")
        .attr("fill", i >= 5 ? "var(--triad-warning)" : "var(--dqn-target)")
        .attr("font-weight", i >= 5 ? "600" : "normal")
        .text(line);
    });

    // Arrow from output to target box
    svg.append("line")
      .attr("x1", lx[3] + nodeR).attr("y1", 40 + totalH / 2)
      .attr("x2", tx).attr("y2", ty + 70)
      .attr("stroke", "var(--dqn-target)").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,2").attr("marker-end", "url(#arr-target)");
  }
}

customElements.define("dqn-q-network-architecture", QNetworkArchitecture);
