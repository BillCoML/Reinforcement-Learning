/**
 * V7 — Inference Graph. Static D3 graphical model diagram: s_t → a_t → O_t,
 * with unrolled view showing forward/backward messages. Hover shows tooltip.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface Node {
  id: string;
  label: string;
  type: "latent" | "observed";
  x: number;
  y: number;
  tooltip: string;
}

interface Edge {
  from: string;
  to: string;
  label?: string;
  dashed?: boolean;
}

customElements.define(
  "inference-graph",
  class extends HTMLElement {
    connectedCallback() {
      const { panel, body } = createPanel({ id: "inference-graph" });
      this.appendChild(panel);

      const W = 540, H = 280;
      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      body.appendChild(wrap);

      const tip = document.createElement("div");
      tip.className = "rl-tooltip";
      tip.style.maxWidth = "260px";
      tip.style.whiteSpace = "normal";
      wrap.appendChild(tip);

      const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svgEl.setAttribute("width", "100%");
      svgEl.classList.add("rl-svg");
      wrap.insertBefore(svgEl, tip);

      const svg = d3.select(svgEl);

      // Defs: arrow marker
      svg.append("defs").append("marker")
        .attr("id", "ig-arrow").attr("viewBox", "0 0 10 10")
        .attr("refX", 9).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "#94a3b8");

      svg.append("defs").append("marker")
        .attr("id", "ig-arrow-back").attr("viewBox", "0 0 10 10")
        .attr("refX", 9).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "var(--maxent-soft)");

      // Three time steps, unrolled
      const T = 3;
      const cols = [90, 210, 330];
      const rowS = 60, rowA = 150, rowO = 240;

      const nodes: Node[] = [];
      const edges: Edge[] = [];

      cols.forEach((cx, t) => {
        const tStr = String(t);
        nodes.push({
          id: `s${t}`, label: `s_${t}`, type: "latent",
          x: cx, y: rowS,
          tooltip: `p(s_{${tStr}+1} | s_${tStr}, a_${tStr}) — dynamics. The state transitions deterministically according to the MDP.`,
        });
        nodes.push({
          id: `a${t}`, label: `a_${t}`, type: "latent",
          x: cx, y: rowA,
          tooltip: `Prior: p(a_${tStr} | s_${tStr}) = 1/|A| — uniform over actions. The posterior given O=1 gives the Boltzmann policy.`,
        });
        nodes.push({
          id: `O${t}`, label: `O_${t}`, type: "observed",
          x: cx, y: rowO,
          tooltip: `Optimality variable O_${tStr} ∈ {0,1}. Observed value: O_${tStr}=1. Likelihood: p(O=1|s_${tStr},a_${tStr}) = exp(r(s_${tStr},a_${tStr})/α).`,
        });

        // s_t → a_t (prior)
        edges.push({ from: `s${t}`, to: `a${t}` });
        // a_t → O_t (likelihood)
        edges.push({ from: `a${t}`, to: `O${t}` });
        // s_t → O_t (reward depends on state too)
        edges.push({ from: `s${t}`, to: `O${t}` });
      });

      // s_t → s_{t+1} transitions
      for (let t = 0; t < T - 1; t++) {
        edges.push({ from: `s${t}`, to: `s${t + 1}`, label: "dynamics" });
        // a_t also contributes to s_{t+1}
        edges.push({ from: `a${t}`, to: `s${t + 1}` });
      }

      // Draw edges
      const nodeMap = new Map(nodes.map(n => [n.id, n]));
      edges.forEach(e => {
        const from = nodeMap.get(e.from)!;
        const to = nodeMap.get(e.to)!;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const r = 22;
        const x1 = from.x + (dx / len) * r;
        const y1 = from.y + (dy / len) * r;
        const x2 = to.x - (dx / len) * r;
        const y2 = to.y - (dy / len) * r;

        svg.append("line")
          .attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
          .attr("stroke", "#cbd5e1").attr("stroke-width", 1.2)
          .attr("marker-end", "url(#ig-arrow)");

        if (e.label) {
          svg.append("text").attr("x", (x1 + x2) / 2 + 4).attr("y", (y1 + y2) / 2 - 4)
            .attr("class", "annot").style("font-size", "8px").attr("fill", "#94a3b8").text(e.label);
        }
      });

      // Backward messages (β): dashed violet arrows going right-to-left at top
      for (let t = T - 1; t > 0; t--) {
        const from = nodeMap.get(`s${t}`)!;
        const to = nodeMap.get(`s${t - 1}`)!;
        svg.append("path")
          .attr("d", `M ${from.x - 24} ${from.y - 18} C ${from.x - 50} ${from.y - 36} ${to.x + 50} ${to.y - 36} ${to.x + 24} ${to.y - 18}`)
          .attr("fill", "none").attr("stroke", "var(--maxent-soft)").attr("stroke-width", 1.2)
          .attr("stroke-dasharray", "4 3").attr("marker-end", "url(#ig-arrow-back)").attr("opacity", 0.7);
      }
      svg.append("text").attr("x", 190).attr("y", 15).style("font-size", "9px")
        .attr("fill", "var(--maxent-soft)").attr("text-anchor", "middle")
        .text("← β (backward msg = exp(V*(s)/α))");

      // Draw nodes
      nodes.forEach(n => {
        const g = svg.append("g").attr("transform", `translate(${n.x},${n.y})`).style("cursor", "help");

        g.append("circle").attr("r", 22)
          .attr("fill", n.type === "observed" ? "#e0f2fe" : "white")
          .attr("stroke", n.type === "observed" ? "var(--maxent-likelihood)" : "#94a3b8")
          .attr("stroke-width", n.type === "observed" ? 2.5 : 1.5);

        if (n.type === "observed") {
          // Shaded ring
          g.append("circle").attr("r", 17).attr("fill", "#bae6fd").attr("opacity", 0.5);
        }

        g.append("text").attr("text-anchor", "middle").attr("dy", "0.35em")
          .style("font-family", "var(--rl-font-mono)").style("font-size", "12px")
          .attr("fill", n.type === "observed" ? "var(--maxent-likelihood)" : "var(--rl-ink)")
          .text(n.label);

        g.on("mousemove", (ev: MouseEvent) => {
          const rect = svgEl.getBoundingClientRect();
          tip.innerHTML = `<strong>${n.label}</strong><br>${n.tooltip}`;
          tip.style.opacity = "1";
          tip.style.left = `${ev.clientX - rect.left + 12}px`;
          tip.style.top = `${ev.clientY - rect.top + 12}px`;
        }).on("mouseleave", () => { tip.style.opacity = "0"; });
      });

      // Legend
      const leg = svg.append("g").attr("transform", `translate(440,60)`);
      [[22, "white", "#94a3b8", "Latent (a_t, s_t)"],
       [22, "#e0f2fe", "var(--maxent-likelihood)", "Observed (O_t=1)"]].forEach(([_r, fill, stroke, label], i) => {
        leg.append("circle").attr("cx", 12).attr("cy", i * 28 + 8).attr("r", 10)
          .attr("fill", fill as string).attr("stroke", stroke as string).attr("stroke-width", 1.5);
        leg.append("text").attr("x", 26).attr("y", i * 28 + 13).style("font-size", "9px")
          .attr("fill", "var(--rl-ink-muted)").text(label as string);
      });
      leg.append("line").attr("x1", 2).attr("x2", 22).attr("y1", 70).attr("y2", 70)
        .attr("stroke", "var(--maxent-soft)").attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
      leg.append("text").attr("x", 26).attr("y", 74).style("font-size", "9px").attr("fill", "var(--rl-ink-muted)").text("β message");
    }
  },
);
