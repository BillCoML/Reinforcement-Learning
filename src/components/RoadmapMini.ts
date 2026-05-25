/**
 * V8 — Roadmap Mini. A placeholder node-and-arrow diagram: Lesson 1 (this one,
 * highlighted) with four arrows to grayed-out future lessons. Hovering a target
 * reveals its title and one-sentence connection. The full curriculum DAG will
 * replace this once the catalog file is written; targets don't exist yet, so
 * nodes are non-navigating.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface Node {
  id: string;
  lesson: string;
  title: string;
  connection: string;
  x: number;
  y: number;
}

const W = 720;
const H = 288;

const SOURCE: Node = {
  id: "bandits",
  lesson: "Lesson 1",
  title: "Multi-Armed Bandits",
  connection: "You are here.",
  x: 120,
  y: H / 2,
};

const TARGETS: Node[] = [
  {
    id: "mdps", lesson: "Lesson 2", title: "Markov Decision Processes",
    connection: "A bandit is a one-state MDP — Bellman equations collapse to trivial identities.",
    x: 560, y: 52,
  },
  {
    id: "dqn", lesson: "Lesson 7", title: "Deep Q-Networks",
    connection: "ε-greedy is the portable workhorse exploration strategy in deep RL.",
    x: 560, y: 120,
  },
  {
    id: "inference", lesson: "Lesson 11", title: "RL as Inference",
    connection: "Thompson's 'act as if your beliefs were true' becomes the whole optimal-policy derivation.",
    x: 560, y: 188,
  },
  {
    id: "rlhf", lesson: "Lesson 17", title: "RLHF & Preference Models",
    connection: "Preference bandits — each pull is a pairwise comparison (Bradley–Terry).",
    x: 560, y: 240,
  },
];

export class RoadmapMini extends HTMLElement {
  connectedCallback(): void {
    this.render();
  }

  private render(): void {
    this.innerHTML = "";
    const { panel, body } = createPanel({ id: "roadmap-mini" });

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const ns = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(ns, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);

    const tip = document.createElement("div");
    tip.className = "rl-tooltip";
    tip.style.maxWidth = "240px";
    tip.style.whiteSpace = "normal";
    wrap.appendChild(tip);

    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl as SVGSVGElement);

    // arrowhead marker
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "rm-arrow").attr("viewBox", "0 0 10 10")
      .attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7)
      .attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "var(--rl-ink-faint)");

    // arrows
    for (const t of TARGETS) {
      const path = d3.path();
      const x0 = SOURCE.x + 56;
      const y0 = SOURCE.y;
      const x1 = t.x - 64;
      const y1 = t.y;
      const mx = (x0 + x1) / 2;
      path.moveTo(x0, y0);
      path.bezierCurveTo(mx, y0, mx, y1, x1, y1);
      svg.append("path").attr("d", path.toString()).attr("fill", "none")
        .attr("stroke", "var(--rl-ink-faint)").attr("stroke-width", 1.4)
        .attr("stroke-dasharray", "4 3").attr("marker-end", "url(#rm-arrow)").attr("opacity", 0.7);
    }

    // source node (highlighted)
    this.drawNode(svg, SOURCE, true, tip);
    // target nodes (grayed, future)
    for (const t of TARGETS) this.drawNode(svg, t, false, tip);

    // caption
    svg.append("text").attr("x", W / 2).attr("y", H - 6).attr("text-anchor", "middle")
      .attr("class", "annot").attr("fill", "var(--rl-ink-faint)")
      .text("future lessons — hover for the connection (not yet written)");
  }

  private drawNode(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    node: Node,
    highlighted: boolean,
    tip: HTMLElement,
  ): void {
    const w = highlighted ? 112 : 128;
    const h = 44;
    const g = svg.append("g")
      .attr("transform", `translate(${node.x - w / 2},${node.y - h / 2})`)
      .style("cursor", "default");

    g.append("rect").attr("width", w).attr("height", h).attr("rx", 8)
      .attr("fill", highlighted ? "var(--rl-ucb-tint)" : "var(--rl-surface-2)")
      .attr("stroke", highlighted ? "var(--rl-algo-ucb)" : "var(--rl-border)")
      .attr("stroke-width", highlighted ? 2 : 1)
      .attr("opacity", highlighted ? 1 : 0.85);

    g.append("text").attr("x", w / 2).attr("y", 17).attr("text-anchor", "middle")
      .attr("class", "annot")
      .attr("fill", highlighted ? "var(--rl-algo-ucb)" : "var(--rl-ink-faint)")
      .style("font-family", "var(--rl-font-ui)").style("font-size", "10px")
      .text(node.lesson);
    g.append("text").attr("x", w / 2).attr("y", 32).attr("text-anchor", "middle")
      .attr("fill", highlighted ? "var(--rl-ink)" : "var(--rl-ink-muted)")
      .style("font-family", "var(--rl-font-ui)").style("font-size", "11px").style("font-weight", "600")
      .text(this.truncate(node.title, w));

    // hover tooltip
    g.on("mousemove", (ev: MouseEvent) => {
      const rect = (svg.node() as SVGSVGElement).getBoundingClientRect();
      tip.innerHTML = `<strong>${node.lesson} — ${node.title}</strong><br>${node.connection}`;
      tip.style.opacity = "1";
      tip.style.left = `${ev.clientX - rect.left + 12}px`;
      tip.style.top = `${ev.clientY - rect.top + 12}px`;
    }).on("mouseleave", () => {
      tip.style.opacity = "0";
    });
  }

  private truncate(s: string, w: number): string {
    const max = Math.floor(w / 6.5);
    return s.length > max ? s.slice(0, max - 1) + "…" : s;
  }
}

customElements.define("roadmap-mini", RoadmapMini);
