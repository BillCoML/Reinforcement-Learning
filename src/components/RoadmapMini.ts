/**
 * V8 — Roadmap Mini. A small node-and-arrow thumbnail of the curriculum DAG,
 * lesson-aware via the `active` attribute. Built lessons (bandits, markov-chains,
 * mdps) render as solid, navigable nodes; unwritten future lessons are grayed,
 * non-navigating placeholders. Hovering any node reveals its one-line connection.
 *
 * On the MDPs page this is where the curriculum first *branches out*: seven
 * forward arrows fan from the MDP to DP, MC, TD, DQN, PG, Max-Ent and Diffusion.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface RNode {
  id: string;
  lesson: string;
  title: string;
  connection: string;
  x: number;
  y: number;
  exists?: boolean;
}

interface Layout {
  W: number;
  H: number;
  source: RNode;
  prereqs: RNode[];
  targets: RNode[];
  caption: string;
}

// Lessons that have actually shipped (solid, navigable).
const BUILT = new Set(["bandits", "markov-chains", "mdps"]);

function banditsLayout(): Layout {
  const H = 288;
  return {
    W: 720,
    H,
    source: { id: "bandits", lesson: "Lesson 1", title: "Multi-Armed Bandits", connection: "You are here.", x: 120, y: H / 2 },
    prereqs: [
      { id: "markov-chains", lesson: "Prereq A", title: "Markov Chains", connection: "The substrate Lesson 2 is built on: a policy turns an MDP into a Markov chain on states.", x: 345, y: 40, exists: true },
    ],
    targets: [
      { id: "mdps", lesson: "Lesson 2", title: "Markov Decision Processes", connection: "A bandit is a one-state MDP — Bellman equations collapse to trivial identities.", x: 560, y: 52, exists: true },
      { id: "dqn", lesson: "Lesson 7", title: "Deep Q-Networks", connection: "ε-greedy is the portable workhorse exploration strategy in deep RL.", x: 560, y: 120 },
      { id: "inference", lesson: "Lesson 11", title: "RL as Inference", connection: "Thompson's 'act as if your beliefs were true' becomes the whole optimal-policy derivation.", x: 560, y: 188 },
      { id: "rlhf", lesson: "Lesson 17", title: "RLHF & Preference Models", connection: "Preference bandits — each pull is a pairwise comparison (Bradley–Terry).", x: 560, y: 240 },
    ],
    caption: "Built lessons are solid; future lessons are gray — hover for the connection",
  };
}

function mdpLayout(): Layout {
  const H = 352;
  const tx = 590;
  const ys = [32, 76, 120, 164, 208, 252, 296];
  return {
    W: 760,
    H,
    source: { id: "mdps", lesson: "Lesson 2", title: "Markov Decision Processes", connection: "You are here. The center of gravity: every lesson ahead solves, approximates, or generalizes the Bellman equations.", x: 200, y: H / 2, exists: true },
    prereqs: [
      { id: "bandits", lesson: "Lesson 1", title: "Multi-Armed Bandits", connection: "A bandit is the one-state MDP; ε-greedy exploration carries forward.", x: 70, y: H / 2 - 52, exists: true },
      { id: "markov-chains", lesson: "Prereq A", title: "Markov Chains", connection: "Fix a policy and the MDP becomes the chain P^π — same matrix, now controlled.", x: 70, y: H / 2 + 52, exists: true },
    ],
    targets: [
      { id: "dynamic-programming", lesson: "Lesson 3", title: "Dynamic Programming", connection: "Policy iteration & value iteration iterate §6's operators when the model is known.", x: tx, y: ys[0] },
      { id: "monte-carlo", lesson: "Lesson 4", title: "Monte Carlo", connection: "Estimate V^π from sampled returns — sidestep the Bellman equation entirely.", x: tx, y: ys[1] },
      { id: "td-learning", lesson: "Lesson 5", title: "TD Learning", connection: "Sampled Bellman backups: replace Σ P(s'|s,a)V(s') with one r + γV(s').", x: tx, y: ys[2] },
      { id: "dqn", lesson: "Lesson 7", title: "Deep Q-Networks", connection: "Learn §7's Q* with a net; the loss is the squared Bellman-optimality residual.", x: tx, y: ys[3] },
      { id: "policy-gradient", lesson: "Lesson 8", title: "Policy Gradient", connection: "The gradient multiplies the score function by §5's advantage A^π.", x: tx, y: ys[4] },
      { id: "max-ent-rl", lesson: "Lesson 10", title: "Max-Entropy RL", connection: "Add an entropy bonus; Bellman gains a log-Z term and π* becomes a softmax over Q.", x: tx, y: ys[5] },
      { id: "diffusion", lesson: "Lesson 16", title: "Diffusion in RL", connection: "Diffusion policies still maximize E_π[Q^π(s,a)] — the same value function.", x: tx, y: ys[6] },
    ],
    caption: "Lesson 2 is the spine — seven forward threads fan out (all unwritten; hover for the connection)",
  };
}

export class RoadmapMini extends HTMLElement {
  connectedCallback(): void {
    this.render();
  }

  private render(): void {
    this.innerHTML = "";
    const active = this.getAttribute("active") ?? "bandits";
    const layout = active === "mdps" ? mdpLayout() : banditsLayout();
    const { W, H } = layout;

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
    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "rm-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M0,0 L10,5 L0,10 z")
      .attr("fill", "var(--rl-ink-faint)");

    const arrow = (x0: number, y0: number, x1: number, y1: number) => {
      const path = d3.path();
      const mx = (x0 + x1) / 2;
      path.moveTo(x0, y0);
      path.bezierCurveTo(mx, y0, mx, y1, x1, y1);
      svg
        .append("path")
        .attr("d", path.toString())
        .attr("fill", "none")
        .attr("stroke", "var(--rl-ink-faint)")
        .attr("stroke-width", 1.4)
        .attr("stroke-dasharray", "4 3")
        .attr("marker-end", "url(#rm-arrow)")
        .attr("opacity", 0.7);
    };

    // source → each target
    for (const t of layout.targets) arrow(layout.source.x + 64, layout.source.y, t.x - 70, t.y);
    // prereqs → source
    for (const pq of layout.prereqs) arrow(pq.x + 64, pq.y, layout.source.x - 64, layout.source.y);

    this.drawNode(svg, layout.source, "active", tip);
    for (const pq of layout.prereqs) this.drawNode(svg, pq, pq.exists ? "real" : "future", tip);
    for (const t of layout.targets) this.drawNode(svg, t, BUILT.has(t.id) ? "real" : "future", tip);

    svg
      .append("text")
      .attr("x", W / 2)
      .attr("y", H - 6)
      .attr("text-anchor", "middle")
      .attr("class", "annot")
      .attr("fill", "var(--rl-ink-faint)")
      .text(layout.caption);
  }

  private drawNode(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    node: RNode,
    kind: "active" | "real" | "future",
    tip: HTMLElement,
  ): void {
    const w = kind === "active" ? 118 : 128;
    const h = 42;
    const navigable = kind === "real";
    const g = svg
      .append("g")
      .attr("transform", `translate(${node.x - w / 2},${node.y - h / 2})`)
      .style("cursor", navigable ? "pointer" : "default");

    g.append("rect")
      .attr("width", w)
      .attr("height", h)
      .attr("rx", 8)
      .attr("fill", kind === "active" ? "var(--rl-ucb-tint)" : kind === "real" ? "var(--rl-surface)" : "var(--rl-surface-2)")
      .attr("stroke", kind === "active" ? "var(--rl-algo-ucb)" : kind === "real" ? "var(--rl-ink-muted)" : "var(--rl-border)")
      .attr("stroke-width", kind === "active" ? 2 : 1)
      .attr("opacity", kind === "future" ? 0.85 : 1);

    g.append("text")
      .attr("x", w / 2)
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .attr("class", "annot")
      .attr("fill", kind === "active" ? "var(--rl-algo-ucb)" : "var(--rl-ink-faint)")
      .style("font-family", "var(--rl-font-ui)")
      .style("font-size", "10px")
      .text(node.lesson);
    g.append("text")
      .attr("x", w / 2)
      .attr("y", 31)
      .attr("text-anchor", "middle")
      .attr("fill", kind === "future" ? "var(--rl-ink-muted)" : "var(--rl-ink)")
      .style("font-family", "var(--rl-font-ui)")
      .style("font-size", "11px")
      .style("font-weight", "600")
      .text(this.truncate(node.title, w));

    if (navigable) {
      g.on("click", () => {
        location.hash = `#${node.id}`;
      });
    }

    g.on("mousemove", (ev: MouseEvent) => {
      const rect = (svg.node() as SVGSVGElement).getBoundingClientRect();
      const nav = navigable ? '<br><span style="color:var(--rl-algo-ucb)">→ click to open</span>' : "";
      tip.innerHTML = `<strong>${node.lesson} — ${node.title}</strong><br>${node.connection}${nav}`;
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
