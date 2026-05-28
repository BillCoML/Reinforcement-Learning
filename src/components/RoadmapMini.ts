/**
 * Roadmap Mini. A small node-and-arrow thumbnail of the curriculum DAG,
 * lesson-aware via the `active` attribute. Every node is a built, navigable
 * lesson — unwritten future lessons are not rendered.
 *
 * Active lesson slugs (must match LESSONS in main.ts):
 *   bandits, markov-chains, mdps, contractions, dynamic-programming,
 *   importance-sampling, monte-carlo, td-learning, function-approximation,
 *   policy-gradient, trpo-ppo, max-ent-rl.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface RNode {
  /** URL slug; must match a built lesson. */
  id: string;
  /** Eyebrow label (e.g., "Lesson 1", "Prereq A"). */
  lesson: string;
  title: string;
  connection: string;
  x: number;
  y: number;
}

interface Layout {
  W: number;
  H: number;
  source: RNode;
  prereqs: RNode[];
  targets: RNode[];
  caption: string;
}

function banditsLayout(): Layout {
  const H = 260;
  return {
    W: 720,
    H,
    source: { id: "bandits", lesson: "Lesson 1", title: "Multi-Armed Bandits", connection: "You are here.", x: 200, y: H / 2 },
    prereqs: [
      { id: "markov-chains", lesson: "Prereq A", title: "Markov Chains", connection: "Shared math prereq — the substrate Lesson 2 (MDPs) is built on.", x: 60, y: H / 2 - 48 },
      { id: "contractions",  lesson: "Prereq C", title: "Contractions",  connection: "Shared math prereq — the theorem that makes value iteration converge in Lesson 3.", x: 60, y: H / 2 + 48 },
    ],
    targets: [
      { id: "mdps",                   lesson: "Lesson 2", title: "MDPs",            connection: "A bandit is a one-state MDP — Bellman equations collapse to trivial identities.", x: 560, y: H / 2 - 48 },
      { id: "function-approximation", lesson: "Lesson 9", title: "Function Approx", connection: "ε-greedy is the portable workhorse exploration strategy in deep Q-learning.", x: 560, y: H / 2 + 48 },
    ],
    caption: "Lesson 1 enters the curriculum at L2 (MDPs); ε-greedy carries forward to L9 (DQN).",
  };
}

function markovLayout(): Layout {
  const H = 220;
  return {
    W: 700,
    H,
    source: { id: "markov-chains", lesson: "Prereq A", title: "Markov Chains", connection: "You are here.", x: 160, y: H / 2 },
    prereqs: [],
    targets: [
      { id: "mdps",         lesson: "Lesson 2", title: "MDPs",         connection: "Fix a policy and the MDP collapses to a Markov chain on states — the same P^π matrix.", x: 500, y: H / 2 - 44 },
      { id: "contractions", lesson: "Prereq C", title: "Contractions", connection: "The contraction proofs use row-stochastic P^π — its properties come from Markov chain theory.", x: 500, y: H / 2 + 44 },
    ],
    caption: "Prereq A feeds Lesson 2 (MDPs) and the contraction proofs in Prereq C.",
  };
}

function mdpLayout(): Layout {
  const H = 360;
  const tx = 590;
  const ys = [40, 96, 152, 208, 264, 320];
  return {
    W: 760,
    H,
    source: { id: "mdps", lesson: "Lesson 2", title: "MDPs", connection: "You are here. The spine of the curriculum: every lesson ahead solves, approximates, or generalizes the Bellman equations.", x: 200, y: H / 2 },
    prereqs: [
      { id: "bandits",       lesson: "Lesson 1", title: "Multi-Armed Bandits", connection: "A bandit is the one-state MDP; ε-greedy exploration carries forward.", x: 60, y: H / 2 - 56 },
      { id: "markov-chains", lesson: "Prereq A", title: "Markov Chains",       connection: "Fix a policy and the MDP becomes the chain P^π — same matrix, now controlled.", x: 60, y: H / 2 },
      { id: "contractions",  lesson: "Prereq C", title: "Contractions",        connection: "The Banach theorem proves value iteration converges — a prereq between L2 and L3.", x: 60, y: H / 2 + 56 },
    ],
    targets: [
      { id: "dynamic-programming",    lesson: "Lesson 3",  title: "Dynamic Programming", connection: "Policy iteration & value iteration iterate the Bellman operators when the model is known.", x: tx, y: ys[0] },
      { id: "importance-sampling",    lesson: "Lesson 6",  title: "Importance Sampling", connection: "IS reweights off-policy trajectories — the bridge from model-known to model-free.", x: tx, y: ys[1] },
      { id: "monte-carlo",            lesson: "Lesson 7",  title: "Monte Carlo",         connection: "Estimate V^π from sampled returns — sidestep the Bellman equation entirely.", x: tx, y: ys[2] },
      { id: "td-learning",            lesson: "Lesson 8",  title: "TD Learning",         connection: "Sampled Bellman backups: replace Σ P(s'|s,a)V(s') with one r + γV(s').", x: tx, y: ys[3] },
      { id: "function-approximation", lesson: "Lesson 9",  title: "Function Approx",     connection: "Learn Q* with a neural net; the loss is the squared Bellman-optimality residual.", x: tx, y: ys[4] },
      { id: "policy-gradient",        lesson: "Lesson 10", title: "Policy Gradient",     connection: "The gradient multiplies the score function by the advantage A^π.", x: tx, y: ys[5] },
    ],
    caption: "Lesson 2 is the spine — six threads fan forward to L3, L6, L7, L8, L9 and L10.",
  };
}

function contractionsLayout(): Layout {
  const H = 260;
  const tx = 560;
  const ys = [40, 100, 160, 220];
  return {
    W: 720,
    H,
    source: { id: "contractions", lesson: "Prereq C", title: "Contractions", connection: "You are here. The single theorem that powers all of dynamic programming.", x: 180, y: H / 2 },
    prereqs: [
      { id: "mdps", lesson: "Lesson 2", title: "MDPs", connection: "The Bellman operators T^π and T^* are introduced here — this prereq proves they're contractions.", x: 55, y: H / 2 },
    ],
    targets: [
      { id: "dynamic-programming",    lesson: "Lesson 3",  title: "Dynamic Programming", connection: "Value iteration is Banach iteration on T^*. The stopping criterion is the a-posteriori bound with c = γ.", x: tx, y: ys[0] },
      { id: "td-learning",            lesson: "Lesson 8",  title: "TD Learning",         connection: "Stochastic Bellman backups are contractions in expectation — the Robbins–Monro theorem takes over.", x: tx, y: ys[1] },
      { id: "function-approximation", lesson: "Lesson 9",  title: "Function Approx",     connection: "The deadly triad: projection + bootstrapping + off-policy breaks the contraction property.", x: tx, y: ys[2] },
      { id: "trpo-ppo",               lesson: "Lesson 11", title: "TRPO / PPO",          connection: "Fixed-point intuition: iterate until convergence, bound per-step movement (trust regions, clipping).", x: tx, y: ys[3] },
    ],
    caption: "Prereq C links L2 (MDPs) to L3 (DP) and cashes in across L8, L9 and L11.",
  };
}

function dpLayout(): Layout {
  const H = 320;
  const tx = 590;
  const ys = [40, 96, 152, 208, 264];
  return {
    W: 760,
    H,
    source: { id: "dynamic-programming", lesson: "Lesson 3", title: "Dynamic Programming", connection: "You are here. PI and VI exactly solve any MDP when the model is known.", x: 200, y: H / 2 },
    prereqs: [
      { id: "mdps",         lesson: "Lesson 2", title: "MDPs",         connection: "DP iterates the Bellman operators defined here.", x: 60, y: H / 2 - 52 },
      { id: "contractions", lesson: "Prereq C", title: "Contractions", connection: "Proves T^π and T^* are γ-contractions — the stopping criterion is the a-posteriori bound.", x: 60, y: H / 2 + 52 },
    ],
    targets: [
      { id: "importance-sampling",    lesson: "Lesson 6",  title: "Importance Sampling", connection: "IS evaluates π_t from π_b trajectories — DP is the model-known baseline IS replaces.", x: tx, y: ys[0] },
      { id: "monte-carlo",            lesson: "Lesson 7",  title: "Monte Carlo",         connection: "MC uses sample-based policy evaluation instead of the Bellman solve.", x: tx, y: ys[1] },
      { id: "td-learning",            lesson: "Lesson 8",  title: "TD Learning",         connection: "TD(0) is a stochastic Bellman backup — sampled DP.", x: tx, y: ys[2] },
      { id: "function-approximation", lesson: "Lesson 9",  title: "Function Approx",     connection: "DQN learns V_θ / Q_θ to approximate VI with a neural net.", x: tx, y: ys[3] },
      { id: "policy-gradient",        lesson: "Lesson 10", title: "Policy Gradient",     connection: "Actor-critic is GPI: critic = policy evaluation, actor = approximate policy improvement.", x: tx, y: ys[4] },
    ],
    caption: "Lesson 3 closes the model-known arc; five threads fan into model-free and deep RL.",
  };
}

function isLayout(): Layout {
  const H = 240;
  const tx = 560;
  const ys = [50, 120, 190];
  return {
    W: 720,
    H,
    source: { id: "importance-sampling", lesson: "Lesson 6", title: "Importance Sampling", connection: "You are here. The IS identity and its variance consequences.", x: 200, y: H / 2 },
    prereqs: [
      { id: "mdps",                lesson: "Lesson 2", title: "MDPs",                connection: "MDPs define the trajectory distribution: IS converts off-policy samples to on-policy estimates.", x: 55, y: H / 2 - 40 },
      { id: "dynamic-programming", lesson: "Lesson 3", title: "Dynamic Programming", connection: "DP needs a model; IS lets you evaluate π_t using π_b trajectories without the transition matrix.", x: 55, y: H / 2 + 40 },
    ],
    targets: [
      { id: "monte-carlo", lesson: "Lesson 7",  title: "Monte Carlo", connection: "Off-policy MC is trajectory IS on episode returns. On-policy MC needs no IS at all.", x: tx, y: ys[0] },
      { id: "td-learning", lesson: "Lesson 8",  title: "TD Learning", connection: "Per-step ratio π(a|s)/µ(a|s) replaces the full product in off-policy TD; Q-learning sidesteps it entirely.", x: tx, y: ys[1] },
      { id: "trpo-ppo",    lesson: "Lesson 11", title: "TRPO / PPO",  connection: "PPO's clipped ratio and TRPO's KL constraint are IS variance bounds in disguise.", x: tx, y: ys[2] },
    ],
    caption: "Lesson 6 feeds L7, L8 and L11 — hover any node for the IS connection.",
  };
}

function mcLayout(): Layout {
  const H = 240;
  const tx = 560;
  const ys = [50, 120, 190];
  return {
    W: 720,
    H,
    source: { id: "monte-carlo", lesson: "Lesson 7", title: "Monte Carlo", connection: "You are here. Model-free policy evaluation and control using sampled returns.", x: 200, y: H / 2 },
    prereqs: [
      { id: "mdps",                lesson: "Lesson 2", title: "MDPs",                connection: "The MDP framework defines V^π and Q^π — MC estimates them from samples.", x: 60, y: H / 2 - 42 },
      { id: "importance-sampling", lesson: "Lesson 6", title: "Importance Sampling", connection: "Off-policy MC reweights trajectories by ρ = π_t/π_b — direct application of IS.", x: 60, y: H / 2 + 42 },
    ],
    targets: [
      { id: "td-learning",            lesson: "Lesson 8",  title: "TD Learning",     connection: "TD bootstraps with V̂(s') instead of the full return; bias ↑ variance ↓.", x: tx, y: ys[0] },
      { id: "function-approximation", lesson: "Lesson 9",  title: "Function Approx", connection: "DQN is off-policy MC control with a neural Q-function and replay.", x: tx, y: ys[1] },
      { id: "policy-gradient",        lesson: "Lesson 10", title: "Policy Gradient", connection: "REINFORCE is on-policy MC gradient ascent; the return G_t is its credit-assignment signal.", x: tx, y: ys[2] },
    ],
    caption: "Lesson 7 closes the model-free evaluation arc and opens model-free control.",
  };
}

function tdLayout(): Layout {
  const H = 260;
  const tx = 560;
  const ys = [60, 130, 200];
  return {
    W: 720,
    H,
    source: { id: "td-learning", lesson: "Lesson 8", title: "TD Learning", connection: "You are here. Bootstrapped Bellman backups: TD(0), SARSA, Q-learning, n-step, TD(λ).", x: 200, y: H / 2 },
    prereqs: [
      { id: "mdps",         lesson: "Lesson 2", title: "MDPs",         connection: "The Bellman equations define what TD targets are converging toward.", x: 60, y: H / 2 - 58 },
      { id: "contractions", lesson: "Prereq C", title: "Contractions", connection: "The Bellman operator is a γ-contraction — the convergence proof uses this directly.", x: 60, y: H / 2 },
      { id: "monte-carlo",  lesson: "Lesson 7", title: "Monte Carlo",  connection: "MC is the λ=1 limit of TD(λ); TD interpolates between one-step bootstrap and full MC return.", x: 60, y: H / 2 + 58 },
    ],
    targets: [
      { id: "function-approximation", lesson: "Lesson 9",  title: "Function Approx", connection: "DQN is Q-learning with a neural Q-function, target network, and replay buffer.", x: tx, y: ys[0] },
      { id: "policy-gradient",        lesson: "Lesson 10", title: "Policy Gradient", connection: "TD value estimates form the critic in actor-critic; GAE uses TD(λ) for advantage.", x: tx, y: ys[1] },
      { id: "trpo-ppo",               lesson: "Lesson 11", title: "TRPO / PPO",      connection: "PPO's GAE-λ parameter is the same λ dial developed in TD(λ).", x: tx, y: ys[2] },
    ],
    caption: "Lesson 8 is model-free learning's pivot — three threads forward to L9, L10 and L11.",
  };
}

function faLayout(): Layout {
  const H = 260;
  const tx = 560;
  const ys = [60, 130, 200];
  return {
    W: 720,
    H,
    source: { id: "function-approximation", lesson: "Lesson 9", title: "Function Approx", connection: "You are here. Semi-gradient TD, the deadly triad, and Deep Q-Networks.", x: 200, y: H / 2 },
    prereqs: [
      { id: "td-learning", lesson: "Lesson 8", title: "TD Learning", connection: "DQN is Q-learning with a neural net; target network + replay fix the instabilities of tabular TD.", x: 60, y: H / 2 - 58 },
      { id: "mdps",        lesson: "Lesson 2", title: "MDPs",        connection: "The Bellman optimality equation is what the Q-network is trained to satisfy.", x: 60, y: H / 2 },
      { id: "monte-carlo", lesson: "Lesson 7", title: "Monte Carlo", connection: "Experience replay turns DQN's online transitions into an iid sample — the MC intuition.", x: 60, y: H / 2 + 58 },
    ],
    targets: [
      { id: "policy-gradient", lesson: "Lesson 10", title: "Policy Gradient", connection: "Actor-critic replaces tabular V with a learned V_θ — the function-approximation leap.", x: tx, y: ys[0] },
      { id: "trpo-ppo",        lesson: "Lesson 11", title: "TRPO / PPO",      connection: "PPO's value network is FA + DQN; trust regions correct for deadly-triad instability.", x: tx, y: ys[1] },
      { id: "max-ent-rl",      lesson: "Lesson 12", title: "Max-Entropy RL",  connection: "SAC extends DQN with a soft Bellman operator; double Q-nets tame maximization bias.", x: tx, y: ys[2] },
    ],
    caption: "Lesson 9 bridges tabular model-free RL and deep RL — three threads forward.",
  };
}

function pgLayout(): Layout {
  const H = 240;
  const tx = 560;
  const ys = [80, 160];
  return {
    W: 720,
    H,
    source: { id: "policy-gradient", lesson: "Lesson 10", title: "Policy Gradient", connection: "You are here. REINFORCE, actor-critic, and the bias–variance tradeoff in advantage estimation.", x: 200, y: H / 2 },
    prereqs: [
      { id: "monte-carlo",            lesson: "Lesson 7", title: "Monte Carlo",     connection: "REINFORCE is on-policy MC gradient ascent; G_t is the credit-assignment signal.", x: 60, y: H / 2 - 56 },
      { id: "td-learning",            lesson: "Lesson 8", title: "TD Learning",     connection: "TD value estimates form the critic; GAE(λ) uses TD(λ) for advantage estimation.", x: 60, y: H / 2 },
      { id: "function-approximation", lesson: "Lesson 9", title: "Function Approx", connection: "Actor-critic replaces tabular V with a learned V_θ — the function-approximation leap.", x: 60, y: H / 2 + 56 },
    ],
    targets: [
      { id: "trpo-ppo",   lesson: "Lesson 11", title: "TRPO / PPO",     connection: "Trust regions and clipped objectives prevent catastrophically large policy gradient steps.", x: tx, y: ys[0] },
      { id: "max-ent-rl", lesson: "Lesson 12", title: "Max-Entropy RL", connection: "Adds an entropy bonus H[π] to the PG objective — prevents collapse and improves exploration.", x: tx, y: ys[1] },
    ],
    caption: "Lesson 10 leads directly into the trust-region (L11) and max-entropy (L12) families.",
  };
}

function ppoLayout(): Layout {
  const H = 240;
  const tx = 560;
  return {
    W: 720,
    H,
    source: { id: "trpo-ppo", lesson: "Lesson 11", title: "TRPO / PPO", connection: "You are here. Trust regions, the clipped surrogate, GAE, and the PPO algorithm.", x: 200, y: H / 2 },
    prereqs: [
      { id: "policy-gradient",        lesson: "Lesson 10", title: "Policy Gradient",     connection: "PPO is actor-critic with a clipped IS surrogate and multiple epochs per batch.", x: 60, y: H / 2 - 56 },
      { id: "importance-sampling",    lesson: "Lesson 6",  title: "Importance Sampling", connection: "PPO's probability ratio r_t = π/π_old is the IS weight — same object, same variance problem.", x: 60, y: H / 2 },
      { id: "function-approximation", lesson: "Lesson 9",  title: "Function Approx",     connection: "The deadly triad motivates PPO's bounded updates — clipping is the on-policy antidote.", x: 60, y: H / 2 + 56 },
    ],
    targets: [
      { id: "max-ent-rl", lesson: "Lesson 12", title: "Max-Entropy RL", connection: "The entropy bonus c₂·H[π] inside PPO becomes the primary objective in max-entropy RL.", x: tx, y: H / 2 },
    ],
    caption: "Lesson 11 ties IS, FA and PG together and feeds forward into Lesson 12.",
  };
}

function maxEntLayout(): Layout {
  const H = 240;
  return {
    W: 560,
    H,
    source: { id: "max-ent-rl", lesson: "Lesson 12", title: "Max-Entropy RL", connection: "You are here. Soft Bellman, Boltzmann policies, and the goal-avoidance failure mode.", x: 380, y: H / 2 },
    prereqs: [
      { id: "dynamic-programming", lesson: "Lesson 3",  title: "Dynamic Programming", connection: "Soft VI/PI are direct analogs of hard VI/PI with logsumexp replacing max.", x: 80, y: H / 2 - 56 },
      { id: "policy-gradient",     lesson: "Lesson 10", title: "Policy Gradient",     connection: "The softmax cap (V ≈ 0.722 < V*) is reframed as the correct soft-optimal policy at α ≈ 0.02.", x: 80, y: H / 2 },
      { id: "trpo-ppo",            lesson: "Lesson 11", title: "TRPO / PPO",          connection: "The entropy bonus c₂·H[π] inside PPO becomes the primary objective here.", x: 80, y: H / 2 + 56 },
    ],
    targets: [],
    caption: "Lesson 12 — current endpoint of the built curriculum.",
  };
}

const LAYOUTS: Record<string, () => Layout> = {
  bandits: banditsLayout,
  "markov-chains": markovLayout,
  mdps: mdpLayout,
  contractions: contractionsLayout,
  "dynamic-programming": dpLayout,
  "importance-sampling": isLayout,
  "monte-carlo": mcLayout,
  "td-learning": tdLayout,
  "function-approximation": faLayout,
  "policy-gradient": pgLayout,
  "trpo-ppo": ppoLayout,
  "max-ent-rl": maxEntLayout,
};

export class RoadmapMini extends HTMLElement {
  connectedCallback(): void {
    this.render();
  }

  private render(): void {
    this.innerHTML = "";
    const active = this.getAttribute("active") ?? "bandits";
    const layout = (LAYOUTS[active] ?? banditsLayout)();
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

    for (const t of layout.targets) arrow(layout.source.x + 64, layout.source.y, t.x - 70, t.y);
    for (const pq of layout.prereqs) arrow(pq.x + 64, pq.y, layout.source.x - 64, layout.source.y);

    this.drawNode(svg, layout.source, true, tip);
    for (const pq of layout.prereqs) this.drawNode(svg, pq, false, tip);
    for (const t of layout.targets) this.drawNode(svg, t, false, tip);

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
    isActive: boolean,
    tip: HTMLElement,
  ): void {
    const w = isActive ? 118 : 128;
    const h = 42;
    const g = svg
      .append("g")
      .attr("transform", `translate(${node.x - w / 2},${node.y - h / 2})`)
      .style("cursor", "pointer");

    g.append("rect")
      .attr("width", w)
      .attr("height", h)
      .attr("rx", 8)
      .attr("fill", isActive ? "var(--rl-ucb-tint)" : "var(--rl-surface)")
      .attr("stroke", isActive ? "var(--rl-algo-ucb)" : "var(--rl-ink-muted)")
      .attr("stroke-width", isActive ? 2 : 1);

    g.append("text")
      .attr("x", w / 2)
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .attr("class", "annot")
      .attr("fill", isActive ? "var(--rl-algo-ucb)" : "var(--rl-ink-faint)")
      .style("font-family", "var(--rl-font-ui)")
      .style("font-size", "10px")
      .text(node.lesson);
    g.append("text")
      .attr("x", w / 2)
      .attr("y", 31)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--rl-ink)")
      .style("font-family", "var(--rl-font-ui)")
      .style("font-size", "11px")
      .style("font-weight", "600")
      .text(this.truncate(node.title, w));

    g.on("click", () => {
      location.hash = `#${node.id}`;
    });

    g.on("mousemove", (ev: MouseEvent) => {
      const rect = (svg.node() as SVGSVGElement).getBoundingClientRect();
      const nav = isActive ? "" : '<br><span style="color:var(--rl-algo-ucb)">→ click to open</span>';
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
