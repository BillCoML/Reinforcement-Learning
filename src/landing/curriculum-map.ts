/**
 * Curriculum map — full DAG of the 12-lesson curriculum, with two math prereqs
 * floating above. Nodes are clickable (navigate via #slug), animated particles
 * stream along the main chain to convey the learning journey, and hovering any
 * node reveals its one-line description in a side panel.
 */

interface MapNode {
  id: string;
  slug: string;
  eyebrow: string;
  title: string;
  blurb: string;
  x: number;
  y: number;
  /** Forward edges (downstream lessons). */
  next: string[];
}

const W = 1400;
const H = 340;

// Main chain at ROW_Y with a gentle ±10 wave; prereqs floating above.
const ROW_Y = 220;
const PREREQ_Y = 80;

// 10 main-chain lessons, evenly spaced.
const X = (i: number) => 90 + i * 135; // i = 0..9

const NODES: MapNode[] = [
  { id: "bandits",                slug: "bandits",                eyebrow: "Lesson 1",  title: "Multi-Armed Bandits",   blurb: "ε-greedy, UCB, Thompson — the exploration/exploitation dilemma in its purest form.",                                 x: X(0), y: ROW_Y - 10, next: ["mdps"] },
  { id: "markov-chains",          slug: "markov-chains",          eyebrow: "Prereq A",  title: "Markov Chains",         blurb: "The substrate every RL algorithm sits on: P^π is a Markov chain on states.",                                         x: X(1), y: PREREQ_Y,   next: ["mdps", "contractions"] },
  { id: "mdps",                   slug: "mdps",                   eyebrow: "Lesson 2",  title: "MDPs",                  blurb: "The Bellman equations. The spine of the curriculum — every later lesson solves, samples, or generalizes them.",     x: X(1), y: ROW_Y + 8,  next: ["dynamic-programming", "monte-carlo", "td-learning", "function-approximation", "policy-gradient"] },
  { id: "contractions",           slug: "contractions",           eyebrow: "Prereq C",  title: "Contractions",          blurb: "Banach's fixed-point theorem. The reason value iteration converges; the foundation of Lesson 3.",                   x: X(2), y: PREREQ_Y,   next: ["dynamic-programming", "td-learning", "function-approximation", "trpo-ppo"] },
  { id: "dynamic-programming",    slug: "dynamic-programming",    eyebrow: "Lesson 3",  title: "Dynamic Programming",   blurb: "Policy iteration & value iteration: exact MDP solvers when the model is known.",                                    x: X(2), y: ROW_Y - 6,  next: ["importance-sampling", "monte-carlo", "td-learning"] },
  { id: "importance-sampling",    slug: "importance-sampling",    eyebrow: "Lesson 6",  title: "Importance Sampling",   blurb: "Reweight off-policy trajectories. The variance bound at the heart of PPO, offline RL, RLHF.",                       x: X(3), y: ROW_Y + 8,  next: ["monte-carlo", "td-learning", "trpo-ppo"] },
  { id: "monte-carlo",            slug: "monte-carlo",            eyebrow: "Lesson 7",  title: "Monte Carlo",           blurb: "Estimate V^π from full returns. The first model-free method; the λ=1 limit of TD(λ).",                             x: X(4), y: ROW_Y - 6,  next: ["td-learning", "function-approximation", "policy-gradient"] },
  { id: "td-learning",            slug: "td-learning",            eyebrow: "Lesson 8",  title: "TD Learning",           blurb: "TD(0), SARSA, Q-learning, n-step, TD(λ). Bootstrapped Bellman backups — RL's pivot.",                               x: X(5), y: ROW_Y + 8,  next: ["function-approximation", "policy-gradient", "trpo-ppo"] },
  { id: "function-approximation", slug: "function-approximation", eyebrow: "Lesson 9",  title: "Function Approx · DQN", blurb: "Semi-gradient TD, the deadly triad, and Deep Q-Networks — RL goes neural.",                                         x: X(6), y: ROW_Y - 6,  next: ["policy-gradient", "trpo-ppo", "max-ent-rl"] },
  { id: "policy-gradient",        slug: "policy-gradient",        eyebrow: "Lesson 10", title: "Policy Gradient",       blurb: "REINFORCE, actor-critic, GAE. Learn the policy directly; the bias-variance tradeoff in advantage estimation.",     x: X(7), y: ROW_Y + 8,  next: ["trpo-ppo", "max-ent-rl"] },
  { id: "trpo-ppo",               slug: "trpo-ppo",               eyebrow: "Lesson 11", title: "TRPO / PPO",            blurb: "Trust regions, clipped surrogates, GAE — the workhorse algorithms behind modern RL and RLHF.",                     x: X(8), y: ROW_Y - 6,  next: ["max-ent-rl"] },
  { id: "max-ent-rl",             slug: "max-ent-rl",             eyebrow: "Lesson 12", title: "Max-Entropy RL",        blurb: "Soft Bellman, Boltzmann policies. The bridge to SAC, RL-as-inference, and KL-controlled fine-tuning.",             x: X(9), y: ROW_Y + 8,  next: [] },
];

function nodeById(id: string): MapNode {
  return NODES.find((n) => n.id === id)!;
}

export function createCurriculumMap(): HTMLElement {
  const wrap = document.createElement("section");
  wrap.className = "landing-section landing-map";
  wrap.id = "curriculum";

  const h = document.createElement("h2");
  h.className = "landing-section__title";
  h.textContent = "The curriculum";
  const sub = document.createElement("p");
  sub.className = "landing-section__sub";
  sub.textContent = "Twelve lessons, two math prereqs — wired in a single dependency graph. Hover any node for its one-line role; click to open.";
  wrap.appendChild(h);
  wrap.appendChild(sub);

  const stage = document.createElement("div");
  stage.className = "landing-map__stage";

  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("class", "landing-map__svg");
  stage.appendChild(svg);

  // Tooltip panel below the map
  const tip = document.createElement("div");
  tip.className = "landing-map__tip";
  tip.innerHTML = `
    <div class="landing-map__tip-eyebrow">Hover a node</div>
    <div class="landing-map__tip-title">The curriculum at a glance</div>
    <div class="landing-map__tip-blurb">
      Twelve lessons forming a single learning path from bandits through PPO and max-entropy RL.
      Roll over any node to see its role; click to jump in.
    </div>
  `;
  stage.appendChild(tip);

  // Arrow marker
  const defs = document.createElementNS(svgNs, "defs");
  defs.innerHTML = `
    <marker id="cm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--rl-ink-faint)"/>
    </marker>
    <marker id="cm-arrow-hot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--rl-algo-ucb)"/>
    </marker>
  `;
  svg.appendChild(defs);

  // Edges
  const edgeLayer = document.createElementNS(svgNs, "g");
  edgeLayer.setAttribute("class", "landing-map__edges");
  svg.appendChild(edgeLayer);

  const edgeEls: { from: string; to: string; el: SVGPathElement }[] = [];
  for (const node of NODES) {
    for (const nx of node.next) {
      const target = nodeById(nx);
      const path = document.createElementNS(svgNs, "path");
      const x0 = node.x;
      const y0 = node.y;
      const x1 = target.x;
      const y1 = target.y;
      const cx = (x0 + x1) / 2;
      path.setAttribute("d", `M ${x0} ${y0} C ${cx} ${y0}, ${cx} ${y1}, ${x1} ${y1}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "var(--rl-ink-faint)");
      path.setAttribute("stroke-width", "1.2");
      path.setAttribute("opacity", "0.55");
      path.setAttribute("marker-end", "url(#cm-arrow)");
      edgeLayer.appendChild(path);
      edgeEls.push({ from: node.id, to: nx, el: path });
    }
  }

  // Flowing-particle layer (gives the map a sense of motion)
  const flowLayer = document.createElementNS(svgNs, "g");
  flowLayer.setAttribute("class", "landing-map__flow");
  svg.appendChild(flowLayer);
  // Spine path connecting L1 → L2 → L3 → L6 → ... → L12 along the main row
  const spineIds = ["bandits", "mdps", "dynamic-programming", "importance-sampling", "monte-carlo", "td-learning", "function-approximation", "policy-gradient", "trpo-ppo", "max-ent-rl"];
  const spineNodes = spineIds.map((id) => nodeById(id));
  let spineD = `M ${spineNodes[0].x} ${spineNodes[0].y}`;
  for (let i = 1; i < spineNodes.length; i++) {
    const a = spineNodes[i - 1];
    const b = spineNodes[i];
    const cx = (a.x + b.x) / 2;
    spineD += ` C ${cx} ${a.y}, ${cx} ${b.y}, ${b.x} ${b.y}`;
  }
  const spine = document.createElementNS(svgNs, "path");
  spine.setAttribute("id", "cm-spine");
  spine.setAttribute("d", spineD);
  spine.setAttribute("fill", "none");
  spine.setAttribute("stroke", "transparent");
  flowLayer.appendChild(spine);
  for (let i = 0; i < 6; i++) {
    const dot = document.createElementNS(svgNs, "circle");
    dot.setAttribute("r", "2.6");
    dot.setAttribute("fill", "var(--rl-algo-ucb)");
    dot.setAttribute("opacity", "0.7");
    const anim = document.createElementNS(svgNs, "animateMotion");
    anim.setAttribute("dur", "9s");
    anim.setAttribute("repeatCount", "indefinite");
    anim.setAttribute("begin", `${i * 1.5}s`);
    anim.innerHTML = `<mpath href="#cm-spine"/>`;
    dot.appendChild(anim);
    flowLayer.appendChild(dot);
  }

  // Nodes
  const nodeLayer = document.createElementNS(svgNs, "g");
  nodeLayer.setAttribute("class", "landing-map__nodes");
  svg.appendChild(nodeLayer);

  for (const n of NODES) {
    const isPrereq = n.eyebrow.startsWith("Prereq");
    const w = isPrereq ? 124 : 130;
    const hh = isPrereq ? 44 : 52;
    const g = document.createElementNS(svgNs, "g");
    g.setAttribute("transform", `translate(${n.x - w / 2}, ${n.y - hh / 2})`);
    g.setAttribute("class", "landing-map__node" + (isPrereq ? " is-prereq" : ""));
    g.setAttribute("data-id", n.id);
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "link");
    g.setAttribute("aria-label", `${n.eyebrow} ${n.title}`);

    const rect = document.createElementNS(svgNs, "rect");
    rect.setAttribute("width", String(w));
    rect.setAttribute("height", String(hh));
    rect.setAttribute("rx", "9");
    rect.setAttribute("fill", isPrereq ? "var(--rl-surface-2)" : "var(--rl-surface)");
    rect.setAttribute("stroke", "var(--rl-ink-muted)");
    rect.setAttribute("stroke-width", "1");
    g.appendChild(rect);

    const eb = document.createElementNS(svgNs, "text");
    eb.setAttribute("x", String(w / 2));
    eb.setAttribute("y", "18");
    eb.setAttribute("text-anchor", "middle");
    eb.setAttribute("fill", "var(--rl-ink-faint)");
    eb.setAttribute("font-family", "Inter, sans-serif");
    eb.setAttribute("font-size", "10");
    eb.setAttribute("letter-spacing", "0.05em");
    eb.textContent = n.eyebrow.toUpperCase();
    g.appendChild(eb);

    const ti = document.createElementNS(svgNs, "text");
    ti.setAttribute("x", String(w / 2));
    ti.setAttribute("y", isPrereq ? "34" : "37");
    ti.setAttribute("text-anchor", "middle");
    ti.setAttribute("fill", "var(--rl-ink)");
    ti.setAttribute("font-family", "Inter, sans-serif");
    ti.setAttribute("font-size", "12");
    ti.setAttribute("font-weight", "600");
    ti.textContent = trimTitle(n.title, isPrereq);
    g.appendChild(ti);

    g.style.cursor = "pointer";
    g.addEventListener("click", () => { location.hash = `#${n.slug}`; });
    g.addEventListener("keydown", (ev: KeyboardEvent) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); location.hash = `#${n.slug}`; }
    });
    g.addEventListener("mouseenter", () => focus(n.id));
    g.addEventListener("focusin", () => focus(n.id));
    g.addEventListener("mouseleave", reset);
    g.addEventListener("focusout", reset);

    nodeLayer.appendChild(g);
  }

  function focus(id: string): void {
    const n = nodeById(id);
    tip.classList.add("is-active");
    (tip.querySelector(".landing-map__tip-eyebrow") as HTMLElement).textContent = n.eyebrow;
    (tip.querySelector(".landing-map__tip-title")   as HTMLElement).textContent = n.title;
    (tip.querySelector(".landing-map__tip-blurb")   as HTMLElement).textContent = n.blurb;
    // Highlight outgoing + incoming edges and adjacent nodes
    const hot = new Set<string>([id]);
    edgeEls.forEach(({ from, to, el }) => {
      const touch = from === id || to === id;
      if (touch) { hot.add(from); hot.add(to); }
      el.setAttribute("stroke", touch ? "var(--rl-algo-ucb)" : "var(--rl-ink-faint)");
      el.setAttribute("stroke-width", touch ? "1.8" : "1.2");
      el.setAttribute("opacity", touch ? "0.95" : "0.18");
      el.setAttribute("marker-end", touch ? "url(#cm-arrow-hot)" : "url(#cm-arrow)");
    });
    nodeLayer.querySelectorAll<SVGGElement>("g.landing-map__node").forEach((g) => {
      const nid = g.getAttribute("data-id");
      g.classList.toggle("is-hot", hot.has(nid!));
      g.classList.toggle("is-dim", !hot.has(nid!));
    });
  }
  function reset(): void {
    tip.classList.remove("is-active");
    (tip.querySelector(".landing-map__tip-eyebrow") as HTMLElement).textContent = "Hover a node";
    (tip.querySelector(".landing-map__tip-title")   as HTMLElement).textContent = "The curriculum at a glance";
    (tip.querySelector(".landing-map__tip-blurb")   as HTMLElement).textContent = "Twelve lessons forming a single learning path from bandits through PPO and max-entropy RL. Roll over any node to see its role; click to jump in.";
    edgeEls.forEach(({ el }) => {
      el.setAttribute("stroke", "var(--rl-ink-faint)");
      el.setAttribute("stroke-width", "1.2");
      el.setAttribute("opacity", "0.55");
      el.setAttribute("marker-end", "url(#cm-arrow)");
    });
    nodeLayer.querySelectorAll<SVGGElement>("g.landing-map__node").forEach((g) => {
      g.classList.remove("is-hot", "is-dim");
    });
  }

  wrap.appendChild(stage);
  return wrap;
}

function trimTitle(s: string, prereq: boolean): string {
  const max = prereq ? 18 : 22;
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}
