/**
 * Curriculum map — full DAG of the 12-lesson curriculum.
 *
 * Layout: two horizontal rows (L1–L7 on top, L8–L12 on the bottom) with the
 * two math prereqs floating above. By default we only render the "spine"
 * edges (the canonical learning order) plus the prereq edges; the full set
 * of secondary forward-arrows is drawn at very low opacity and pops to full
 * brightness when the user hovers any node.
 */

interface MapNode {
  id: string;
  slug: string;
  eyebrow: string;
  title: string;
  blurb: string;
  x: number;
  y: number;
  /** Direct successor(s) used to draw the spine. */
  spine: string[];
  /** All forward edges (secondary edges are hidden until hover). */
  next: string[];
}

const W = 1180;
const H = 480;

const ROW1_Y = 200;
const ROW2_Y = 400;
const PREREQ_Y = 70;

const NODES: MapNode[] = [
  // Math prereqs — top row
  { id: "markov-chains", slug: "markov-chains", eyebrow: "Prereq A", title: "Markov Chains",
    blurb: "The substrate every RL algorithm sits on: P^π is a Markov chain on states.",
    x: 320, y: PREREQ_Y, spine: ["contractions"], next: ["mdps", "contractions"] },
  { id: "contractions", slug: "contractions", eyebrow: "Prereq C", title: "Contractions",
    blurb: "Banach's fixed-point theorem. The reason value iteration converges; the foundation of Lesson 3.",
    x: 530, y: PREREQ_Y, spine: ["dynamic-programming"], next: ["dynamic-programming", "td-learning", "function-approximation", "trpo-ppo"] },

  // Row 1 — foundations & evaluation (L1, L2, L3, L6, L7)
  { id: "bandits", slug: "bandits", eyebrow: "Lesson 1", title: "Multi-Armed Bandits",
    blurb: "ε-greedy, UCB, Thompson — the exploration/exploitation dilemma in its purest form.",
    x: 110, y: ROW1_Y, spine: ["mdps"], next: ["mdps"] },
  { id: "mdps", slug: "mdps", eyebrow: "Lesson 2", title: "MDPs",
    blurb: "The Bellman equations. The spine of the curriculum — every later lesson solves, samples, or generalizes them.",
    x: 320, y: ROW1_Y, spine: ["dynamic-programming"],
    next: ["dynamic-programming", "monte-carlo", "td-learning", "function-approximation", "policy-gradient"] },
  { id: "dynamic-programming", slug: "dynamic-programming", eyebrow: "Lesson 3", title: "Dynamic Programming",
    blurb: "Policy iteration & value iteration: exact MDP solvers when the model is known.",
    x: 530, y: ROW1_Y, spine: ["importance-sampling"],
    next: ["importance-sampling", "monte-carlo", "td-learning", "function-approximation", "policy-gradient"] },
  { id: "importance-sampling", slug: "importance-sampling", eyebrow: "Lesson 6", title: "Importance Sampling",
    blurb: "Reweight off-policy trajectories. The variance bound at the heart of PPO, offline RL, RLHF.",
    x: 740, y: ROW1_Y, spine: ["monte-carlo"], next: ["monte-carlo", "td-learning", "trpo-ppo"] },
  { id: "monte-carlo", slug: "monte-carlo", eyebrow: "Lesson 7", title: "Monte Carlo",
    blurb: "Estimate V^π from full returns. The first model-free method; the λ=1 limit of TD(λ).",
    x: 950, y: ROW1_Y, spine: ["td-learning"], next: ["td-learning", "function-approximation", "policy-gradient"] },

  // Row 2 — model-free & modern RL (L8, L9, L10, L11, L12)
  { id: "td-learning", slug: "td-learning", eyebrow: "Lesson 8", title: "TD Learning",
    blurb: "TD(0), SARSA, Q-learning, n-step, TD(λ). Bootstrapped Bellman backups — RL's pivot.",
    x: 110, y: ROW2_Y, spine: ["function-approximation"],
    next: ["function-approximation", "policy-gradient", "trpo-ppo"] },
  { id: "function-approximation", slug: "function-approximation", eyebrow: "Lesson 9", title: "Function Approx · DQN",
    blurb: "Semi-gradient TD, the deadly triad, and Deep Q-Networks — RL goes neural.",
    x: 320, y: ROW2_Y, spine: ["policy-gradient"], next: ["policy-gradient", "trpo-ppo", "max-ent-rl"] },
  { id: "policy-gradient", slug: "policy-gradient", eyebrow: "Lesson 10", title: "Policy Gradient",
    blurb: "REINFORCE, actor-critic, GAE. Learn the policy directly; the bias-variance tradeoff in advantage estimation.",
    x: 530, y: ROW2_Y, spine: ["trpo-ppo"], next: ["trpo-ppo", "max-ent-rl"] },
  { id: "trpo-ppo", slug: "trpo-ppo", eyebrow: "Lesson 11", title: "TRPO / PPO",
    blurb: "Trust regions, clipped surrogates, GAE — the workhorse algorithms behind modern RL and RLHF.",
    x: 740, y: ROW2_Y, spine: ["max-ent-rl"], next: ["max-ent-rl"] },
  { id: "max-ent-rl", slug: "max-ent-rl", eyebrow: "Lesson 12", title: "Max-Entropy RL",
    blurb: "Soft Bellman, Boltzmann policies. The bridge to SAC, RL-as-inference, and KL-controlled fine-tuning.",
    x: 950, y: ROW2_Y, spine: [], next: [] },
];

function nodeById(id: string): MapNode {
  return NODES.find((n) => n.id === id)!;
}

const SPINE_PAIRS = new Set<string>();
for (const n of NODES) for (const s of n.spine) SPINE_PAIRS.add(`${n.id}→${s}`);
const PREREQ_IDS = new Set(["markov-chains", "contractions"]);
function isPrimary(from: string, to: string): boolean {
  if (SPINE_PAIRS.has(`${from}→${to}`)) return true;
  if (PREREQ_IDS.has(from) && (to === "mdps" || to === "dynamic-programming")) return true;
  return false;
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
  sub.textContent = "Twelve lessons, two math prereqs — wired in a single dependency graph. Bold arrows are the spine you'll walk; hover any node to see its full forward fan-out.";
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

  // Defs — arrow markers
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

  // Edges (two layers: secondary first so primary draws on top)
  const edgeLayer = document.createElementNS(svgNs, "g");
  edgeLayer.setAttribute("class", "landing-map__edges");
  svg.appendChild(edgeLayer);

  type EdgeEl = { from: string; to: string; el: SVGPathElement; primary: boolean };
  const edgeEls: EdgeEl[] = [];

  // Compute edges, place primary edges last so they paint on top
  const edgeSpecs: { from: string; to: string; primary: boolean }[] = [];
  for (const node of NODES) {
    for (const nx of node.next) {
      edgeSpecs.push({ from: node.id, to: nx, primary: isPrimary(node.id, nx) });
    }
  }
  edgeSpecs.sort((a, b) => Number(a.primary) - Number(b.primary));

  for (const { from, to, primary } of edgeSpecs) {
    const a = nodeById(from);
    const b = nodeById(to);
    const path = document.createElementNS(svgNs, "path");
    path.setAttribute("d", edgePath(a, b));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", primary ? "var(--rl-ink-muted)" : "var(--rl-ink-faint)");
    path.setAttribute("stroke-width", primary ? "1.7" : "1");
    path.setAttribute("opacity", primary ? "0.85" : "0.10");
    path.setAttribute("marker-end", "url(#cm-arrow)");
    if (!primary) path.setAttribute("stroke-dasharray", "4 3");
    edgeLayer.appendChild(path);
    edgeEls.push({ from, to, el: path, primary });
  }

  // Row labels (faint)
  for (const [label, y] of [["FOUNDATIONS · EVALUATION", ROW1_Y - 60], ["MODEL-FREE · MODERN RL", ROW2_Y - 60]] as const) {
    const t = document.createElementNS(svgNs, "text");
    t.setAttribute("x", "40");
    t.setAttribute("y", String(y));
    t.setAttribute("fill", "var(--rl-ink-faint)");
    t.setAttribute("font-family", "JetBrains Mono, monospace");
    t.setAttribute("font-size", "9");
    t.setAttribute("letter-spacing", "0.14em");
    t.textContent = label;
    svg.appendChild(t);
  }

  // Flowing particles along the spine path (gives a sense of "flow")
  const spinePathD = spinePath();
  const spine = document.createElementNS(svgNs, "path");
  spine.setAttribute("id", "cm-spine-path");
  spine.setAttribute("d", spinePathD);
  spine.setAttribute("fill", "none");
  spine.setAttribute("stroke", "transparent");
  svg.appendChild(spine);
  for (let i = 0; i < 5; i++) {
    const dot = document.createElementNS(svgNs, "circle");
    dot.setAttribute("r", "3");
    dot.setAttribute("fill", "var(--rl-algo-ucb)");
    dot.setAttribute("opacity", "0.65");
    const anim = document.createElementNS(svgNs, "animateMotion");
    anim.setAttribute("dur", "12s");
    anim.setAttribute("repeatCount", "indefinite");
    anim.setAttribute("begin", `${i * 2.4}s`);
    const mp = document.createElementNS(svgNs, "mpath");
    mp.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", "#cm-spine-path");
    mp.setAttribute("href", "#cm-spine-path");
    anim.appendChild(mp);
    dot.appendChild(anim);
    svg.appendChild(dot);
  }

  // Nodes
  const nodeLayer = document.createElementNS(svgNs, "g");
  nodeLayer.setAttribute("class", "landing-map__nodes");
  svg.appendChild(nodeLayer);

  for (const n of NODES) {
    const isPrereq = n.eyebrow.startsWith("Prereq");
    const w = 156;
    const hh = isPrereq ? 50 : 60;
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
    rect.setAttribute("rx", "10");
    rect.setAttribute("fill", isPrereq ? "var(--rl-surface-2)" : "var(--rl-surface)");
    rect.setAttribute("stroke", "var(--rl-ink-muted)");
    rect.setAttribute("stroke-width", "1");
    g.appendChild(rect);

    const eb = document.createElementNS(svgNs, "text");
    eb.setAttribute("x", String(w / 2));
    eb.setAttribute("y", isPrereq ? "20" : "22");
    eb.setAttribute("text-anchor", "middle");
    eb.setAttribute("fill", "var(--rl-ink-faint)");
    eb.setAttribute("font-family", "Inter, sans-serif");
    eb.setAttribute("font-size", "11");
    eb.setAttribute("letter-spacing", "0.07em");
    eb.textContent = n.eyebrow.toUpperCase();
    g.appendChild(eb);

    const ti = document.createElementNS(svgNs, "text");
    ti.setAttribute("x", String(w / 2));
    ti.setAttribute("y", isPrereq ? "38" : "42");
    ti.setAttribute("text-anchor", "middle");
    ti.setAttribute("fill", "var(--rl-ink)");
    ti.setAttribute("font-family", "Inter, sans-serif");
    ti.setAttribute("font-size", "14");
    ti.setAttribute("font-weight", "600");
    ti.textContent = n.title;
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
    const hot = new Set<string>([id]);
    for (const e of edgeEls) {
      const touch = e.from === id || e.to === id;
      if (touch) { hot.add(e.from); hot.add(e.to); }
      e.el.setAttribute("stroke", touch ? "var(--rl-algo-ucb)" : (e.primary ? "var(--rl-ink-muted)" : "var(--rl-ink-faint)"));
      e.el.setAttribute("opacity", touch ? "0.95" : (e.primary ? "0.35" : "0.05"));
      e.el.setAttribute("stroke-width", touch ? "2" : (e.primary ? "1.7" : "1"));
      e.el.setAttribute("marker-end", touch ? "url(#cm-arrow-hot)" : "url(#cm-arrow)");
    }
    nodeLayer.querySelectorAll<SVGGElement>("g.landing-map__node").forEach((gn) => {
      const nid = gn.getAttribute("data-id");
      gn.classList.toggle("is-hot", hot.has(nid!));
      gn.classList.toggle("is-dim", !hot.has(nid!));
    });
  }
  function reset(): void {
    tip.classList.remove("is-active");
    (tip.querySelector(".landing-map__tip-eyebrow") as HTMLElement).textContent = "Hover a node";
    (tip.querySelector(".landing-map__tip-title")   as HTMLElement).textContent = "The curriculum at a glance";
    (tip.querySelector(".landing-map__tip-blurb")   as HTMLElement).textContent = "Twelve lessons forming a single learning path from bandits through PPO and max-entropy RL. Roll over any node to see its role; click to jump in.";
    for (const e of edgeEls) {
      e.el.setAttribute("stroke", e.primary ? "var(--rl-ink-muted)" : "var(--rl-ink-faint)");
      e.el.setAttribute("opacity", e.primary ? "0.85" : "0.10");
      e.el.setAttribute("stroke-width", e.primary ? "1.7" : "1");
      e.el.setAttribute("marker-end", "url(#cm-arrow)");
    }
    nodeLayer.querySelectorAll<SVGGElement>("g.landing-map__node").forEach((gn) => {
      gn.classList.remove("is-hot", "is-dim");
    });
  }

  wrap.appendChild(stage);
  return wrap;
}

/** Cubic-Bezier edge between two nodes. Within-row edges are clean S-curves;
 *  cross-row edges (especially the L7→L8 wrap-back) flare with larger tangents. */
function edgePath(a: MapNode, b: MapNode): string {
  const exitX = a.x + 78;       // right side of source node
  const entryX = b.x - 78;      // left side of target node
  const dy = Math.abs(b.y - a.y);
  const dx = Math.abs(b.x - a.x);
  const sameRow = dy < 20;
  if (sameRow) {
    const tan = Math.max(40, dx * 0.4);
    return `M ${exitX} ${a.y} C ${exitX + tan} ${a.y}, ${entryX - tan} ${b.y}, ${entryX} ${b.y}`;
  }
  // Cross-row: exit tangent right, enter tangent right (curve swings through the page).
  const tan = Math.max(200, dx * 0.45);
  return `M ${exitX} ${a.y} C ${exitX + tan} ${a.y}, ${entryX - tan} ${b.y}, ${entryX} ${b.y}`;
}

/** Path used by the flowing particles. Goes through every node in canonical order. */
function spinePath(): string {
  const order = ["bandits", "mdps", "dynamic-programming", "importance-sampling", "monte-carlo", "td-learning", "function-approximation", "policy-gradient", "trpo-ppo", "max-ent-rl"];
  const pts = order.map((id) => nodeById(id));
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    const cx = (a.x + b.x) / 2;
    d += ` C ${cx} ${a.y}, ${cx} ${b.y}, ${b.x} ${b.y}`;
  }
  return d;
}
