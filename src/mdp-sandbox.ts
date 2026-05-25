/**
 * Dev-only sandbox (not in the router, not in the production build) to exercise
 * every GridworldRenderer mode and the MDPEditor before integrating into V1–V7.
 * Open at /mdp-sandbox.html with `npm run dev`.
 */
import "@fontsource/ibm-plex-serif/400.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "./styles/tokens.css";
import "./styles/markov-tokens.css";
import "./styles/mdp-tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/mdp-components.css";

import { buildGridworld, uniformPolicy } from "./mdp/gridworld";
import { policyEvaluationExact } from "./mdp/policy-evaluation";
import { optimalValue, greedyPolicy } from "./mdp/value-iteration";
import { qFromV, advantage } from "./mdp/q-and-advantage";
import { idx, RIGHT, DOWN } from "./mdp/types";
import { GridworldRenderer } from "./components/GridworldRenderer";
import { MDPEditor } from "./components/MDPEditor";

const root = document.getElementById("sandbox")!;

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
const uniform = uniformPolicy(mdp);
const Vpi = policyEvaluationExact(mdp, uniform);
const Vstar = optimalValue(mdp, 200);
const Qstar = qFromV(mdp, Vstar);
const Astar = advantage(Qstar, Vstar);
const greedy = greedyPolicy(mdp, Vstar);

function block(title: string): HTMLElement {
  const sec = document.createElement("section");
  sec.style.margin = "0 0 32px";
  const h = document.createElement("h3");
  h.textContent = title;
  h.style.fontFamily = "var(--rl-font-ui)";
  sec.appendChild(h);
  const host = document.createElement("div");
  sec.appendChild(host);
  root.appendChild(sec);
  return host;
}

// 1. Anatomy: selection + start + terminal glyphs, no values
new GridworldRenderer(block("1 · bare grid (start tag, terminal glyphs, selection)"), {
  mdp,
  startState: idx(0, 0),
  selectedState: idx(1, 0),
  onCellClick: (s) => console.log("click", s),
});

// 2. Policy arrows — uniform
new GridworldRenderer(block("2 · uniform policy arrows"), { mdp, policy: uniform });

// 3. Policy arrows — deterministic greedy
new GridworldRenderer(block("3 · greedy policy arrows"), { mdp, policy: greedy });

// 4. Value heatmap (V^π uniform)
new GridworldRenderer(block("4 · V^π heatmap (uniform)"), { mdp, valueFn: Vpi });

// 5. Value heatmap (V*) + greedy arrows
new GridworldRenderer(block("5 · V* heatmap + greedy arrows"), {
  mdp,
  valueFn: Vstar,
  policy: greedy,
  showArrows: true,
});

// 6. Q quadrants
new GridworldRenderer(block("6 · Q* quadrants"), { mdp, qValues: Qstar, showQuadrants: true });

// 7. Advantage quadrants
new GridworldRenderer(block("7 · A* quadrants (advantage)"), {
  mdp,
  qValues: Astar,
  showQuadrants: true,
  advantageMode: true,
});

// 8. Backup highlight (source + inputs)
new GridworldRenderer(block("8 · backup highlight (source (1,0), inputs)"), {
  mdp,
  valueFn: Vstar,
  highlightState: idx(1, 0),
  highlightInputs: [idx(0, 0), idx(1, 1), idx(2, 0)],
});

// 9. MDPEditor + live re-render
{
  const host = block("9 · MDPEditor drives a live grid");
  let m = buildGridworld({ slippery: false, gamma: 0.9 });
  const gr = new GridworldRenderer(host, { mdp: m, valueFn: policyEvaluationExact(m, uniformPolicy(m)) });
  new MDPEditor(host, {
    opts: { slippery: false, gamma: 0.9 },
    onChange: (opts) => {
      m = buildGridworld(opts);
      gr.update({ mdp: m, valueFn: policyEvaluationExact(m, uniformPolicy(m)) }, { animate: true });
    },
  });
}

console.log("sandbox ready", { Vpi0: Vpi[idx(0, 0)], Vstar0: Vstar[idx(0, 0)], qDown: Qstar[idx(1, 0)][DOWN], qRight: Qstar[idx(1, 0)][RIGHT] });
