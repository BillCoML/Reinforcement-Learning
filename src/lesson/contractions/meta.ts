import type { LessonMeta } from "../meta";

export const contractionsMeta: LessonMeta = {
  id: "contractions",
  title: "Contractions and the Banach Fixed-Point Theorem",
  subtitle: "The engine of dynamic programming",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 35,
  role: "prereq",
  prerequisites: [
    { external: true, label: "Linear algebra: norms, matrix operator norms" },
    { external: true, label: "Real analysis: sequences and limits" },
    { label: "Lesson 2 — MDPs", to: "mdps", anchor: "bellman-expectation" },
  ],
  exportedAnchors: [
    "metric-space",
    "contraction-property",
    "lipschitz-constant",
    "banach-fixed-point",
    "geometric-error-bound",
    "completeness-required",
    "bellman-pi-contraction",
    "bellman-star-contraction",
  ],
  centerpieceComponent: "BellmanContractionExplorer",
  forwardLinksWhenReady: [
    { to: "dynamic-programming", anchor: "value-iteration" },
    { to: "dynamic-programming", anchor: "policy-iteration" },
    { to: "td-learning",         anchor: "td-convergence-theory" },
    { to: "function-approx",     anchor: "deadly-triad" },
  ],
};
