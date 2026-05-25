/** MDP lesson metadata. Mirrors the curriculum LessonMeta shape. */
import type { LessonMeta } from "../meta";

export const mdpMeta: LessonMeta = {
  id: "mdps",
  title: "Markov Decision Processes",
  subtitle: "The formal model of sequential decision-making",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 70,
  role: "critical-path",
  prerequisites: [
    { label: "Lesson 1 — Bandits", to: "bandits", anchor: "regret-definition" },
    { label: "Prereq A — Markov Chains", to: "markov-chains", anchor: "policy-induced-chain" },
  ],
  exportedAnchors: [
    "mdp-tuple",
    "policy",
    "return-discount",
    "state-value-function",
    "action-value-function",
    "advantage-function",
    "bellman-expectation",
    "bellman-optimality",
    "optimal-policy-exists-deterministic",
    "matrix-form-policy-evaluation",
  ],
  centerpieceComponent: "BellmanBackupLab",
  forwardLinksWhenReady: [
    { to: "dynamic-programming", anchor: "policy-iteration" },
    { to: "td-learning", anchor: "sample-based-bellman" },
    { to: "policy-gradient", anchor: "advantage-actor-critic" },
    { to: "max-ent-rl", anchor: "soft-bellman" },
  ],
};
