import type { LessonMeta } from "../meta";

export const maxentMeta: LessonMeta = {
  id: "max-ent-rl",
  title: "Maximum-Entropy RL and RL-as-Inference",
  subtitle: "When stochasticity is the objective, not the regularizer",
  tier: 2,
  difficulty: 5,
  estimatedReadMinutes: 60,
  role: "theory-bridge",
  prerequisites: [
    { label: "Lesson 11 — TRPO/PPO", to: "trpo-ppo", anchor: "l11-forward-links" },
    { label: "Lesson 10 — Policy Gradient", to: "policy-gradient", anchor: "policy-gradient-motivation" },
    { label: "Lesson 8 — TD Learning", to: "td-learning", anchor: "bellman-backup" },
    { label: "Lesson 3 — Dynamic Programming", to: "dynamic-programming", anchor: "policy-iteration" },
    { label: "Prereq C — Contractions", to: "contractions", anchor: "banach-fixed-point" },
  ],
  exportedAnchors: [
    "why-stochastic",
    "entropy-regularized-objective",
    "soft-bellman",
    "boltzmann-policy",
    "maxent-failure-mode",
    "entropy-slider-lab",
    "rl-as-inference",
    "l12-forward-links",
  ],
  centerpieceComponent: "EntropySliderLab",
  forwardLinksWhenReady: [
    { to: "sac", anchor: "continuous-soft-actor-critic" },
    { to: "rlhf-dpo", anchor: "kl-to-reference-objective" },
  ],
};
