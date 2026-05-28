import type { LessonMeta } from "../meta";

export const ppoMeta: LessonMeta = {
  id: "trpo-ppo",
  title: "Trust Regions: TRPO and PPO",
  subtitle: "Stepping carefully in policy space",
  tier: 2,
  difficulty: 5,
  estimatedReadMinutes: 65,
  role: "workhorse",
  prerequisites: [
    { label: "Lesson 10 — Policy Gradient", to: "policy-gradient", anchor: "policy-gradient-motivation" },
    { label: "Lesson 9 — Function Approx & DQN", to: "function-approximation", anchor: "deadly-triad" },
    { label: "Lesson 6 — Importance Sampling", to: "importance-sampling", anchor: "is-identity" },
    { label: "Lesson 3 — Dynamic Programming", to: "dynamic-programming", anchor: "policy-iteration" },
  ],
  exportedAnchors: [
    "step-size-problem",
    "trust-region-kl",
    "trpo-algorithm",
    "ppo-clipped",
    "gae-full",
    "ppo-lab",
    "ppo-empirics",
    "l11-forward-links",
  ],
  centerpieceComponent: "PPOLab",
  forwardLinksWhenReady: [
    { to: "max-ent-rl", anchor: "entropy-regularized-objective" },
    { to: "sac", anchor: "continuous-actor-critic" },
    { to: "rlhf", anchor: "ppo-for-language-models" },
  ],
};
