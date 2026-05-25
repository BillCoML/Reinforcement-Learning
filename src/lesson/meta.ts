/** Lesson metadata. The curriculum's catalog will eventually consume this. */

export interface Prerequisite {
  external?: boolean;
  label: string;
  to?: string;
  anchor?: string;
}

export interface ForwardLink {
  to: string;
  anchor: string;
}

export interface LessonMeta {
  id: string;
  title: string;
  subtitle: string;
  tier: number;
  difficulty: number; // out of 5
  estimatedReadMinutes: number;
  prerequisites: Prerequisite[];
  exportedAnchors: string[];
  centerpieceComponent: string;
  forwardLinksWhenReady: ForwardLink[];
}

export const lessonMeta: LessonMeta = {
  id: "bandits",
  title: "Multi-Armed Bandits",
  subtitle: "Exploration, exploitation, and the geometry of regret",
  tier: 1,
  difficulty: 2,
  estimatedReadMinutes: 45,
  prerequisites: [
    { external: true, label: "Basic probability (expectation, variance)" },
    { external: true, label: "Comfort with O(·) notation" },
  ],
  exportedAnchors: [
    "regret-definition",
    "exploration-exploitation",
    "epsilon-greedy",
    "ucb1",
    "thompson-sampling",
    "lai-robbins-lower-bound",
    "hoeffding-inequality",
    "beta-bernoulli-conjugacy",
  ],
  centerpieceComponent: "AlgorithmBattleArena",
  forwardLinksWhenReady: [
    { to: "mdps", anchor: "bandit-as-1-state-mdp" },
    { to: "dqn", anchor: "exploration-in-deep-rl" },
    { to: "rlhf", anchor: "preference-bandits" },
  ],
};

/** Curriculum-wide algorithm signature colors (CSS variable names). */
export const ALGO_COLORS: Record<string, string> = {
  greedy: "var(--rl-algo-greedy)",
  eps001: "var(--rl-algo-greedy)",
  eps01: "var(--rl-algo-greedy)",
  ucb: "var(--rl-algo-ucb)",
  ucb1: "var(--rl-algo-ucb)",
  thompson: "var(--rl-algo-thompson)",
  random: "var(--rl-algo-random)",
  optimal: "var(--rl-algo-optimal)",
};
