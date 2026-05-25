/** Markov Chains lesson metadata. Mirrors the curriculum LessonMeta shape. */
import type { LessonMeta } from "../meta";

export const markovMeta: LessonMeta = {
  id: "markov-chains",
  title: "Markov Chains and Stationary Distributions",
  subtitle: "The substrate of every policy",
  tier: 1,
  difficulty: 2,
  estimatedReadMinutes: 50,
  role: "prereq", // not on the main critical path; lands before Lesson 2
  prerequisites: [
    { external: true, label: "Linear algebra: matrix multiplication, eigenvalues" },
    { external: true, label: "Probability: conditional probability, expectations" },
  ],
  exportedAnchors: [
    "markov-property",
    "transition-matrix",
    "n-step-transitions",
    "communicating-classes",
    "irreducible-aperiodic",
    "stationary-distribution",
    "ergodic-theorem",
    "detailed-balance",
    "policy-induced-chain",
  ],
  centerpieceComponent: "ConvergenceLab",
  forwardLinksWhenReady: [
    { to: "mdps", anchor: "policy-induced-chain" },
    { to: "td-learning", anchor: "stationary-distribution-sampling" },
    { to: "policy-gradient", anchor: "on-policy-distribution" },
  ],
};
