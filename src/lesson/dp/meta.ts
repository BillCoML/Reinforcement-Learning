import type { LessonMeta } from '../meta';

export const dpMeta: LessonMeta = {
  id: 'dynamic-programming',
  title: 'Dynamic Programming',
  subtitle: 'Policy iteration, value iteration, and the GPI pattern',
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 70,
  role: 'critical-path',
  prerequisites: [
    { label: 'Lesson 2 — MDPs', to: 'mdps', anchor: 'bellman-expectation' },
    { label: 'Lesson 2 — MDPs', to: 'mdps', anchor: 'bellman-optimality' },
    { label: 'Prereq C — Contractions', to: 'contractions', anchor: 'bellman-pi-contraction' },
    { label: 'Prereq C — Contractions', to: 'contractions', anchor: 'bellman-star-contraction' },
  ],
  exportedAnchors: [
    'iterative-policy-evaluation',
    'policy-improvement-theorem',
    'policy-iteration',
    'value-iteration',
    'modified-policy-iteration',
    'asynchronous-dp',
    'gauss-seidel',
    'generalized-policy-iteration',
    'stopping-criterion',
  ],
  centerpieceComponent: 'DPAlgorithmLab',
  forwardLinksWhenReady: [
    { to: 'monte-carlo',     anchor: 'model-free-evaluation' },
    { to: 'td-learning',     anchor: 'td-as-sampled-dp' },
    { to: 'dqn',             anchor: 'deep-vi' },
    { to: 'policy-gradient', anchor: 'actor-critic-as-gpi' },
  ],
};
