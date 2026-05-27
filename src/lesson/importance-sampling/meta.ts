import type { LessonMeta } from '../meta';

export const isMeta: LessonMeta = {
  id: 'importance-sampling',
  title: 'Importance Sampling',
  subtitle: 'Evaluating one distribution with samples from another',
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 40,
  role: 'critical-path',
  prerequisites: [
    { external: true, label: 'Probability: expectations, variance, change of variables' },
    { label: 'Lesson 2 — MDPs', to: 'mdps', anchor: 'state-value-function' },
    { label: 'Lesson 3 — Dynamic Programming', to: 'dynamic-programming', anchor: 'iterative-policy-evaluation' },
  ],
  exportedAnchors: [
    'is-identity',
    'ordinary-is-estimator',
    'weighted-is-estimator',
    'is-variance',
    'infinite-variance-condition',
    'trajectory-is',
    'per-decision-is',
    'effective-sample-size',
  ],
  centerpieceComponent: 'TrajectoryISExplorer',
  forwardLinksWhenReady: [
    { to: 'monte-carlo',   anchor: 'off-policy-mc' },
    { to: 'td-learning',   anchor: 'off-policy-td' },
    { to: 'trust-region',  anchor: 'ppo-clipped-ratio' },
    { to: 'offline-rl',    anchor: 'distribution-shift' },
  ],
};
