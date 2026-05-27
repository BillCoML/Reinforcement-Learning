import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const isSection05: Section = {
  id: 'is-forward-links',
  title: 'Where You\'ll See This Again',
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>Where You'll See This Again</h2>
<p class="tagline">Four downstream lessons depend on this prereq, and one of them gets clipped.</p>

<p>Four subsequent lessons depend directly on this one. Naming them now pre-loads
the recognition that will happen in each lesson when the importance-sampling
machinery first appears.</p>

<p>The first is <strong>Lesson 7 (Monte Carlo)</strong>. Monte Carlo methods sample
trajectories under a fixed policy and estimate the value function by averaging the
returns. The on-policy version requires no importance sampling, since the trajectories
are sampled under the same policy being evaluated. The off-policy version is exactly
the trajectory IS apparatus we just built, applied directly. Lesson 7's off-policy MC
section will reference the trajectory IS estimator, the weighted IS estimator, and the
effective-sample-size diagnostic by name.</p>

<p>The second is <strong>Lesson 8 (Temporal-Difference Learning)</strong>. TD methods
do one-step backups, replacing trajectories with $(s, a, r, s')$ tuples. The
importance ratio for off-policy TD therefore degrades from a length-$T$ product to a
single per-step ratio $\\pi(a \\mid s)/\\mu(a \\mid s)$. This is vastly lower variance
than trajectory IS but brings its own subtleties (the deadly triad of bootstrapping
plus off-policy plus function approximation, which we will meet in Lesson 9).
Q-learning is a clever scheme that maximally collapses the IS apparatus by
re-deriving the off-policy target as a max over actions, avoiding the ratio entirely.</p>

<p>The third is <strong>Lesson 11 (Trust Region and Proximal Methods)</strong>. When
optimizing a parametric policy $\\pi_\\theta$, the surrogate objective uses trajectories
collected under the previous parameter setting and weights them by the policy ratio
$\\pi_\\theta(a \\mid s)/\\pi_{\\theta_{\\text{old}}}(a \\mid s)$. This is importance
sampling in disguise. TRPO's KL constraint and PPO's ratio clipping are both
techniques for bounding the IS variance — keeping the ratio close to 1 so the
estimator stays trustworthy. Without this prereq, both algorithms read like
incomprehensible heuristics.</p>

<p>The fourth is <strong>Lesson 15 (Offline RL)</strong>. Offline RL has a fixed
dataset collected under some unknown behavior policy and the goal is to evaluate or
improve a different policy. Evaluating any policy requires importance sampling, and
the variance problem is catastrophic in offline settings. Almost the entire offline-RL
toolkit — conservative Q-learning, behavior regularization, density-ratio estimation,
BCQ's action constraint — is working around the IS variance in some way. Lesson 15
will spend a section diagnosing exactly which IS variance issue each method targets.</p>

<div class="component-host">
  <roadmap-mini active="importance-sampling"></roadmap-mini>
</div>
`);
  },
};
