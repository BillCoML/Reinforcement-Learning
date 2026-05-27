import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const dpSection08: Section = {
  id: 'dp-forward-links',
  title: 'Where You\'ll See This Again',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§8</span>Where You'll See This Again</h2>
<p class="tagline">DP is the gold standard. Every model-free RL algorithm approximates it.</p>

<p>We've shipped two algorithms — policy iteration and value iteration — that <em>exactly</em>
solve any finite MDP in polynomial time, assuming we know the model. The rest of the
curriculum relaxes the "know the model" assumption, one direction at a time.</p>

<p><strong>Direction 1 — Unknown model, known returns.</strong> Lesson 4 (Monte Carlo) estimates
$V^\\pi$ and $Q^\\pi$ by averaging <em>observed</em> returns from sampled trajectories. No model
$(P, r)$ is needed. The downside: trajectories must reach a terminal state before any
update can happen, which is slow and only works for episodic tasks.</p>

<p><strong>Direction 2 — Unknown model, sampled Bellman backups.</strong> Lesson 5 (Temporal-Difference
Learning) replaces the expectation in the Bellman backup,
$\\sum_{s'} P(s'|s,a) V(s')$, with the single sample $V(s')$ from one observed
transition. <em>One-step samples</em> replace <em>full trajectories</em>. The benefit: updates
happen at every step, not at the end of episodes.</p>

<p><strong>Direction 3 — Large/continuous state spaces.</strong> Lesson 6 (Function Approximation)
replaces tabular $V$ with $V_\\theta$, a parameterized function. The Bellman backup
becomes a gradient step on a Bellman residual. Convergence guarantees become subtler
(the "deadly triad" warning), and we trade exactness for scalability.</p>

<p><strong>Direction 4 — Deep RL.</strong> Lesson 7 (DQN) combines directions 2 and 3: sampled
Bellman backups against a deep-net $Q_\\theta$, with target networks and replay buffers
added to stabilize. The Bellman optimality equation is <em>exactly</em> what DQN's loss
function is squaring.</p>

<p><strong>Direction 5 — Policy parameterization.</strong> Lesson 8 (Policy Gradient) takes a
different route: instead of learning $V$ or $Q$ and acting greedily, learn $\\pi_\\theta$
directly. The improvement step becomes a <em>gradient step</em> on $J(\\theta)$, computed via
the policy gradient theorem.</p>

<p><strong>Direction 6 — Model-based modern.</strong> Lesson 13 (Model-Based RL) learns the model
$(P_\\theta, r_\\theta)$ from data and then <em>runs DP inside it</em>. Both are direct heirs of
the DP we just built.</p>

<p>This is where the curriculum's "knowing the model" assumption ends. Everything after
is RL proper.</p>

<roadmap-mini active="dynamic-programming"></roadmap-mini>`);
  },
};
