import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection08: Section = {
  id: "td-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§8</span>Where You'll See This Again</h2>
<p class="tagline"><em>Four downstream lessons build directly on TD.</em></p>

<p><strong>Lesson 9 (Function Approximation and DQN)</strong> is the most direct descendant.
DQN is Q-learning with two modifications: the Q function is parameterized by a neural
network, and a target network provides a delayed bootstrap to stabilize the dynamics.
The convergence-theory section of Lesson 9 picks up exactly where Section 7 of this
lesson left off: the projected Bellman operator's contraction properties, the deadly
triad, and the engineering tactics (target networks, experience replay, prioritized
replay) that recover stability. Double DQN, dueling DQN, and rainbow DQN are all
direct extensions.</p>

<p><strong>Lesson 10 (Policy Gradient)</strong> uses TD machinery for the value baseline that
reduces gradient variance. The advantage function
$A^\\pi(s, a) = Q^\\pi(s, a) - V^\\pi(s)$ is estimated by some flavor of TD, and the
$n$-step / GAE machinery from Sections 5 and 6 of this lesson is exactly what GAE
generalizes. Policy gradient's "critic" in actor-critic methods is a TD value function.</p>

<p><strong>Lesson 11 (TRPO and PPO)</strong> continues the actor-critic line and uses GAE explicitly.
The bias-variance tuning via $\\lambda$ that we developed here for TD($\\lambda$) is the
same dial that PPO exposes via its GAE-$\\lambda$ parameter.</p>

<p><strong>Lesson 15 (Offline RL)</strong> revisits the off-policy TD setting with emphasis on data
distribution shift. Retrace, V-trace, and the conservative-Q-learning family are all
extensions of the off-policy TD apparatus introduced here. The maximization-bias concern
from Section 4 becomes critical in offline RL, where the agent cannot collect new data
to correct over-optimistic estimates.</p>

<div class="component-host">
  <roadmap-mini active="td-learning"></roadmap-mini>
</div>
`);
  },
};
