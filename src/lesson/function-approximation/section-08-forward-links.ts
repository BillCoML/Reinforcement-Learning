import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection08: Section = {
  id: "dqn-family",
  title: "DQN in Practice and Forward Links",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§8</span>DQN in Practice and Forward Links</h2>
<p class="tagline"><em>When DQN works, when it doesn't, and what comes next.</em></p>

<p>DQN works well on <strong>discrete-action environments</strong> with non-sparse rewards
and abundant simulated data (Atari, board games, simple robotics with discretized control).
It works poorly on continuous-action environments (the $\\max$ is intractable), very
sparse-reward settings (signal-to-noise degrades), and settings with limited data.</p>

<p>The DQN family dominates the discrete-action benchmark literature but has been
overshadowed by policy-gradient methods in continuous control and RLHF.</p>

<h3>Where this goes</h3>

<p><strong>Lesson 10 (Policy Gradient).</strong> Policy gradient methods directly parameterize
the policy $\\pi_\\theta(a \\mid s)$, handle continuous actions naturally, and admit
actor-critic extensions where the "critic" is a value-function approximator trained with
TD — inheriting all the DQN machinery.</p>

<p><strong>Lesson 11 (TRPO/PPO).</strong> Uses TD-style critics (Generalized Advantage
Estimation) for variance reduction. DQN's contribution to TRPO/PPO is the critic's value
estimation, not the policy itself.</p>

<p><strong>Lesson 13 (SAC).</strong> Soft Actor-Critic uses <em>two</em> Q-networks,
addressing maximization bias — a direct descendant of Double DQN, refined for
continuous-action settings.</p>

<p><strong>Lesson 15 (Offline RL).</strong> Offline value learning revisits Q-learning when
no further data can be collected. The deadly triad becomes acute (extreme distribution
mismatch); remedies like CQL modify the Q-learning update to be conservative. DQN
machinery is the substrate.</p>

<p><strong>Lesson 16 / Lesson 17 (Diffusion &amp; RLHF).</strong> DQN-style value learning
appears as a supporting component in diffusion-based policies and reward-model training.</p>

<dqn-roadmap></dqn-roadmap>
`);
  },
};
