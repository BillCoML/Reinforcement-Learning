import type { Section } from '../section';
import { sectionFromHTML } from '../section';
import { forwardLink } from '../../components/CrosslinkCallout';

export const dpSection01: Section = {
  id: 'known-model-planning',
  title: 'Setting Up: Known-Model Planning',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§1</span>Setting Up: Known-Model Planning</h2>
<p class="tagline">We have the MDP. Now what?</p>

<p>For the first and last time in this curriculum, we'll assume we know the
transition kernel $P$ and the reward function $r$ exactly. This is the
<strong>planning</strong> setting: the agent has a complete model of the world and needs to
compute the optimal policy. There's no learning, no estimation, no exploration —
just computation.</p>

<p>Why bother, if real RL doesn't have this? Three reasons.</p>

<ol>
  <li><strong>The classical algorithms are extraordinarily clean.</strong> Policy
    iteration and value iteration are short, elegant, and <em>exactly</em> solve the MDP.
    They serve as a reference point — the gold standard against which all model-free
    algorithms are compared.</li>
  <li><strong>The pattern generalizes.</strong> The "interleave policy evaluation and
    policy improvement" pattern that defines policy iteration is what actor-critic
    methods (Lesson 8), DQN (Lesson 7), and most modern RL algorithms approximate.
    Master the pattern in its clean DP form; everything that follows is variations.</li>
  <li><strong>Some real problems actually have models.</strong> Board games, robotic
    systems with good simulators, financial decision processes with well-understood
    dynamics — all admit some flavor of DP. AlphaZero's MCTS is a generalization of
    value iteration; Dreamer's planning step (Lesson 14) does DP in a learned world
    model.</li>
</ol>

${forwardLink({
  destination: 'Lesson 2 — MDPs',
  ready: true,
  href: '#mdps',
  html: `<p>Lesson 2 defined $V^\\pi$, $Q^\\pi$, $V^*$, $Q^*$ and the two Bellman equations
    they satisfy. This lesson does nothing more than iteratively solve those equations.
    Prereq C (Contractions) proved that the iteration converges. With those two
    foundations in hand, every algorithm in this lesson can be written down in a
    few lines.</p>`,
})}

<p>We continue with our 3×3 gridworld from Lesson 2. Goal at (2,2), pit at (1,1),
$\\gamma = 0.9$. All numerics below are reproducible by the in-browser implementation.</p>`);
  },
};
