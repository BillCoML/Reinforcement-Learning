import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { sidebar } from "../../components/CrosslinkCallout";

export const mdpSection01: Section = {
  id: "mdp-tuple",
  title: "From Bandits + Chains to MDPs",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§1</span>From Bandits + Chains to MDPs</h2>
<p class="tagline">Add a control knob to a Markov chain. Watch it become decision-making.</p>

<p>Two lessons ago, the agent chose between $K$ arms in a single state — Bandits.
The world didn't move; only the agent's beliefs did. One lesson ago, a system
evolved on its own via a transition matrix $P$ — Markov Chains. The agent didn't
exist; the world moved without input.</p>

<p>The <strong>Markov Decision Process</strong> is the synthesis. The agent observes a
state $s$, chooses an action $a$, and the world responds: it transitions to a new
state $s'$ and delivers a scalar reward $r$. The transition distribution depends on
both the current state <em>and</em> the chosen action.</p>

${sidebar(
  "Back-link → Bandits",
  `<p>Bandits is the special case $|\\mathcal{S}| = 1$. There is no “next state” because
  there's only one. The Bellman equation we'll derive in §6 collapses to
  $V^\\pi = \\sum_a \\pi(a)\\, r(a)$: the policy's expected immediate reward. That's exactly
  the quantity ε-greedy and UCB estimate.</p>`,
)}

<p>Formally, a <strong>finite Markov Decision Process</strong> is a tuple</p>

$$\\boxed{\\mathcal{M} \\;=\\; (\\mathcal{S},\\, \\mathcal{A},\\, P,\\, r,\\, \\gamma)}$$

<p>with:</p>
<ul>
  <li>$\\mathcal{S}$: a finite <strong>state space</strong>, $|\\mathcal{S}| = K$.</li>
  <li>$\\mathcal{A}$: a finite <strong>action space</strong>, $|\\mathcal{A}| = m$. (We assume the
  same action set is available in every state, but everything generalizes to
  state-dependent action sets $\\mathcal{A}(s)$.)</li>
  <li>$P$: a <strong>transition kernel</strong>,
  $P(s' \\mid s, a) = \\Pr(S_{t+1} = s' \\mid S_t = s, A_t = a)$. For each $(s,a)$, the row
  $P(\\cdot \\mid s, a)$ is a probability distribution over $\\mathcal{S}$.</li>
  <li>$r$: a <strong>reward function</strong>,
  $r(s, a) = \\mathbb{E}[R_{t+1} \\mid S_t = s, A_t = a]$ — the expected immediate reward
  as a function of the <em>current</em> state and action.</li>
  <li>$\\gamma \\in [0, 1)$: the <strong>discount factor</strong>. How much the agent values
  future reward vs immediate reward (§3).</li>
</ul>

<p><strong>The Markov property carries over from Prereq A</strong>: the transition and reward
depend only on the current $(s, a)$, not on the full history. This is what makes the
whole apparatus tractable.</p>

<h3>Running example: the 3×3 gridworld</h3>
<p>We'll use this single example throughout the lesson. Memorize it.</p>
<ul>
  <li>$\\mathcal{S}$ = the nine grid cells, indexed $(r, c)$ with $r, c \\in \\{0,1,2\\}$.</li>
  <li>$\\mathcal{A} = \\{\\text{Up}, \\text{Right}, \\text{Down}, \\text{Left}\\}$.</li>
  <li>Transitions are <strong>deterministic</strong> in the basic version: the agent moves one
  cell in the chosen direction. Walls bounce: the agent stays in place. (A stochastic
  variant appears in §6.)</li>
  <li>Rewards: $+1$ upon entering the goal, $-1$ upon entering the pit, $0$ otherwise.
  The reward is collected on the transition <em>into</em> a state, not for being in it.</li>
  <li>Pit and goal are <strong>terminal</strong>: once entered, the episode ends.</li>
  <li>Discount $\\gamma = 0.9$.</li>
</ul>
<p>Nine states, four actions, 36 $(s,a)$ pairs — small enough that the full transition
table fits on one screen, yet the optimal policy must <em>navigate around the pit</em>: a
genuine planning problem.</p>

<p>The point of the explorer below is <em>tactile mastery of the dynamics</em>. Click a cell,
pick an action, and read off where you land, with what probability, for what reward.
Flip on “slippery” to see the action fan out into the 80-10-10 slip distribution.</p>

<mdp-anatomy-explorer></mdp-anatomy-explorer>`,
    );
  },
};
