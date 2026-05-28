import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection01: Section = {
  id: "tabular-limits",
  title: "Why Tabular Fails",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>Why Tabular Fails</h2>
<p class="tagline"><em>Q-tables don't scale. And cannot generalize. Two related but distinct problems.</em></p>

<p>The Q-learning algorithm from Lesson 8 is a small marvel. With a finite state space,
Robbins-Monro step sizes, and adequate exploration, it converges to $Q^*$ with probability
one. The proof extends the contraction-theoretic machinery of Lesson 4 directly: the Bellman
optimality operator is a $\\gamma$-contraction in the sup norm, and Q-learning is a noisy
stochastic approximation of one application of the operator per transition.</p>

<p>The catch is that the algorithm stores $Q(s, a)$ for every state-action pair in a lookup
table. The table's size is $|\\mathcal{S}| \\times |\\mathcal{A}|$. For the curriculum's
$3 \\times 3$ gridworld with 9 states and 4 actions, that is 36 cells. Tabular Q-learning
is trivial in this regime.</p>

<p>For Atari games — the canonical deep RL benchmark — the state space is the set of
$84 \\times 84 \\times 4$ byte arrays (four stacked grayscale frames at 84-pixel resolution).
Each pixel takes 256 values. The total number of distinguishable states is
$256^{84 \\times 84 \\times 4} \\approx 10^{67{,}914}$,
a number larger than the number of atoms in the observable universe by an unfathomable
margin. A Q-table indexed by Atari frames is not merely impractical — it is physically
impossible.</p>

<p>The same calculation rules out tabular methods for Go ($\\sim 3^{361} \\approx 10^{172}$
board positions), chess ($\\sim 10^{47}$), continuous robot control (infinitely many
states), and almost every real-world setting.</p>

<h3>Generalization, not just storage</h3>

<p>The size argument is decisive on its own. There is a second, deeper argument. Suppose we
had infinite memory and could afford a trillion-entry Q-table. The agent has visited a
million states. What should it predict for the trillions of unvisited states?</p>

<p>The tabular algorithm has no answer. Each entry is independent of every other. The fact
that some states are "almost the same as" states the agent has seen is invisible to the
algorithm. Every new state is a cold start.</p>

<p>Function approximation imposes a <em>smoothness</em> on the Q-function: a parametric form
$Q_\\theta(s, a)$ where $\\theta$ is a low-dimensional parameter vector. Once $\\theta$ is
learned from observed transitions, $Q_\\theta(s, a)$ produces <em>predictions</em> for any
input, including unseen states. The prediction quality depends on how well the function class
captures the true $Q^*$ and how representative the training data was. But predictions exist,
where the tabular algorithm produced only "undefined."</p>

<p>This is the central pivot of the lesson. We are not introducing function approximation
just to save memory — we are introducing it to enable <strong>generalization</strong>. The
two motivations are related but distinct, and only function approximation addresses both.</p>

<blockquote class="crosslink">
  <strong>Crosslink to Lesson 8.</strong> The deadly-triad preview in §8 of Lesson 8 named
  the combination of "function approximation + bootstrapping + off-policy learning" as the
  source of trouble. We are about to add the first of those three ingredients to our
  Q-learning algorithm. The next three sections trace the consequences.
</blockquote>

<dqn-tabular-limits></dqn-tabular-limits>
`);
  },
};
