import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection04: Section = {
  id: "neural-q-network",
  title: "Neural Networks as Function Approximators",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>Neural Networks as Function Approximators</h2>
<p class="tagline"><em>Same idea, more capacity. Backprop, ReLU, target networks.</em></p>

<p>Linear function approximation requires hand-designed features $\\phi(s)$. For images,
joint angles, or any input where "natural" features are unknown, this is a major limitation.
Neural networks remove it: a multi-layer perceptron or convolutional network learns the
features from data, taking the raw state as input.</p>

<p>The Q-network for our $7 \\times 7$ gridworld is a small MLP:</p>

<pre class="rl-pre">Input: state index, one-hot encoded → ℝ⁴⁹
Hidden 1: linear (49 → 64), ReLU
Hidden 2: linear (64 → 64), ReLU
Output: linear (64 → 4)   # Q-values for 4 actions</pre>

<p>About 8,000 parameters. The training data is $(s, a, r, s', \\text{done})$ transitions.
The target is the Bellman-optimality target
$r + \\gamma \\max_{a'} Q_\\theta(s', a')$ and the loss is squared error between the
network's $Q(s, a)$ and this target.</p>

<h3>Why the basic recipe fails</h3>

<p>Naive Q-learning with a neural network is the deadly triad in extreme form.
Two structural problems are easy to identify:</p>

<p><strong>Moving target.</strong> At each SGD step, we update $\\theta$ to better fit the
target $r + \\gamma \\max_{a'} Q_\\theta(s', a')$. But this target depends on $\\theta$
itself — the very parameter we are updating. As $\\theta$ changes, the target changes;
the gradient direction moves. The optimization is chasing its own tail. With function
approximation, every update changes the target for <em>every</em> state.</p>

<p><strong>Correlated samples.</strong> SGD's convergence theory assumes independent gradients.
The data from a trajectory is highly correlated: consecutive states are nearby. Correlated
gradients push $\\theta$ in the same direction for many steps, causing overshooting and
instability.</p>

<p>The next two sections introduce the two key DQN fixes. Target networks address the
moving-target problem. Experience replay addresses the correlated-samples problem.</p>

<dqn-q-network-architecture></dqn-q-network-architecture>
`);
  },
};
