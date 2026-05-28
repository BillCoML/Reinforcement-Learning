import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection05: Section = {
  id: "dqn",
  title: "DQN: The Canonical Recipe",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>DQN: The Canonical Recipe</h2>
<p class="tagline"><em>Target network + experience replay + Q-learning + neural net. The 2015 Nature paper.</em></p>

<p>Deep Q-Networks (DQN), as introduced by Mnih et al. in their 2015 Nature paper, is
Q-learning with two key modifications:</p>

<ol>
  <li id="target-network"><strong>Target network.</strong> Maintain a separate copy of the
  Q-network parameters, $\\theta^-$, used only for computing the bootstrap target. The online
  parameters $\\theta$ are updated by gradient descent; the target parameters $\\theta^-$ are
  copied from $\\theta$ periodically (every $C$ updates) and otherwise frozen.</li>
  <li id="experience-replay"><strong>Experience replay.</strong> Store every observed
  transition $(s, a, r, s', \\text{done})$ in a buffer of size $N$. Each SGD update samples
  a minibatch of $B$ transitions uniformly from the buffer.</li>
</ol>

<pre class="rl-pre">DQN(α, ε, γ, C, N, B):
    initialize Q-network θ
    initialize target network θ⁻ ← θ
    initialize replay buffer D ← empty (capacity N)
    for episode = 1, 2, ...:
        s ← initial state
        while s is not terminal:
            a ← ε-greedy(Q_θ, s)
            observe r, s' from environment
            store (s, a, r, s', done) in D
            sample minibatch B from D
            for each (sᵢ, aᵢ, rᵢ, s'ᵢ, doneᵢ) in minibatch:
                targetᵢ ← rᵢ + (1 − doneᵢ) γ max_{a'} Q_{θ⁻}(s'ᵢ, a')
            θ ← θ − α ∇_θ (1/B) Σᵢ (Q_θ(sᵢ, aᵢ) − targetᵢ)²
            every C updates: θ⁻ ← θ
            s ← s'</pre>

<h3>Why target networks work</h3>

<p>The target network "freezes" the target while $\\theta^-$ is constant. SGD updates on
this fixed target converge to the best fit — exactly the tabular Q-learning situation.
Then $\\theta^-$ is updated to the current $\\theta$, and the process repeats.
This restores enough of the Bellman contraction to make convergence plausible.</p>

<h3>Why replay works</h3>

<p>Replay buffers decorrelate the data. Each minibatch samples from many different times and
episodes, making the gradient approximately i.i.d. Two additional benefits: (i)
<strong>sample efficiency</strong> — each transition is used many times; (ii)
<strong>distribution smoothing</strong> — the buffer averages over many time-varying
behaviors.</p>

<h3>DQN ablation on the $3 \\times 3$ gridworld (5 seeds, 800 episodes)</h3>

<table class="rl-table">
  <thead>
    <tr>
      <th>Configuration</th>
      <th>$Q(0,0,\\text{right})$ mean ± std</th>
      <th>true $Q^*$</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Full DQN (target + replay)</td><td>0.701 ± 0.055</td><td>0.729</td></tr>
    <tr><td>Target only, no replay</td><td>0.624 ± 0.054</td><td>0.729</td></tr>
    <tr><td>Replay only, no target</td><td>0.614 ± 0.057</td><td>0.729</td></tr>
    <tr><td>Naive (no target, no replay)</td><td>0.614 ± 0.051</td><td>0.729</td></tr>
  </tbody>
</table>

<p>Full DQN is closest to truth. All ablated versions undershoot, but naive DQN's
<em>variance</em> is higher. On a small problem like $3 \\times 3$, even naive DQN
doesn't catastrophically diverge — the function class is so expressive. On larger
problems, the gap between naive and full DQN is the difference between learning and chaos.</p>

<dqn-tricks-explained></dqn-tricks-explained>
`);
  },
};
