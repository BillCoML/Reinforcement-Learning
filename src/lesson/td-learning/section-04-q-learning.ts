import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection04: Section = {
  id: "q-learning",
  title: "Q-Learning: Off-Policy TD Control",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>Q-Learning: Off-Policy TD Control</h2>
<p class="tagline"><em>Bootstrap with the action the optimal policy would take. Sidestep importance sampling entirely.</em></p>

<p><strong>Q-learning</strong> changes exactly one line of SARSA. Instead of bootstrapping with
$Q(s', a')$ — where $a'$ is the action the <em>behavior</em> policy will take — Q-learning
bootstraps with $\\max_{a'} Q(s', a')$, the value of the action the
<em>optimal</em> policy would take. The update rule:</p>

$$Q(s, a) \\;\\leftarrow\\; Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right].$$

<pre><code>Q-learning:
    Q(s, a) ← 0 for all s, a
    for episode = 1, ..., N:
        observe s
        until terminal:
            a ~ ε-greedy(Q, s)
            take action a; observe r, s'
            Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
            s ← s'</code></pre>

<h3>Q-learning converges to $Q^*$, regardless of behavior policy</h3>

<p>This is one of the most consequential results in RL. As long as the behavior policy
visits every $(s, a)$ pair infinitely often, and the Robbins-Monro step-size conditions
hold, Q-learning's $Q$ converges almost surely to $Q^*$. Empirically with
$\\varepsilon = \\alpha = 0.1$ at $N = 10{,}000$:</p>

<table class="rl-table">
<thead><tr><th>Action</th><th>Q-learning estimate</th><th>$Q^*$ target</th><th>$Q^\\pi_{\\varepsilon\\text{-soft}}$ (ref)</th></tr></thead>
<tbody>
<tr><td>up</td><td><strong>0.6561</strong></td><td>0.6561</td><td>0.5646</td></tr>
<tr><td>right</td><td><strong>0.7290</strong></td><td>0.7290</td><td>0.6307</td></tr>
<tr><td>down</td><td>0.5905</td><td>0.7290</td><td>0.6307</td></tr>
<tr><td>left</td><td><strong>0.6561</strong></td><td>0.6561</td><td>0.5646</td></tr>
</tbody>
</table>

<h3>The off-policy / IS-bypass moment</h3>

<p>Lesson 6 developed the trajectory importance sampling apparatus for off-policy
correction: multiply each return by the trajectory weight $\\rho_{0:T-1}$. Lesson 7
cashed this apparatus in for off-policy MC, with the consequence that variance grows
exponentially in the horizon — at $N = 10{,}000$, only 39 of 10,000 trajectories
contributed to the estimate; 99.6% of the sampling budget was wasted.</p>

<p>Q-learning <strong>does not use importance sampling at all</strong>. The off-policy correction
comes from a different source: the $\\max$ operator. The Bellman optimality equation
$Q^*(s, a) = \\mathbb{E}[R + \\gamma \\max_{a'} Q^*(s', a')]$ involves a max over actions,
not an expectation over the policy. Once you have $Q^*$, the optimal policy is
$\\arg\\max_a Q^*(s, a)$ — but you don't need the optimal policy's action distribution to
<em>learn</em> $Q^*$; you only need to observe enough $(s, a, r, s')$ transitions to bootstrap.</p>

<div class="callout callout--info">
  <strong>The structural insight.</strong>
  Q-learning works as an off-policy algorithm because the Bellman <em>optimality</em> equation
  can be expressed without reference to any specific policy: it is a property of the MDP
  itself, encoded as a max over actions. The IS apparatus from Lesson 6 is bypassed
  completely. This is valid specifically for the optimal-value problem; any
  expectation-based off-policy method must handle the policy mismatch via IS or operator
  tricks.
</div>

<h3>Maximization bias</h3>

<p>Q-learning has a structural weakness: $\\max_{a'} Q(s', a')$ uses noisy estimates inside
a maximum, and the expected value of the max of noisy estimates is greater than the max
of expected values. This systematic over-estimation is called <strong>maximization bias</strong>.</p>

<p>Consider the canonical Sutton-Barto fig. 6.5 setup. State A is the start; the agent
has two actions: "right" goes to terminal with reward 0; "left" goes to state B. From
B, there are 10 actions, each leading to terminal with reward
$\\mathcal{N}(-0.1, 1)$. The optimal action from A is <strong>right</strong> (expected reward 0 > −0.1).
Q-learning nonetheless chooses "left" from A about 35% of the time due to overestimating
$\\max_{a'} Q(B, a')$. We demonstrate this in V7.</p>

<div class="component-host">
  <sarsa-vs-qlearning></sarsa-vs-qlearning>
</div>
`);
  },
};
