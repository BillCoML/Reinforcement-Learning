import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection04: Section = {
  id: "mc-control-exploring-starts",
  title: "MC Control via Generalized Policy Iteration",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>MC Control via Generalized Policy Iteration</h2>
<p class="tagline"><em>Estimate Q, improve the policy, repeat. Exploring Starts and ε-greedy.</em></p>

<p>Policy evaluation by itself is useful, but the larger goal is <strong>control</strong>: find a
good (ideally optimal) policy. The dynamic programming track in Lesson 5 reached
this goal via policy iteration, alternating between policy evaluation and policy
improvement. The generalized policy iteration (GPI) picture emphasized that this
alternation does not need to be exact at either step: we can do a <em>partial</em>
policy evaluation followed by a <em>greedy</em> policy improvement and still converge.</p>

<p>MC control instantiates GPI with the policy evaluation step replaced by Monte Carlo
sampling. The two algorithms differ in how the evaluation step is structured and what
policy improvement step is applied.</p>

<h3>MC Control with Exploring Starts</h3>

<p>The simplest version of MC control makes a strong assumption: every (state, action)
pair is chosen as the initial state-action of some episode with positive probability.
This is the <strong>exploring starts</strong> assumption. Under it, the algorithm is:</p>

<pre class="algo-block"><code>MC_ES():
    Q(s, a) ← 0 for all s, a
    π(s) ← arbitrary deterministic policy
    for episode = 1, ..., N:
        s₀, a₀ ← uniformly random from valid (s, a)
        roll out one episode from (s₀, a₀) following π
        compute returns G₀, G₁, ..., G_{T−1}
        for each first visit to (s, a) in episode:
            update Q(s, a) by running-mean rule
            π(s) ← argmax_a Q(s, a)
    return π, Q</code></pre>

<p>Under the exploring-starts assumption, MC ES converges to the optimal deterministic
policy $\\pi^*$ and its action-value function $Q^*$. Empirically on the running
gridworld with 50,000 episodes, MC ES recovers $Q^*$ to within $0.01$ of every
entry and produces the correct optimal policy.</p>

<h3>The exploring-starts problem</h3>

<p>In practice we cannot choose initial states arbitrarily. The agent starts where
the environment starts. The exploring-starts assumption is mathematically convenient
but operationally unrealistic. We need a way to ensure that every (state, action)
gets visited often enough without controlling the initial condition.</p>

<p>The standard workaround is to make the policy itself <strong>soft</strong>: always assign
positive probability to every action. The simplest soft policy is
<strong>ε-greedy</strong>: with probability $1 - \\epsilon$, pick the greedy action
$\\arg\\max_a Q(s, a)$; with probability $\\epsilon$, pick uniformly among all
actions.</p>

<h3>MC Control with ε-greedy</h3>

<pre class="algo-block"><code>MC_ε_greedy():
    Q(s, a) ← 0 for all s, a
    for episode = 1, ..., N:
        roll out one episode from s₀ = env_start following ε-greedy(Q)
        compute returns G₀, G₁, ..., G_{T−1}
        for each first visit to (s, a) in episode:
            update Q(s, a) by running-mean rule
    return ε-greedy(Q), Q</code></pre>

<p>This is the workhorse MC control algorithm. It converges, but <strong>not to $Q^*$</strong>.
With a fixed $\\epsilon$, ε-greedy MC converges to the Q-function of the ε-soft
optimal policy — the best policy <em>among ε-soft policies</em>, which is necessarily
worse than the true optimal deterministic policy whenever $\\epsilon > 0$.</p>

<p>Empirically on the running gridworld with $\\epsilon = 0.1$ and 20,000 episodes,
the estimated $Q(0, 0)$ is:</p>

<table class="data-table">
  <thead><tr><th></th><th>up</th><th>right</th><th>down</th><th>left</th></tr></thead>
  <tbody>
    <tr><td>$Q^*$</td><td>0.6561</td><td>0.7290</td><td>0.7290</td><td>0.6561</td></tr>
    <tr><td>$Q^{\\epsilon=0.1}$</td><td>0.5646</td><td>0.6307</td><td>0.6307</td><td>0.5646</td></tr>
    <tr><td>MC ε-greedy</td><td>~0.49</td><td>~0.62</td><td>~0.53</td><td>~0.51</td></tr>
  </tbody>
</table>

<p>The MC ε-greedy estimates converge toward the ε-soft optimal Q-function (middle
row), <em>not</em> toward $Q^*$ (top row). The gap is real and persistent.</p>

<h3>The GLIE schedule and recovery of $Q^*$</h3>

<p>To recover the true optimal policy, we need <strong>Greedy in the Limit with Infinite
Exploration (GLIE)</strong>: a schedule in which $\\epsilon_n \\to 0$ as $n \\to \\infty$,
but slowly enough that every (state, action) is still visited infinitely often. A
common choice is $\\epsilon_n = 1/\\sqrt{n}$. Under GLIE, MC control converges to
$Q^*$ and the deterministic optimal policy.</p>

<div class="crosslink-callout">
  <strong>Forward link · Bandits.</strong>
  GLIE is the multi-state analog of the decaying-$\\epsilon$ schedule from the Bandits
  lesson (Lesson 1). In bandits the decay was justified by the regret bound; here it
  is justified by the convergence-to-optimal argument. The intuition is the same:
  enough exploration to know the world, enough exploitation to act on that knowledge.
</div>

<div class="component-host">
  <mc-control-learning-curve></mc-control-learning-curve>
</div>
`);
  },
};
