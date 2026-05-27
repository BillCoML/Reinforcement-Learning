import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection03: Section = {
  id: "sarsa",
  title: "SARSA: On-Policy TD Control",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>SARSA: On-Policy TD Control</h2>
<p class="tagline"><em>Bootstrap with the action the policy will actually take next.</em></p>

<p>We move from prediction (estimate $V^\\pi$ for a fixed $\\pi$) to control (find a good
$\\pi$). TD control has the same structure as MC control from Lesson 7, with one extra
twist: we get an additional, <em>off-policy</em> algorithm (Q-learning, Section 4) that no
MC method easily replicates.</p>

<p><strong>SARSA</strong> is the on-policy TD control algorithm. Its name comes from the quintuple
$(S, A, R, S', A')$ of data used in each update: the agent observes state $s$, takes
action $a$, observes reward $r$ and next state $s'$, then samples the
<strong>next action</strong> $a'$ from its current $\\varepsilon$-greedy policy. The update rule is</p>

$$Q(s, a) \\;\\leftarrow\\; Q(s, a) + \\alpha \\left[ r + \\gamma Q(s', a') - Q(s, a) \\right].$$

<p>The bootstrap target uses $Q(s', a')$ — the value of the action the policy
<em>will actually take</em> next. This makes SARSA <strong>on-policy</strong>: the target reflects
the policy's actual behavior.</p>

<pre><code>SARSA:
    Q(s, a) ← 0 for all s, a
    for episode = 1, ..., N:
        observe s;  a ~ ε-greedy(Q, s)
        until terminal:
            take action a; observe r, s'
            a' ~ ε-greedy(Q, s')          ← sampled once, used for target AND next step
            Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]
            s, a ← s', a'</code></pre>

<h3>SARSA converges to $Q^\\pi_{\\varepsilon\\text{-soft}}$, not $Q^*$</h3>

<p>Because SARSA bootstraps from the actually-sampled next action $a'$, which is
$\\varepsilon$-greedy, its target reflects the exploration noise that $\\varepsilon$
introduces. The fixed point of the SARSA update is the action-value function of the
$\\varepsilon$-soft optimal policy. With $\\varepsilon = 0.1$ on the running gridworld:</p>

$$Q^\\pi_{\\varepsilon\\text{-soft}}(0, 0, \\text{right}) \\;=\\; 0.6307,$$

<p>which is strictly less than $Q^*(0, 0, \\text{right}) = 0.7290$. The gap is the cost
of exploration. The value $V^{\\pi_{\\varepsilon\\text{-soft}}}(0,0) \\approx 0.6274$ accounts
for the expected mix of greedy and exploratory actions.</p>

<div class="callout callout--warning">
  <strong>The critical implementation detail.</strong>
  $a'$ must be sampled <em>exactly once</em> and used both for the target computation
  <em>and</em> as the next step's action. Sampling $a'$ a second time for the next step is a
  common bug that silently converts SARSA into Q-learning. The test is: at $N = 10{,}000$
  with $\\varepsilon = \\alpha = 0.1$, $Q(0,0,\\text{right})$ should be in
  $[0.45, 0.65]$ — not near $0.7290$.
</div>

<h3>GLIE for SARSA</h3>

<p>To recover $Q^*$ from SARSA, anneal $\\varepsilon$ to zero slowly enough that every
state-action pair is still visited infinitely often. The standard choice is
$\\varepsilon_n = 1/n^c$ for some $c \\in (0, 1)$. Under GLIE, SARSA converges to
$Q^*$ and the deterministic optimal policy — the direct TD analog of the MC GLIE
result from Lesson 7.</p>

<div class="crosslink-callout">
  <strong>Connection to Lesson 7.</strong>
  The $Q^\\pi_{\\varepsilon\\text{-soft}}$ convergence target of SARSA is identical to the
  target of MC $\\varepsilon$-greedy control. They are two algorithms for the same fixed
  point. The anchor value $Q^\\pi_{\\varepsilon\\text{-soft}}(0,0,\\text{right}) = 0.6307$
  appears in both lessons.
</div>

<div class="component-host">
  <td-algorithm-lab></td-algorithm-lab>
</div>
`);
  },
};
