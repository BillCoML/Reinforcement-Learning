import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection03: Section = {
  id: "mc-policy-evaluation",
  title: "MC Policy Evaluation",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>MC Policy Evaluation</h2>
<p class="tagline"><em>Sample trajectories, accumulate returns, watch the value function fill in.</em></p>

<p>Given a fixed policy $\\pi$, the MC policy evaluation problem is to estimate
$V^\\pi(s)$ for every state $s$ using sampled trajectories. The algorithm is short:
sample $N$ trajectories under $\\pi$, compute the per-step returns, and for each
state visited update its running estimate via the first-visit (or every-visit) update
from Section 2.</p>

<pre class="algo-block"><code>MC_PolicyEvaluation(π, N, first_visit=True):
    V(s) ← 0 for all s
    N(s) ← 0 for all s
    for episode = 1, ..., N:
        sample trajectory τ ~ π
        compute returns G₀, G₁, ..., G_{T−1}
        visited ← ∅
        for t = 0, ..., T − 1:
            s ← s_t
            if first_visit and s ∈ visited: continue
            visited ← visited ∪ {s}
            N(s) ← N(s) + 1
            V(s) ← V(s) + (1/N(s)) · (G_t − V(s))
    return V</code></pre>

<p>The algorithm converges to $V^\\pi$ under mild conditions: every state must be
visited infinitely often as $N \\to \\infty$, and the rewards must have finite
variance. Under these conditions,
$\\hat V^\\pi(s) \\xrightarrow{a.s.} V^\\pi(s)$ for every state $s$.</p>

<h3>Convergence rate</h3>

<p>The per-state convergence rate is $O(1/\\sqrt{N(s)})$, where $N(s)$ is the number
of episodes that visited $s$. For states the policy visits frequently, $N(s)$ grows
roughly linearly in $N$ and the estimator converges at $O(1/\\sqrt{N})$. For states
the policy visits rarely, convergence is correspondingly slower. This non-uniform
convergence rate is one of MC's structural features — and one reason off-policy MC
(Section 5) is painful: the target policy may want to evaluate states the behavior
policy visits rarely or never.</p>

<h3>The full gridworld value function under uniform random</h3>

<p>From Lesson 5 the exact $V^\\pi(s)$ under uniform random policy is:</p>

<table class="data-table">
  <thead><tr><th></th><th>col 0</th><th>col 1</th><th>col 2</th></tr></thead>
  <tbody>
    <tr><td>row 0</td><td>−0.4205</td><td>−0.5139</td><td>−0.2386</td></tr>
    <tr><td>row 1</td><td>−0.5139</td><td>0.0000</td><td>−0.0693</td></tr>
    <tr><td>row 2</td><td>−0.2386</td><td>−0.0693</td><td>0.0000</td></tr>
  </tbody>
</table>

<p>(Where $(1,1)$ is the pit with terminal reward $-1$ and $(2,2)$ is the goal with
terminal reward $+1$; both terminal states have value $0$ by convention.)</p>

<p>Running first-visit MC for $N = 10000$ episodes produces estimates within $0.01$
of every cell. Running for $N = 100$ episodes produces estimates with per-cell
standard deviations of roughly $0.04$ to $0.10$, depending on how frequently each
cell is visited. The hardest cells to estimate are those the policy visits least
often.</p>

<div class="component-host centerpiece-host">
  <mc-estimator-lab></mc-estimator-lab>
</div>
`);
  },
};
