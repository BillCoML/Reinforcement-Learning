import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection03: Section = {
  id: "deadly-triad",
  title: "The Deadly Triad: When Linear FA Diverges",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>The Deadly Triad: When Linear FA Diverges</h2>
<p class="tagline"><em>Three properties, individually fine, jointly catastrophic. Baird's counterexample.</em></p>

<p>The convergence story from §2 has a critical assumption: <strong>on-policy</strong> data.
When the data distribution does <em>not</em> match the policy being evaluated, the
contraction property of $\\Pi T^\\pi$ can fail.</p>

<p>The standard formulation of when divergence occurs is the <strong>deadly triad</strong>,
three properties whose combination breaks convergence:</p>

<ol>
  <li><strong>Function approximation.</strong> $V$ or $Q$ is represented parametrically
  rather than as a table. The tabular setting is exempt by definition.</li>
  <li><strong>Bootstrapping.</strong> The update target involves the current estimate
  (the TD target $r + \\gamma V_\\theta(s')$) rather than the true return (the MC target
  $G_t$). Bootstrapping creates the feedback loop where $\\theta$ affects the target that
  updates $\\theta$.</li>
  <li><strong>Off-policy learning.</strong> The data is collected under a behavior policy
  that differs from the policy being evaluated. Q-learning's $\\max_{a'} Q(s', a')$ target
  is implicitly off-policy.</li>
</ol>

<p>The triad theorem (Sutton &amp; Barto 2018): when all three properties hold
simultaneously, semi-gradient updates can diverge. The parameter norm $\\|\\theta\\|$ can
grow without bound. Each property in isolation is fine. <strong>The combination is what
kills you.</strong></p>

<h3 id="bairds-counterexample">Baird's counterexample</h3>

<p>The canonical demonstration is a 7-state MDP from Baird (1995):</p>

<ul>
  <li>States 1–7. Two actions: <em>dashed</em> and <em>solid</em>.</li>
  <li>Action <em>dashed</em>: go to one of states 1–6 uniformly at random.</li>
  <li>Action <em>solid</em>: go to state 7 deterministically.</li>
  <li>Reward 0 everywhere. $\\gamma = 0.99$.</li>
  <li>Behavior policy: 50/50 between dashed and solid.</li>
  <li>Target policy: always solid (IS ratio = 2 for solid, 0 for dashed).</li>
</ul>

<p>The true $V^\\pi(s) = 0$ for every state. The features are 8-dimensional and chosen so
that $\\theta = \\mathbf{0}$ gives $V_\\theta(s) = 0$ for every $s$. The function class
contains the true $V^\\pi$.</p>

<p>Despite this, <strong>off-policy semi-gradient TD diverges</strong>. With $\\alpha = 0.01$
and initial $\\theta = (1, 1, 1, 1, 1, 1, 10, 1)$:</p>

<table class="rl-table">
  <thead><tr><th>Iteration</th><th>$\\|\\theta\\|$</th></tr></thead>
  <tbody>
    <tr><td>0</td><td>10.3</td></tr>
    <tr><td>100</td><td>18.1</td></tr>
    <tr><td>500</td><td>107.6</td></tr>
    <tr><td>1,000</td><td>475.4</td></tr>
    <tr><td>1,500</td><td>1,525.7</td></tr>
    <tr><td>2,000</td><td>4,582.3</td></tr>
  </tbody>
</table>

<p>The norm grows roughly <em>exponentially</em>. The true $\\theta^* = \\mathbf{0}$ is in
the function class, and the algorithm is moving <em>away</em> from it.</p>

<h3>Killing one component restores convergence</h3>

<table class="rl-table">
  <thead><tr><th>Configuration</th><th>$\\|\\theta\\|$ at iteration 2,000</th></tr></thead>
  <tbody>
    <tr><td>All three (FA + bootstrap + off-policy)</td><td>4,582</td></tr>
    <tr><td>No off-policy (behavior = target)</td><td>22 (bounded)</td></tr>
    <tr><td>No bootstrapping (MC returns)</td><td>8.6 (converges)</td></tr>
    <tr><td>No function approximation (tabular)</td><td>exactly 0</td></tr>
  </tbody>
</table>

<h3>What this means for DQN</h3>

<p>DQN is exactly the deadly triad: function approximation (neural net), bootstrapping
(TD target), off-policy ($\\varepsilon$-greedy behavior vs. greedy target via $\\max$).
By the triad theorem, DQN <em>should</em> diverge. In practice, DQN often does — the
original paper documents many failed runs. Sections 5 and 6 introduce target networks
and experience replay as engineering responses to the deadly triad.</p>

<dqn-baird-counterexample></dqn-baird-counterexample>
`);
  },
};
