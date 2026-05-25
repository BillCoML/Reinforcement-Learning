import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mdpSection05: Section = {
  id: "action-value-function",
  title: "Action Value Qπ and Advantage Aπ",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§5</span>Action Value <span class="rl-mono">Q<sup>π</sup></span> and Advantage <span class="rl-mono">A<sup>π</sup></span></h2>
<p class="tagline">V, conditioned on the first move. And how much better that move is than average.</p>

<p>Sometimes we want “the expected return if I take action $a$ <em>now</em>, then follow $\\pi$.”
That's the <strong>action value function</strong>:</p>

$$\\boxed{Q^\\pi(s, a) \\;:=\\; \\mathbb{E}_\\pi\\!\\left[G_t \\mid S_t = s,\\, A_t = a\\right]}$$

<p>The state value is the action-value averaged over the policy's choice:</p>

$$V^\\pi(s) \\;=\\; \\sum_a \\pi(a \\mid s)\\, Q^\\pi(s, a) \\;=\\; \\mathbb{E}_{a \\sim \\pi(\\cdot|s)}\\!\\left[Q^\\pi(s, a)\\right].$$

<p>Conversely, $Q^\\pi$ decomposes into immediate reward plus discounted next-state value:</p>

$$\\boxed{Q^\\pi(s, a) \\;=\\; r(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a)\\, V^\\pi(s')}$$

<p>These are baby versions of the Bellman expectation equation (§6). The second expresses
$Q^\\pi$ purely in terms of $V^\\pi$ — they're not independent.</p>

<p><strong>Why two value functions?</strong></p>
<ol>
  <li><strong>Action selection.</strong> To act greedily, $\\pi'(s) = \\arg\\max_a Q^\\pi(s, a)$, you need
  $Q^\\pi$ — or $V^\\pi$ plus the model $(P, r)$.</li>
  <li><strong>Q-learning.</strong> Samples give $(s,a,r,s')$ tuples; $Q(s,a)$ is what those tuples
  are about (Lesson 5).</li>
  <li><strong>Function approximation.</strong> $Q_\\theta(s,a)$ is one forward pass — no need to know
  $P$ or sum over $s'$. DQN (Lesson 7) exploits this.</li>
</ol>

<h3>The advantage function</h3>

$$\\boxed{A^\\pi(s, a) \\;:=\\; Q^\\pi(s, a) - V^\\pi(s)}$$

<p>How much better than average action $a$ is at $s$. Observations:</p>
<ul>
  <li>$\\sum_a \\pi(a|s) A^\\pi(s, a) = 0$ — the policy-weighted advantage is zero by
  construction.</li>
  <li>For deterministic policies, $A^\\pi(s, \\pi(s)) = 0$ and other actions are
  $\\leq 0$.</li>
  <li>The advantage is <strong>invariant to a state-dependent baseline</strong> — adding $b(s)$ to all
  $Q^\\pi(s, \\cdot)$ leaves $A^\\pi$ unchanged. This is why advantage estimators reduce variance
  in policy gradient (Lesson 8) without biasing it.</li>
</ul>

<h3>Worked example (pre-verified): Q* and A* at (1,0) under π*</h3>
<p>State $(1,0)$ is the leftmost middle cell, with $V^*((1,0)) = 0.81$.</p>

<table class="numeric">
  <thead><tr><th>action</th><th>landing in</th><th>reward</th><th>$Q^* = r + \\gamma V^*(s')$</th></tr></thead>
  <tbody>
    <tr><td>Up</td><td>(0,0)</td><td>0</td><td>$0 + 0.9\\cdot 0.729 = 0.6561$</td></tr>
    <tr><td>Right</td><td>(1,1) = pit</td><td>−1</td><td>$-1 + 0 = -1.0000$</td></tr>
    <tr><td>Down</td><td>(2,0)</td><td>0</td><td>$0 + 0.9\\cdot 0.9 = 0.8100$ ← argmax</td></tr>
    <tr><td>Left</td><td>(1,0) bounce</td><td>0</td><td>$0 + 0.9\\cdot 0.81 = 0.7290$</td></tr>
  </tbody>
</table>

<p>So $V^*((1,0)) = 0.81$. The advantages:</p>

<table class="numeric">
  <thead><tr><th>action</th><th>$A^*((1,0), a) = Q^* - V^*$</th></tr></thead>
  <tbody>
    <tr><td>Up</td><td>−0.1539</td></tr>
    <tr><td>Right</td><td>−1.8100 ← <em>very</em> bad</td></tr>
    <tr><td>Down</td><td>0.0000 ← optimal</td></tr>
    <tr><td>Left</td><td>−0.0810</td></tr>
  </tbody>
</table>

<p>$A^* = -1.81$ for “Right” is beautifully horrible: that single action costs nearly two
future-reward units — one from the immediate $-1$, plus the $0.81$ given up by abandoning
the path to the goal. The advantage exposes <em>how much each action choice matters</em>, which
is exactly what an actor-critic needs.</p>

${forwardLink({
  destination: "Lesson 8 — Policy Gradient",
  html: `<p>The policy-gradient theorem reads
  $\\nabla_\\theta J(\\theta) = \\mathbb{E}_{s \\sim d^\\pi,\\, a \\sim \\pi}[\\nabla_\\theta \\log \\pi_\\theta(a|s)\\, A^\\pi(s,a)]$.
  The advantage is exactly what multiplies the score function — the variance-optimal
  baseline.</p>`,
})}

<p>Below, each cell is split into four action quadrants. Toggle Q vs A, and hover the
<strong>Right</strong> quadrant of $(1,0)$ under the optimal policy to read off
$Q = -1.000$, $A = -1.810$.</p>

<q-quadrants-advantage></q-quadrants-advantage>`,
    );
  },
};
