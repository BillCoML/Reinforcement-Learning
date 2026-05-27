import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const dpSection03: Section = {
  id: 'policy-improvement-theorem',
  title: 'Policy Improvement',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§3</span>Policy Improvement</h2>
<p class="tagline">Given V<sup>π</sup>, the greedy policy is at least as good as π.</p>

<p>Suppose we have $V^\\pi$ (computed via §2). Can we use it to find a
<em>better</em> policy? Yes, and the construction is mechanical.</p>

<p><strong>The greedy improvement step.</strong> Define a new policy $\\pi'$ by</p>

$$\\boxed{\\pi'(s) \\in \\arg\\max_a Q^\\pi(s, a) = \\arg\\max_a \\left[ r(s,a) + \\gamma \\sum_{s'} P(s'|s,a)\\, V^\\pi(s') \\right].}$$

<p>$\\pi'$ is <strong>greedy with respect to $V^\\pi$</strong>. It looks one step ahead and picks
whichever action maximizes immediate-plus-discounted-future value under the
<em>current</em> value estimate.</p>

<h3>Policy Improvement Theorem</h3>

<p>For deterministic $\\pi'$ defined as above, $V^{\\pi'}(s) \\geq V^\\pi(s)$ for every state $s$.</p>

<p><em>Proof.</em> Start from the inequality $Q^\\pi(s, \\pi'(s)) \\geq V^\\pi(s)$, which holds
because $\\pi'(s)$ is the argmax. Expand:</p>

$$V^\\pi(s) \\leq r(s, \\pi'(s)) + \\gamma \\sum_{s'} P(s'|s, \\pi'(s))\\, V^\\pi(s').$$

<p>Apply this same inequality to each $V^\\pi(s')$ on the right. Unrolling indefinitely:</p>

$$\\begin{aligned}
V^\\pi(s) &\\leq r(s, \\pi'(s)) + \\gamma \\sum_{s'} P(s'|s, \\pi'(s)) \\left[ r(s', \\pi'(s')) + \\cdots \\right] \\\\
&= \\mathbb{E}_{\\pi'} \\left[ R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\cdots \\mid S_t = s \\right] \\\\
&= V^{\\pi'}(s). \\qquad \\blacksquare
\\end{aligned}$$

<p>The proof is a telescoping expansion of one inequality — the workhorse of every
"monotonic improvement" result in RL, including TRPO's (Lesson 9).</p>

<hr>

<h3>A Small Example: Greedy Improvement on Uniform Random</h3>

<p>Take $V^{\\pi_0}$ (the uniform random policy's value, computed in §2) and apply one
greedy improvement step. Selected states:</p>

<div class="dp-table-wrap">
<table class="dp-table">
  <thead><tr><th>State</th><th>$V^{\\pi_0}$</th><th>$\\arg\\max_a Q^{\\pi_0}(s,a)$</th><th>Reason</th></tr></thead>
  <tbody>
    <tr><td>(0,0)</td><td>−0.4205</td><td><strong>Up</strong> or <strong>Left</strong></td><td>Both bounce off wall; staying "here" beats moving toward negative neighbors</td></tr>
    <tr><td>(0,1)</td><td>−0.5139</td><td>Right</td><td>Leads toward (0,2), least negative V</td></tr>
    <tr><td>(1,2)</td><td>−0.0693</td><td>Down</td><td>Leads to goal! $r=+1$</td></tr>
    <tr><td>(2,1)</td><td>−0.0693</td><td>Right</td><td>Into goal</td></tr>
  </tbody>
</table>
</div>

<p>Note the <strong>counter-intuitive</strong> behaviour at state (0,0): greedy improvement says
"stay put by bouncing off the wall." Why? Because every other action leads to a cell
with <em>more negative</em> value under the <em>uniform-random</em> $V^{\\pi_0}$. The new policy
isn't optimal — it's just better than uniform random.</p>

<p>The bounce-off-wall behaviour will be fixed in the next round of policy improvement.
This phenomenon — greedy improvement producing a <em>non-optimal but better</em> policy —
is exactly why we need <em>iterated</em> policy improvement. One step isn't enough.</p>

<policy-improvement-inspector></policy-improvement-inspector>`);
  },
};
