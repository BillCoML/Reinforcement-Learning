import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mdpSection07: Section = {
  id: "bellman-optimality",
  title: "Bellman Optimality",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§7</span>Bellman Optimality</h2>
<p class="tagline">Replace the sum-over-π with a max. Get V*. Greedy w.r.t. V* is optimal.</p>

<p>$V^\\pi$ is policy-specific. The <strong>optimal value function</strong> $V^*$ is policy-free —
the best you could possibly do from each state:</p>

$$V^*(s) \\;:=\\; \\max_\\pi V^\\pi(s), \\qquad Q^*(s,a) := \\max_\\pi Q^\\pi(s,a).$$

<p>A non-obvious fact: $V^*$ and $Q^*$ satisfy their own <strong>Bellman optimality equations</strong>:</p>

$$\\boxed{V^*(s) \\;=\\; \\max_a \\!\\left[\\, r(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a)\\, V^*(s') \\,\\right]}$$

$$\\boxed{Q^*(s, a) \\;=\\; r(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a) \\max_{a'} Q^*(s', a')}$$

<p>The only change from the expectation equation is the $\\max$ in place of
$\\sum_a \\pi(a|\\cdot)$: under the optimal policy you take the best action, not the
policy-weighted average. These equations are <em>nonlinear</em> (the max), so we can't solve
them with $(I - \\gamma P)^{-1} R$. But the <strong>optimality operator</strong>
$T^* V := \\max_a [r(\\cdot, a) + \\gamma P(\\cdot|\\cdot, a) V]$ is also a γ-contraction, so
iterating it converges to $V^*$. That's <em>value iteration</em>, Lesson 3.</p>

<h3 id="optimal-policy-exists-deterministic">The fundamental theorem of MDPs</h3>
<ol>
  <li><strong>Existence.</strong> For finite MDPs with bounded rewards and $\\gamma \\in [0,1)$, $V^*$
  exists, is finite, and is the unique fixed point of $T^*$.</li>
  <li><strong>A deterministic optimal policy exists.</strong> Define
  $\\pi^*(s) \\in \\arg\\max_a [\\, r(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^*(s') \\,]$. Then
  $V^{\\pi^*} = V^*$ — this deterministic policy is optimal.</li>
  <li><strong>Greedy is optimal w.r.t. $V^*$.</strong> Once you have $V^*$, the optimal policy is
  <em>one-step greedy</em>: pick the action maximizing immediate reward plus discounted
  next-state value. No long-term planning required.</li>
</ol>

<p>This last point is striking: once you've absorbed all future structure into $V^*$, every
decision becomes a <em>myopic, immediate-reward-plus-value maximization</em>. This is the deep
reason value functions are useful — they <strong>replace planning with arithmetic</strong>.</p>

${forwardLink({
  destination: "Lessons 5 & 7 — Q-learning / DQN",
  html: `<p>Q-learning and DQN learn $Q^*$ directly from samples. Once $Q^*$ is learned,
  $\\pi^*(s) = \\arg\\max_a Q^*(s,a)$ needs no model $(P,r)$ — no planning, no sampling. That's
  what makes DQN “model-free”.</p>`,
})}

<h3>Worked example: V* for the gridworld</h3>

<table class="numeric">
  <thead><tr><th></th><th>col 0</th><th>col 1</th><th>col 2</th></tr></thead>
  <tbody>
    <tr><td><strong>row 0</strong></td><td>0.729</td><td>0.810</td><td>0.900</td></tr>
    <tr><td><strong>row 1</strong></td><td>0.810</td><td>0.000 (pit)</td><td>1.000</td></tr>
    <tr><td><strong>row 2</strong></td><td>0.900</td><td>1.000 (goal)</td><td>0.000</td></tr>
  </tbody>
</table>

<ul>
  <li>$(0,0) \\to 0.729 = \\gamma^3$: three optimal steps from the goal, discounted reward
  $\\gamma^3 \\cdot 1$.</li>
  <li>$(2,1) \\to 1.0$: one step from the goal — the $+1$ is collected on entry.</li>
  <li>$(0,1) \\to 0.81 = \\gamma^2$: two steps under the route through $(0,2)\\to(1,2)\\to(2,2)$.</li>
  <li>Pit and goal are terminal, $V = 0$.</li>
</ul>

<p>The optimal policy routes around the pit. Crucially, both
$(0,0)\\to(0,1)\\to(0,2)\\to(1,2)\\to(2,2)$ and $(0,0)\\to(1,0)\\to(2,0)\\to(2,1)\\to(2,2)$ achieve
$\\gamma^3$ — so <strong>(0,0) has two optimal actions</strong> (Right and Down), and any tie-broken
$\\arg\\max$ is fine. The badge in the explorer below flags exactly these multi-optimal cells.</p>

<p>Toggle to “regret” to see $V^* - V^\\pi$ — how much each policy leaves on the table — and
hover any cell to fill in its Bellman optimality equation.</p>

<optimality-explorer></optimality-explorer>`,
    );
  },
};
