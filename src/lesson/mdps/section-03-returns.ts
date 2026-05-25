import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mdpSection03: Section = {
  id: "return-discount",
  title: "Returns and Discounting",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§3</span>Returns and Discounting</h2>
<p class="tagline">Adding up rewards over time, with a thumb on the scale.</p>

<p>The agent collects rewards $R_1, R_2, R_3, \\ldots$ over time. To talk about
“performance” we aggregate them into a single number. The standard choice is the
<strong>discounted return</strong> starting from time $t$:</p>

$$\\boxed{G_t \\;:=\\; R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\cdots \\;=\\; \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}}$$

<p>with $\\gamma \\in [0, 1]$. (Rewards are indexed with $t+1$ to emphasize they're received
<em>after</em> the action at time $t$; the convention doesn't matter as long as you're
consistent.)</p>

<p><strong>Why discount?</strong> Five answers, each illuminating.</p>
<ol>
  <li><strong>Mathematical convergence.</strong> For continuing tasks, undiscounted returns can be
  infinite, so every policy has the same return and no comparison is possible. $\\gamma < 1$
  guarantees finite returns whenever rewards are bounded.</li>
  <li><strong>Computational stability.</strong> The Bellman operator (§6) is a contraction with
  modulus $\\gamma$. Iterating the backup converges geometrically; it diverges if $\\gamma = 1$
  and the chain isn't absorbing.</li>
  <li><strong>Economic interpretation.</strong> A reward 10 steps from now is worth $\\gamma^{10}$ of a
  reward now — a standard discount; in RL it captures “preference for sooner”.</li>
  <li><strong>Uncertainty about the future.</strong> With $\\gamma = 0.99$ you behave as if the world
  has a $1\\%$ chance of ending each step. Discounting bakes in some pessimism about the
  horizon.</li>
  <li><strong>Modeling continuing tasks as episodic.</strong> A continuing task with discount
  $\\gamma$ has the same expected return as an episodic task that ends randomly at each step
  with probability $1 - \\gamma$.</li>
</ol>

<p>The case $\\gamma = 1$ only makes sense for <strong>episodic tasks</strong> where every trajectory
reaches a terminal state in finite time with probability 1. Our gridworld is episodic, so
we <em>could</em> use $\\gamma = 1$; we use $\\gamma = 0.9$ because it gives us numerical room to
distinguish “close to goal” from “far from goal”.</p>

<h3>Numerical example (pre-verified)</h3>
<p>Consider a trajectory from start to goal in four steps:
$(0,0) \\to (1,0) \\to (2,0) \\to (2,1) \\to (2,2)\\,[\\text{goal}]$, with
$R_1 = R_2 = R_3 = 0$, $R_4 = +1$. The return at time 0 is</p>

$$G_0 \\;=\\; \\gamma^0\\cdot 0 + \\gamma^1\\cdot 0 + \\gamma^2\\cdot 0 + \\gamma^3\\cdot 1 \\;=\\; \\gamma^3.$$

<table class="numeric">
  <thead><tr><th>$\\gamma$</th><th>$G_0 = \\gamma^3$</th></tr></thead>
  <tbody>
    <tr><td>0.0</td><td class="rl-mono">0.0000</td></tr>
    <tr><td>0.5</td><td class="rl-mono">0.1250</td></tr>
    <tr><td>0.9</td><td class="rl-mono">0.7290</td></tr>
    <tr><td>0.95</td><td class="rl-mono">0.8574</td></tr>
    <tr><td>0.99</td><td class="rl-mono">0.9703</td></tr>
    <tr><td>1.0</td><td class="rl-mono">1.0000</td></tr>
  </tbody>
</table>

<p>At $\\gamma = 0$ the agent is purely myopic; at $\\gamma = 1$ all future rewards are equally
weighted. At $\\gamma = 0.9$, a reward 4 steps away is worth 73% of an immediate reward —
strong but not extreme preference for sooner.</p>

${forwardLink({
  destination: "Lesson 10 — Max-Entropy RL",
  html: `<p>There we'll <em>augment</em> the reward with the policy's entropy,
  $\\tilde{r}(s,a) = r(s,a) + \\alpha\\,\\mathcal{H}[\\pi(\\cdot|s)]$. The discounted augmented return
  becomes the standard max-ent objective — the discounting machinery doesn't change, only
  the reward definition.</p>`,
})}

<p>Sample a trajectory below and watch the discounted contributions $\\gamma^k R_{k+1}$ stack
up to $G_0$. Drag $\\gamma$ to rescale the bars in real time.</p>

<return-composer></return-composer>`,
    );
  },
};
