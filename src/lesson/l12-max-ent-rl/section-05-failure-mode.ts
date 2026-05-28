import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection05: Section = {
  id: "maxent-failure-mode",
  title: "The Failure Mode",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>The Failure Mode</h2>
<p class="tagline"><em>What happens when entropy starts to dominate.</em></p>

<p>We have established the theory. The soft Bellman operator is a contraction. The
soft-optimal policy is a Boltzmann distribution. Everything is mathematically clean.
So we run soft VI on the gridworld at $\\alpha = 0.2$ and ask: how often does this
policy reach the goal?</p>

<p><strong>Answer: 5.1%.</strong> Across 5,000 Monte Carlo rollouts from the start state,
the policy at $\\alpha = 0.2$ reaches the goal in 51 of them. The other 4,949 trajectories
are still wandering around the grid 500 steps later.</p>

<p>This is not a numerical error. It is the correct answer to the stated optimization
problem. The per-step entropy bonus at $\\alpha = 0.2$ on a 4-action problem is up to
$0.2 \\times \\log 4 \\approx 0.277$ reward-units. Over an infinite discounted horizon
the accumulated entropy from continued non-terminal motion is roughly:</p>

$$\\frac{0.277}{1 - \\gamma} = \\frac{0.277}{0.1} = 2.77 \\text{ reward-units.}$$

<p>Compare to the $+1$ terminal reward for reaching the goal. The entropy bonus from
dawdling is worth $\\approx 2.77\\times$ the terminal reward. The optimal policy
under this objective <strong>avoids the goal</strong>.</p>

<h3>The empirical table</h3>

<table class="data-table">
<thead><tr>
  <th>$\\alpha$</th><th>Goal reach</th><th>Timeout</th><th>Mean steps (terminated)</th>
</tr></thead>
<tbody>
<tr><td>0.001</td><td>100.0%</td><td>0.0%</td><td>4.0</td></tr>
<tr><td>0.01</td><td>100.0%</td><td>0.0%</td><td>4.0</td></tr>
<tr><td>0.05</td><td>100.0%</td><td>0.0%</td><td>6.2</td></tr>
<tr><td>0.10</td><td>99.9%</td><td>0.1%</td><td>79.5</td></tr>
<tr class="failure-row"><td>0.20</td><td>5.4%</td><td>94.6%</td><td>249.8</td></tr>
<tr class="failure-row"><td>0.50</td><td>0.2%</td><td>99.8%</td><td>—</td></tr>
</tbody>
</table>

<p>At $\\alpha = 0.05$, the policy reaches the goal but takes 6.2 steps on average (vs 4
for greedy) — a small efficiency cost for stochastic exploration. At $\\alpha = 0.1$
the cost is steep (79.5 steps). At $\\alpha = 0.2$ the policy fundamentally fails.</p>

<h3>The wall-bumping diagnosis</h3>

<p>Looking at the policy at $\\alpha = 0.2$ from state $(0,0)$:</p>
<ul>
<li><strong>Up</strong> (bounces wall back to $(0,0)$): 28.6%</li>
<li><strong>Right</strong> ($\\to (0,1)$): 21.4%</li>
<li><strong>Down</strong> ($\\to (1,0)$): 21.4%</li>
<li><strong>Left</strong> (bounces wall back to $(0,0)$): 28.6%</li>
</ul>

<p>The wall-bumping actions are <em>preferred</em>. The agent has learned that staying in
place earns the maximum entropy bonus over time. This is a real, well-known phenomenon
in max-ent RL on episodic tasks (Ziebart 2010; Haarnoja et al. 2017 discuss workarounds).</p>

<failure-diagnostic></failure-diagnostic>

<h3>Three responses to the failure mode</h3>

<p><strong>(a) Choose $\\alpha$ small enough.</strong> On the gridworld, $\\alpha \\leq 0.05$
keeps the policy goal-directed. But the "right" $\\alpha$ is task-dependent, and tuning
it by hand is exactly the hyperparameter sensitivity max-ent RL was supposed to reduce.</p>

<p><strong>(b) Use infinite-horizon settings.</strong> If episodes never terminate, the agent
has no terminals to avoid. Continuous-control tasks (pendulum, half-cheetah, etc.) are
typically infinite-horizon. SAC (Lesson 13) operates in this regime and the failure mode
does not appear. This is the response of choice in deep continuous control.</p>

<p><strong>(c) Replace entropy with KL-to-reference.</strong> Instead of maximizing
$\\mathcal{H}(\\pi)$, maximize $-D_{\\mathrm{KL}}(\\pi \\| \\pi_{\\mathrm{ref}})$ for a
goal-directed reference policy. Now deviating from $\\pi_{\\mathrm{ref}}$ to dawdle costs
you the KL penalty. This is the RLHF response (Lesson 17) — the reference is the
supervised fine-tuned LLM, and the agent is rewarded for being close to it while
also earning reward.</p>
`);
  },
};
