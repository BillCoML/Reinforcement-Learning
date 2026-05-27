import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection05: Section = {
  id: "off-policy-mc",
  title: "Off-Policy Monte Carlo via Importance Sampling",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>Off-Policy Monte Carlo via Importance Sampling</h2>
<p class="tagline"><em>Estimate one policy's value using another policy's trajectories. Variance returns to haunt us.</em></p>

<p>The MC methods of Sections 2 through 4 are <strong>on-policy</strong>: the policy being
evaluated (or improved) is the same as the policy that generates the trajectories.
In the off-policy setting we have trajectories sampled under a <strong>behavior policy</strong>
$\\pi_b$ and we want to estimate $V^{\\pi_t}$ for a different <strong>target policy</strong>
$\\pi_t$. Lesson 6 built the trajectory importance sampling apparatus that handles
this problem. We now apply it directly.</p>

<h3>The off-policy MC estimator</h3>

<p>Recall from Lesson 6 that for a trajectory $\\tau = (s_0, a_0, r_1, s_1, \\ldots, s_T)$
sampled under $\\pi_b$, the trajectory importance ratio is</p>

$$\\rho_{0:T-1} \\;=\\; \\prod_{t=0}^{T-1} \\frac{\\pi_t(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}$$

<p>and the identity is $V^{\\pi_t}(s_0) = \\mathbb{E}_{\\pi_b}[\\rho_{0:T-1} \\cdot G_0]$.
The ordinary IS estimator after $N$ trajectories is</p>

$$\\hat V^{\\pi_t}_{\\text{ord}}(s_0) \\;=\\; \\frac{1}{N} \\sum_{i=1}^N \\rho_{0:T-1}^{(i)} \\cdot G_0^{(i)}$$

<p>and the weighted IS estimator is</p>

$$\\hat V^{\\pi_t}_{\\text{wt}}(s_0) \\;=\\; \\frac{\\sum_{i=1}^N \\rho_{0:T-1}^{(i)} \\cdot G_0^{(i)}}{\\sum_{i=1}^N \\rho_{0:T-1}^{(i)}}.$$

<h3>The gridworld numerics, revisited</h3>

<p>From Lesson 6: behavior = uniform random, target = deterministic optimal,
$V^{\\pi_t}(0, 0) = 0.729$. The probability a uniform-random trajectory matches the
deterministic optimal sequence step-for-step is $(1/4)^4 = 1/256 \\approx 0.0039$.
Matching trajectories have weight exactly $4^4 = 256$.</p>

<p>The off-policy MC estimator's empirical behavior across 50 trials:</p>

<table class="data-table">
  <thead>
    <tr>
      <th>$N$</th><th>Non-zero / N</th>
      <th>Ordinary IS (mean ± SD)</th><th>Weighted IS (mean ± SD)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>100</td><td>0.6 / 100</td><td>0.86 ± 1.14</td><td>0.32 ± 0.37</td></tr>
    <tr><td>1,000</td><td>4.2 / 1000</td><td>0.79 ± 0.35</td><td>0.71 ± 0.10</td></tr>
    <tr><td>10,000</td><td>39.7 / 10000</td><td>0.73 ± 0.13</td><td><strong>0.7290 ± 0.0000</strong></td></tr>
  </tbody>
</table>

<p>At $N = 10000$, weighted IS converges to exactly $0.7290$ with zero sample
variance across all 50 trials, because all matching trajectories are identical
(same path, same return, same weight). At small $N$, weighted IS has significant
bias because the few matching trajectories dominate a denominator that has not yet
stabilized.</p>

<h3>Effective sample size as a diagnostic</h3>

<p>From Lesson 6, the effective sample size is</p>

$$N_{\\text{eff}} \\;=\\; \\frac{(\\sum_i \\rho_i)^2}{\\sum_i \\rho_i^2}.$$

<p>For the uniform-to-optimal gridworld at $N = 10000$, every non-zero weight equals
$256$, so $N_{\\text{eff}} = 39$, exactly the number of non-zero trajectories.
Ninety-nine point six percent of the sampling budget is wasted. This is the
off-policy MC variance problem in numerical form.</p>

<h3>Off-policy MC control</h3>

<p>In principle one can do MC control entirely off-policy: sample under any fixed
behavior policy with sufficient coverage, and use weighted IS to estimate $Q(s, a)$
for any target policy of interest. In practice this is rarely a winning strategy. The
trajectory weight variance compounds; the ESS is tiny; the algorithm needs absurd
amounts of data. Off-policy methods come into their own only when we move from MC to
TD (Lesson 8), where one-step backups replace trajectory weights with single
per-step ratios. The variance goes from "exponential in horizon" to "bounded by
the ratio range."</p>

<div class="crosslink-callout">
  <strong>Forward link · Q-learning.</strong>
  Q-learning sidesteps the off-policy ratio entirely by re-deriving the off-policy
  target as a max over actions: instead of $\\rho \\cdot G$, it uses
  $r + \\gamma \\max_{a'} Q(s', a')$, which has no IS factor at all. This is one of the
  most consequential tricks in RL and is the central content of Lesson 8.
</div>

<div class="component-host">
  <off-policy-vs-on-policy></off-policy-vs-on-policy>
</div>
`);
  },
};
