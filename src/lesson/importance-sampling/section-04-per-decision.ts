import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const isSection04: Section = {
  id: 'per-decision-is',
  title: 'Per-Decision Importance Sampling',
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>Per-Decision Importance Sampling</h2>
<p class="tagline">Each reward only needs the ratios up to that point.</p>

<p>The trajectory IS estimator weights the entire return
$G_0 = r_1 + \\gamma r_2 + \\gamma^2 r_3 + \\cdots$ by the full trajectory weight
$\\rho_{0:T-1}$. But the reward $r_{t+1}$ was determined by the state-action pair
$(s_t, a_t)$ and the random transition to $s_{t+1}$. It does not depend on the later
actions $a_{t+1}, a_{t+2}, \\ldots, a_{T-1}$. So multiplying $r_{t+1}$ by
$\\rho_{t+1:T-1}$ (the ratios from steps <em>after</em> $t$) is adding noise without
adding signal.</p>

<p>The <strong>per-decision importance sampling estimator</strong> weights each reward
by only the ratios up to its own time step:</p>

$$\\boxed{\\hat V^{\\pi_t}(s_0) \\;=\\; \\frac{1}{N} \\sum_{i=1}^N \\sum_{t=0}^{T_i-1}
\\gamma^t \\, r_{t+1}^{(i)} \\, \\rho_{0:t}^{(i)}.}$$

<p>This estimator is still unbiased, but the variance is strictly lower or equal
because the unnecessary noise from future ratios has been removed.</p>

<h3>Why doesn't per-decision IS help in our running gridworld example?</h3>

<p>Because the only non-zero reward arrives at the terminal step. At that step, the
per-decision weight $\\rho_{0:T-1}$ equals the full trajectory weight, so
per-decision IS collapses to trajectory IS exactly. The standard deviations are
identical:</p>

<div class="table-wrap">
<table class="spec-table">
  <thead>
    <tr>
      <th>$N$</th>
      <th>Ordinary IS SD</th>
      <th>Per-decision IS SD</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1,000</td><td>0.380</td><td>0.380 (identical)</td></tr>
    <tr><td>10,000</td><td>0.129</td><td>0.129 (identical)</td></tr>
  </tbody>
</table>
</div>

<p>For environments with <strong>dense rewards</strong> (rewards at every step),
per-decision IS gives substantial variance reduction. The reduction grows with
the horizon — in horizon-100 settings, ten to a hundred times variance reduction
is typical. Lesson 8 will show this on a dense-reward problem; here we introduce
the mechanism for later use.</p>

<div class="callout callout--forward">
  <strong>Forward link</strong> — Per-decision IS is the basis of truncated
  importance sampling estimators and <strong>V-trace</strong> (Espeholt et al. 2018),
  the algorithm used by IMPALA. V-trace clips per-step ratios at runtime to control
  variance and is the backbone of large-scale distributed deep RL. We will meet
  it in Lesson 10's discussion of distributed actor-critic methods.
</div>

<div class="component-host">
  <per-decision-is-stepwise></per-decision-is-stepwise>
</div>
`);
  },
};
