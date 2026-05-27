import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const dpSection07: Section = {
  id: 'generalized-policy-iteration',
  title: 'Generalized Policy Iteration',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§7</span>Generalized Policy Iteration</h2>
<p class="tagline">V and π pull each other toward V* and π*. Every RL algorithm is some flavor of this.</p>

<p>Step back. What did we just do? Two operations:</p>

<ul>
  <li><strong>Policy evaluation</strong> moves $V$ closer to $V^\\pi$ for the current $\\pi$.</li>
  <li><strong>Policy improvement</strong> moves $\\pi$ to be more consistent with $V$ (greedy w.r.t. $V$).</li>
</ul>

<p>Each operation pulls one of $(V, \\pi)$ closer to a configuration in which they're
<em>mutually consistent</em>. The fixed point of both operations simultaneously is
$(V^*, \\pi^*)$: the optimal value function and an optimal policy that is greedy w.r.t. it.</p>

<p><strong>Generalized Policy Iteration (GPI)</strong> is the meta-pattern: alternate some amount
of evaluation with some amount of improvement, in any interleaving, until convergence.
The algorithms in this lesson are the <em>pure</em> cases:</p>

<div class="dp-table-wrap">
<table class="dp-table">
  <thead><tr><th>Algorithm</th><th>Eval</th><th>Improve</th><th>Async?</th></tr></thead>
  <tbody>
    <tr><td>Full Policy Iteration</td><td>full</td><td>full sweep</td><td>sync</td></tr>
    <tr><td>Value Iteration</td><td>1 step</td><td>full sweep</td><td>sync</td></tr>
    <tr><td>Modified PI ($m$ steps)</td><td>$m$ steps</td><td>full sweep</td><td>sync</td></tr>
    <tr><td>Gauss-Seidel VI</td><td>1 step</td><td>per-state</td><td>async, in-place</td></tr>
  </tbody>
</table>
</div>

<p>GPI doesn't prescribe the schedule. It says: <strong>as long as both processes keep
happening, both $V$ and $\\pi$ will converge</strong> to the joint fixed point.</p>

<h3>Why This View Matters</h3>

<p>Every algorithm in the rest of this curriculum will be a variant of GPI:</p>

<ul>
  <li><strong>Monte Carlo (Lesson 4):</strong> evaluation by sampling returns; improvement by
    greedy w.r.t. estimated $Q$. Sample-based evaluation; same improvement.</li>
  <li><strong>TD learning (Lesson 5):</strong> evaluation via sampled Bellman backups (TD(0));
    improvement via greedy or ε-greedy. Stochastic version with step-size $\\alpha$.</li>
  <li><strong>DQN (Lesson 7):</strong> evaluation by gradient descent on the squared TD error;
    improvement by $\\arg\\max_a Q_\\theta(s, a)$. Approximate evaluation with a neural
    net, same improvement.</li>
  <li><strong>Policy Gradient (Lesson 8):</strong> evaluation via the advantage estimator;
    improvement by gradient ascent on $\\log \\pi_\\theta \\cdot A$. Continuous improvement
    in policy <em>parameter space</em>.</li>
  <li><strong>SAC (Lesson 12):</strong> entropy-regularized evaluation; improvement via
    soft-greedy (Boltzmann). The full GPI with a temperature.</li>
</ul>

<p>The vocabulary changes (estimation, gradient, neural net, regularization) but the
structural pattern is identical: <strong>make $V$ more consistent with $\\pi$, make $\\pi$
more consistent with $V$, iterate.</strong></p>

<gpi-visualizer></gpi-visualizer>`);
  },
};
