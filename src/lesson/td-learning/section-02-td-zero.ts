import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection02: Section = {
  id: "td-zero-prediction",
  title: "TD(0) Prediction and Robbins-Monro",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>TD(0) Prediction and Robbins-Monro</h2>
<p class="tagline"><em>The simplest bootstrap, and the stochastic-approximation theory that makes it converge.</em></p>

<p>The simplest TD algorithm is <strong>TD(0)</strong>, also called one-step TD: at every step,
apply the TD update with $\\alpha$ as a step size and the one-step bootstrap target.</p>

<pre><code>TD(0) prediction:
    V(s) ← 0 for all s
    for episode = 1, ..., N:
        observe s
        until terminal:
            take action a ~ π(·|s)
            observe r, s'
            V(s) ← V(s) + α [r + γ V(s') - V(s)]
            s ← s'</code></pre>

<p>The algorithm is online: each $(s, a, r, s')$ triggers exactly one update. There is no
episode buffer, no batch processing, no episode boundary required for correctness.</p>

<h3>Convergence: TD(0) is a Robbins-Monro stochastic approximation</h3>

<p>The Robbins-Monro framework treats stochastic algorithms that update an estimate
$\\theta_n$ toward a noisy observation of a target:</p>

$$\\theta_{n+1} \\;=\\; \\theta_n + \\alpha_n (Y_n - \\theta_n),$$

<p>where $Y_n$ is a noisy observation of some unknown $\\theta^*$. Under the
<strong>Robbins-Monro conditions</strong> on the step sizes,</p>

$$\\sum_{n=1}^\\infty \\alpha_n \\;=\\; \\infty, \\qquad \\sum_{n=1}^\\infty \\alpha_n^2 \\;<\\; \\infty,$$

<p>the estimate $\\theta_n$ converges almost surely to $\\theta^*$. The classical example is
the running mean, where $\\alpha_n = 1/n$ satisfies both conditions.</p>

<p>TD(0) has the same structure. The update $V(s) \\leftarrow V(s) + \\alpha [r + \\gamma V(s') - V(s)]$
moves $V(s)$ toward $r + \\gamma V(s')$, which is a noisy observation of
$(\\mathcal{T}^\\pi V)(s)$ — the Bellman operator applied to the current estimate.
<strong>Tsitsiklis (1994)</strong> proved that TD(0) converges to $V^\\pi$ almost surely under three
conditions: every state is visited infinitely often, the Robbins-Monro conditions on
$\\alpha_n(s)$ hold per-state, and rewards have bounded variance. The proof uses two
ingredients: the Bellman operator $\\mathcal{T}^\\pi$ is a $\\gamma$-contraction (Lesson 4);
stochastic approximation extends to moving targets driven by a contraction.</p>

<div class="callout callout--info">
  <strong>Pedagogical aside on Robbins-Monro choices.</strong>
  The condition $\\sum \\alpha_n = \\infty$ ensures the estimator can travel an
  unbounded distance; $\\sum \\alpha_n^2 < \\infty$ ensures the noise accumulates only
  finitely. A constant $\\alpha$ satisfies the first but violates the second — and indeed
  constant-$\\alpha$ TD(0) does not converge to $V^\\pi$; it oscillates around $V^\\pi$
  with variance proportional to $\\alpha$. This is correct, important, and often overlooked.
</div>

<h3>Empirical: TD(0) on the gridworld</h3>

<p>On the running $3\\times 3$ gridworld with uniform random policy
(true $V^\\pi(0,0) = {-0.4205}$), TD(0) with constant $\\alpha = 0.1$ produces:</p>

<table class="rl-table">
<thead><tr><th>$N$ (episodes)</th><th>mean</th><th>std</th><th>RMSE</th></tr></thead>
<tbody>
<tr><td>100</td><td>−0.4087</td><td>0.0481</td><td>0.0496</td></tr>
<tr><td>500</td><td>−0.4382</td><td>0.0539</td><td>0.0566</td></tr>
<tr><td>2,000</td><td>−0.4450</td><td>0.0296</td><td>0.0385</td></tr>
<tr><td>5,000</td><td>−0.4255</td><td>0.0450</td><td>0.0453</td></tr>
</tbody>
</table>

<p>The estimator's standard deviation does not shrink monotonically with $N$: it bottoms
out around $\\sqrt{\\alpha/2} \\cdot \\sigma \\approx 0.04$ and oscillates. This is the
constant-$\\alpha$ behavior the Robbins-Monro framework predicts.</p>

<p><strong>Step-size sensitivity</strong> ($N = 5000$, 20 trials):</p>

<table class="rl-table">
<thead><tr><th>$\\alpha$</th><th>mean</th><th>std</th><th>RMSE</th></tr></thead>
<tbody>
<tr><td>0.01</td><td>−0.4266</td><td>0.0156</td><td>0.0168</td></tr>
<tr><td>0.05</td><td>−0.4294</td><td>0.0312</td><td>0.0325</td></tr>
<tr><td>0.10</td><td>−0.4255</td><td>0.0450</td><td>0.0453</td></tr>
<tr><td>0.20</td><td>−0.4338</td><td>0.0742</td><td>0.0754</td></tr>
</tbody>
</table>

<p>The standard deviation grows roughly linearly in $\\alpha$, as predicted. Smaller $\\alpha$
gives lower asymptotic variance but slower convergence; larger $\\alpha$ gives faster
convergence but larger asymptotic variance. A decaying schedule satisfying the
Robbins-Monro conditions avoids the trade-off: the estimator both reaches $V^\\pi$ and
stays there.</p>

<h3>A surprising comparison: TD(0) vs MC on this gridworld</h3>

<p>From Lesson 7, MC at $N = 2000$ has $\\text{std} \\approx 0.011$ (50 trials). TD(0) at
the same $N$ with $\\alpha = 0.1$ has $\\text{std} \\approx 0.030$. <strong>MC has lower variance
than TD(0) here.</strong> This contradicts the textbook claim that "TD has lower variance
than MC." The textbook claim is true on long-horizon problems where MC's per-trajectory
variance grows large. On short-horizon problems (this gridworld has mean episode length
$\\approx 8.7$), MC's per-trajectory variance is small, and TD's constant-$\\alpha$
oscillation dominates the comparison.</p>

<p>The honest summary: TD's variance advantage over MC scales with the horizon and
per-step reward variance. Modern deep RL operates almost entirely in the second regime
(long horizon, noisy rewards), which is why TD dominates practice.</p>

<div class="component-host">
  <td-zero-explorer></td-zero-explorer>
</div>
`);
  },
};
