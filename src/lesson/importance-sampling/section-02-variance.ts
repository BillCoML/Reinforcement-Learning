import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const isSection02: Section = {
  id: 'is-variance',
  title: 'Variance: When IS Works and When It Fails',
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>Variance: When IS Works and When It Fails</h2>
<p class="tagline">The estimator is unbiased. The variance is the problem.</p>

<p>The ordinary IS estimator is unbiased. So what is the catch? Variance. The
variance of $\\hat\\mu_{\\text{ord}}$ is</p>

$$\\text{Var}_q[\\hat\\mu_{\\text{ord}}] \\;=\\; \\frac{1}{N} \\, \\text{Var}_q[w(X)\\,f(X)]
\\;=\\; \\frac{1}{N}\\left(\\mathbb{E}_q[w(X)^2 f(X)^2] - \\mathbb{E}_p[f(X)]^2\\right).$$

<p>If $\\mathbb{E}_q[w(X)^2 f(X)^2]$ is finite, the IS estimator has finite variance,
the law of large numbers applies in the usual way, and the estimator converges at
the standard $1/\\sqrt{N}$ rate. If $\\mathbb{E}_q[w(X)^2 f(X)^2]$ is infinite, the
IS estimator has <strong>infinite variance</strong>. The central limit theorem fails,
sample averages can wander arbitrarily far from the true expectation, and no amount
of additional sampling can rescue the situation in the way we are used to.</p>

<p>The condition for finite variance can be written as</p>

$$\\mathbb{E}_q[w(X)^2 f(X)^2] \\;=\\; \\int \\frac{p(x)^2}{q(x)} f(x)^2 \\, dx
\\;<\\; \\infty.$$

<p>In words: the importance weight $p(x)/q(x)$ and the integrand $f(x)$ should not
simultaneously be large on regions where $q$ has appreciable mass. The single most
dangerous case is when $q$ has <em>lighter tails</em> than $p$. In the tails of $p$,
the density of $q$ becomes tiny while $p$ still has meaningful mass; the weight
$w(x) = p(x)/q(x)$ then explodes. If $f(x)$ also grows in the tails, the product
$w(x)^2 f(x)^2$ may fail to be integrable.</p>

<p>A useful rule of thumb is that the proposal $q$ should always have heavier tails
than the target $p$. Using a wider Gaussian to estimate a narrower Gaussian is safe.
Using a narrower Gaussian to estimate a wider one is a recipe for variance blow-up.
The variance problem here is not a small numerical issue; it can be catastrophic.</p>

<h3>A worked example</h3>

<p>Take $p = \\mathcal{N}(0,1)$ and $f(X) = X^2$. The true expectation is
$\\mathbb{E}_p[X^2] = 1$ (the variance of the standard normal). We compare three
choices of proposal $q = \\mathcal{N}(0, \\sigma_q^2)$, running fifty trials each
with $N = 1000$ samples.</p>

<div class="table-wrap">
<table class="spec-table">
  <thead>
    <tr>
      <th>$\\sigma_q$</th>
      <th>Ordinary IS SD</th>
      <th>Weighted IS SD</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.0</td><td>0.039</td><td>0.039</td>
      <td>no IS needed; $q = p$</td>
    </tr>
    <tr>
      <td>2.0</td><td>0.021</td><td>0.033</td>
      <td>wider proposal; variance drops</td>
    </tr>
    <tr class="danger-row">
      <td>0.5</td><td>1.460</td><td>0.479</td>
      <td>narrower proposal; near infinite-variance regime</td>
    </tr>
  </tbody>
</table>
</div>

<p>The theoretical threshold for finite variance in this setup is
$\\sigma_q > 1/\\sqrt{2} \\approx 0.707$. With $\\sigma_q = 0.5$, the IS variance is
mathematically infinite, and the empirical estimator is correspondingly unstable.
The ordinary IS standard deviation of 1.46 is larger than the true value of 1.0,
which is the definition of useless.</p>

<p>Notice that the weighted IS estimator is also unstable in the $\\sigma_q = 0.5$
case, but markedly less so. Weighted IS is biased (the denominator introduces a
correlation with the numerator), but its bias is typically $O(1/N)$ and vanishes
asymptotically; in exchange it often delivers a constant-factor variance reduction.
In the $\\sigma_q = 0.5$ case the variance reduction is roughly three-fold.</p>

<h3>Ordinary versus weighted: the trade-off</h3>

<div class="table-wrap">
<table class="spec-table">
  <thead>
    <tr><th></th><th>Ordinary IS</th><th>Weighted IS</th></tr>
  </thead>
  <tbody>
    <tr><td>Bias</td><td>exactly zero</td><td>$O(1/N)$, vanishes asymptotically</td></tr>
    <tr><td>Variance</td><td>possibly infinite</td><td>usually finite, often much smaller</td></tr>
    <tr><td>Use when</td><td>small variance, low-stakes bias</td><td>large variance, can tolerate bias</td></tr>
  </tbody>
</table>
</div>

<p>In practice, weighted IS is strongly preferred in reinforcement learning settings,
even though it is biased. The bias vanishes asymptotically; the variance reduction
is a constant factor that can be ten to a hundred times.</p>

<h3 id="effective-sample-size">Effective sample size</h3>

<p>A useful diagnostic for whether your IS is producing trustworthy estimates is
the <strong>effective sample size</strong>:</p>

$$\\boxed{N_{\\text{eff}} \\;:=\\; \\frac{\\left( \\sum_i w(X_i) \\right)^2}{\\sum_i w(X_i)^2}.}$$

<p>When all weights are equal, $N_{\\text{eff}} = N$. When one sample has all the
weight, $N_{\\text{eff}} = 1$. Values in between measure roughly how many of your
$N$ samples are effectively contributing to the estimator. The practical rule of
thumb is that if $N_{\\text{eff}}/N$ falls below ten percent, your IS estimator is
on shaky ground.</p>

<div class="callout callout--forward">
  <strong>Forward link to PPO</strong> — The clipped surrogate objective in PPO
  (Lesson 11), which has the form
  $\\min(r_t A_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t)$,
  is a heuristic that keeps importance ratios in $[1-\\epsilon, 1+\\epsilon]$ and
  thereby controls the IS variance. Recognizing the clipping for what it is —
  variance control rather than a "weird trick" — is one of the payoffs of this lesson.
</div>

<div class="component-host">
  <variance-explorer></variance-explorer>
</div>
`);
  },
};
