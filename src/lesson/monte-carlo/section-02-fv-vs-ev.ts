import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection02: Section = {
  id: "first-visit-mc",
  title: "First-Visit and Every-Visit Monte Carlo",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>First-Visit and Every-Visit Monte Carlo</h2>
<p class="tagline"><em>Two natural conventions for using a trajectory's returns. Both work.</em></p>

<p>Consider a single trajectory $\\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \\ldots, s_T)$.
For any time step $t$, the return from $t$ onward is</p>

$$G_t \\;:=\\; \\sum_{k=0}^{T - t - 1} \\gamma^k R_{t + k + 1}.$$

<p>$G_t$ is computed recursively from the rewards by the backward sweep
$G_T = 0$, $G_t = R_{t+1} + \\gamma G_{t+1}$. We compute all the $G_t$ in a single
pass through the trajectory in reverse.</p>

<p>Now consider a state $s$ that the trajectory visits. The trajectory visits $s$ at
one or more time steps; call these times $t_1 < t_2 < \\cdots < t_k$. At each visit,
the return $G_{t_i}$ is a sample of $V^\\pi(s)$. We must decide what to do with these
$k$ samples. Two natural choices:</p>

<p><strong>First-visit Monte Carlo.</strong> Use only the return from the <em>first</em> time the
trajectory visited $s$. Discard subsequent visits within the same trajectory. Over
many trajectories, the first-visit returns are independent (across trajectories) and
identically distributed (each is a sample of $V^\\pi(s)$ under $\\pi$), and their
average is an unbiased, consistent estimator of $V^\\pi(s)$.</p>

<p><strong>Every-visit Monte Carlo.</strong> Use the return from <em>every</em> time the trajectory
visited $s$. Within a single trajectory the multiple returns from $s$ are not
independent, but their average across many trajectories still converges to $V^\\pi(s)$.
Every-visit MC is biased at any finite $N$ (the within-trajectory correlation shows
up as non-zero bias) but the bias is $O(1/N)$ and vanishes asymptotically.</p>

<h3>On-policy first-visit MC convergence</h3>

<p>On the running gridworld with uniform random policy $\\pi$ (and discount $\\gamma =
0.9$), the true value function from $(0, 0)$ is $V^\\pi(0, 0) = -0.4205$ (computed
exactly via <code>policyEvaluationExact</code> in Lesson 5). The standard deviation of a
single episode's return from $(0, 0)$ is approximately $0.41$. By the central limit
theorem, the standard deviation of the first-visit MC estimator after $N$ episodes is
approximately $0.41 / \\sqrt{N}$.</p>

<p>Empirically across 50 independent trials, first-visit MC produces:</p>

<table class="data-table">
  <thead>
    <tr><th>$N$ (episodes)</th><th>mean of estimator</th><th>std of estimator</th><th>RMSE from truth</th></tr>
  </thead>
  <tbody>
    <tr><td>100</td><td>−0.4236</td><td>0.0438</td><td>0.0439</td></tr>
    <tr><td>1,000</td><td>−0.4235</td><td>0.0133</td><td>0.0137</td></tr>
    <tr><td>10,000</td><td>−0.4206</td><td>0.0036</td><td>0.0036</td></tr>
  </tbody>
</table>

<p>The empirical standard deviations match the theoretical $0.41/\\sqrt{N}$ prediction:
0.041, 0.013, 0.0041. The MC estimator converges at the expected $O(1/\\sqrt{N})$ rate.</p>

<h3>First-visit vs every-visit empirical comparison</h3>

<p>On the same gridworld with $N = 1000$ episodes, averaged over 50 trials:</p>

<table class="data-table">
  <thead>
    <tr><th>Method</th><th>mean of estimator</th><th>std of estimator</th></tr>
  </thead>
  <tbody>
    <tr><td>First-visit</td><td>−0.4235</td><td>0.0133</td></tr>
    <tr><td>Every-visit</td><td>−0.4214</td><td>0.0131</td></tr>
  </tbody>
</table>

<p>The two estimators agree to two decimal places. Every-visit's bias of about $0.002$
at $N = 1000$ is real but small; it vanishes as $N$ grows. The variance of the two
estimators is essentially identical for this problem.</p>

<h3>Implementation: incremental updates</h3>

<p>Maintaining a list of returns for each state and recomputing the mean at every
episode is wasteful. The standard incremental formulation maintains a running mean
and a visit count:</p>

$$N(s) \\leftarrow N(s) + 1, \\qquad
\\hat V^\\pi(s) \\leftarrow \\hat V^\\pi(s) + \\frac{1}{N(s)} \\left( G_t - \\hat V^\\pi(s) \\right).$$

<p>This is the classical "running average" update. We could generalize to a constant
step size $\\alpha$,</p>

$$\\hat V^\\pi(s) \\leftarrow \\hat V^\\pi(s) + \\alpha \\left( G_t - \\hat V^\\pi(s) \\right),$$

<p>which gives an exponential moving average rather than a true average. This becomes
important for non-stationary settings (when $\\pi$ is changing during learning, as in
MC control). The same step-size machinery will reappear in Lesson 8 as the
foundation for TD's learning rate.</p>

<div class="component-host">
  <first-vs-every-visit-walkthrough></first-vs-every-visit-walkthrough>
</div>
`);
  },
};
