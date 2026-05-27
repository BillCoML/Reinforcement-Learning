import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const isSection03: Section = {
  id: 'trajectory-is',
  title: 'Trajectory Importance Sampling in Reinforcement Learning',
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>Trajectory Importance Sampling in Reinforcement Learning</h2>
<p class="tagline">Estimate the value of one policy using trajectories sampled under another.</p>

<p>We now arrive at the application that justifies this lesson. Suppose we have a
target policy $\\pi_t$ and we want to estimate $V^{\\pi_t}(s_0)$, the value of
$\\pi_t$ starting from state $s_0$. We do not have the ability to roll out $\\pi_t$
directly. Perhaps we are debugging an algorithm that decides which policy to deploy
and we want to know how well a candidate policy would have performed before deploying
it; perhaps we are in the offline setting of Lesson 15 where the dataset was
collected under some unknown behavior policy and there is no possibility of new
sampling. Whatever the reason, we have trajectories from a behavior policy $\\pi_b$,
and we need to extract a $V^{\\pi_t}$ estimate from them.</p>

<p>A trajectory $\\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \\ldots, s_T)$ has joint
probability under any policy $\\pi$:</p>

$$\\Pr_\\pi(\\tau) \\;=\\; \\mu_0(s_0) \\prod_{t=0}^{T-1} \\pi(a_t \\mid s_t) \\,
P(s_{t+1} \\mid s_t, a_t).$$

<p>The return is $G_0 := \\sum_{t=0}^{T-1} \\gamma^t r_{t+1}$. By the
importance-sampling identity applied to trajectories under $\\pi_b$ versus $\\pi_t$:</p>

$$\\mathbb{E}_{\\pi_t}[G_0] \\;=\\; \\mathbb{E}_{\\pi_b}\\!\\left[
\\frac{\\Pr_{\\pi_t}(\\tau)}{\\Pr_{\\pi_b}(\\tau)} G_0 \\right].$$

<p>The ratio of the two trajectory probabilities simplifies. The initial state
distribution $\\mu_0$ and the transition kernel $P$ are properties of the
environment, not the policy, so they appear in both numerator and denominator and
cancel. What remains is the product of per-step policy ratios:</p>

$$\\boxed{\\rho_{0:T-1} \\;:=\\; \\prod_{t=0}^{T-1}
\\frac{\\pi_t(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}.}$$

<p>We call this the <strong>trajectory importance weight</strong>. The identity now reads</p>

$$\\boxed{V^{\\pi_t}(s_0) \\;=\\; \\mathbb{E}_{\\pi_b}\\!\\left[
\\rho_{0:T-1} \\cdot G_0 \\right].}$$

<p>This is trajectory-level importance sampling for reinforcement learning. It is
exact: the expectation on the right equals $V^{\\pi_t}(s_0)$ precisely. The
estimator that averages this expression across $N$ independently sampled trajectories
is unbiased. The variance, however, is where the story begins.</p>

<h3>A worked example on the gridworld</h3>

<p>Take the running gridworld from Lessons 2 through 3. The target policy $\\pi_t$ is
the deterministic optimal policy (right–right–down–down from the start state). The
behavior policy $\\pi_b$ is uniform random over the four actions. The discount factor
is $\\gamma = 0.9$. The true value $V^{\\pi_t}(0,0)$ is $\\gamma^3 = 0.729$ (computed
in Lesson 3 via exact policy evaluation).</p>

<p>The optimal trajectory from $(0,0)$ to the goal under this deterministic $\\pi_t$
is unique and has length four. Under the uniform behavior policy, the probability
that a sampled trajectory matches the optimal sequence step-for-step is
$(1/4)^4 = 1/256 \\approx 0.0039$. Any deviation from the optimal action at any
step makes the corresponding per-step ratio zero, which kills the entire trajectory
weight by the product structure.</p>

<p>When a trajectory does match, the weight is $(1/(1/4))^4 = 4^4 = 256$. So we are
computing an estimator where roughly 99.6% of trajectories contribute zero and
roughly 0.4% contribute $256 \\times G_0 = 256 \\times 0.729 \\approx 186.6$.</p>

<p>The estimator behavior at three sample sizes, averaged over fifty trials:</p>

<div class="table-wrap">
<table class="spec-table">
  <thead>
    <tr>
      <th>$N$ (trajectories)</th>
      <th>Non-zero / $N$</th>
      <th>Ordinary IS (mean ± SD)</th>
      <th>Weighted IS (mean ± SD)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>100</td><td>0.3 / 100</td><td>0.63 ± 0.96</td><td>0.23 ± 0.34 (huge bias)</td></tr>
    <tr><td>1,000</td><td>3.7 / 1000</td><td>0.68 ± 0.38</td><td>0.70 ± 0.14</td></tr>
    <tr><td>10,000</td><td>38 / 10000</td><td>0.71 ± 0.13</td><td><strong>0.7290 ± 0.0000</strong></td></tr>
  </tbody>
</table>
</div>

<p>The last row deserves attention. At $N = 10000$, weighted IS converges exactly to
$0.7290$ with zero sample variance across the fifty trials. Why? Because all
matching trajectories under this configuration are identical: they all take the same
path, the same length, the same return, and have the same weight. The weighted-IS
numerator and denominator are perfectly proportional, and the ratio is exact. This is
a remarkable property unique to deterministic targets.</p>

<h3>The variance of $\\rho_{0:T-1}$ is the entire problem</h3>

<p>For an episodic task with horizon $T$, the trajectory weight is a product of $T$
independent random ratios. The variance of a product grows roughly multiplicatively
in $T$. So trajectory IS has variance that grows <em>exponentially in the
horizon</em>. This is why off-policy methods become brittle as episodes get longer,
and why we will work hard in Lesson 8 to compute per-step backups (which require only
one ratio at a time and avoid the exponential blow-up).</p>

<div class="callout callout--warning">
  <strong>Practical implication.</strong> Off-policy Monte Carlo with a behavior
  policy that barely overlaps with the target policy needs enormous sample sizes.
  In our toy example, getting a reliable estimate required ten thousand trajectories.
  Lessons 8 (TD) and 11 (PPO) will partially address this by working at the per-step
  level rather than at the trajectory level.
</div>

<div class="component-host component-host--wide">
  <trajectory-is-explorer></trajectory-is-explorer>
</div>
`);
  },
};
