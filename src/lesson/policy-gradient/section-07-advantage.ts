import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection07: Section = {
  id: "advantage-function",
  title: "Advantage Estimation and GAE",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§7</span>Advantage Estimation and GAE</h2>
<p class="tagline"><em>n-step returns interpolate between MC (zero bias, high variance) and TD(0) (high bias, low variance).</em></p>

<p>Both REINFORCE and one-step actor-critic are extreme points on the bias-variance spectrum.
The $n$-step advantage estimator interpolates between them:</p>

$$\\hat{A}^{(n)}_t = \\left(\\sum_{k=0}^{n-1} \\gamma^k r_{t+k}\\right) + \\gamma^n V_\\phi(s_{t+n}) - V_\\phi(s_t).$$

<ul>
  <li>$n = 1$: one-step TD error. High bias (if $V_\\phi$ is wrong), low variance.</li>
  <li>$n \\to \\infty$: full MC return minus baseline. Zero bias, high variance.</li>
  <li>Intermediate $n$: the sweet spot — depends on how good $V_\\phi$ is and how long the episode is.</li>
</ul>

<h3>Generalized Advantage Estimation (GAE)</h3>

<p>Rather than choosing a fixed $n$, Generalized Advantage Estimation (Schulman et al., 2016)
takes an exponentially-weighted sum of all $n$-step estimators parameterized by $\\lambda \\in [0,1]$:</p>

$$\\hat{A}^{\\text{GAE}(\\lambda)}_t = \\sum_{l=0}^{\\infty} (\\gamma \\lambda)^l \\, \\delta_{t+l},
\\quad \\delta_t = r_{t+1} + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t).$$

<ul>
  <li>$\\lambda = 0$: reduces to one-step TD. Minimum variance, maximum bias.</li>
  <li>$\\lambda = 1$: reduces to MC minus baseline. Zero bias, maximum variance.</li>
  <li>$\\lambda \\in (0,1)$: smoothly interpolates. In practice $\\lambda = 0.95$ is a common default.</li>
</ul>

<p>GAE is the advantage estimator used in PPO, A3C/A2C, and most modern policy gradient implementations.
Understanding its bias-variance trade-off is the direct prerequisite to understanding why
PPO uses clipping rather than a hard trust region.</p>

<pg-bias-variance-advantage></pg-bias-variance-advantage>
`);
  },
};
