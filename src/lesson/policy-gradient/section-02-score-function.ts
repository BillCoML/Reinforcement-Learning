import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection02: Section = {
  id: "score-function-estimator",
  title: "The Score Function Estimator",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>The Score Function Estimator</h2>
<p class="tagline"><em>The log-derivative trick: how to differentiate through an expectation over θ-dependent samples.</em></p>

<p>We want to compute $\\nabla_\\theta J(\\theta)$, but $J(\\theta)$ is an expectation over trajectories
sampled from $\\pi_\\theta$. The challenge: differentiating through the sampling distribution
itself. The <em>score function estimator</em> (also called REINFORCE or the log-derivative trick)
resolves this with a single identity.</p>

<h3>The log-derivative identity</h3>

<p>For any differentiable distribution $p_\\theta(x)$ and function $f(x)$:</p>

$$\\nabla_\\theta \\,\\mathbb{E}_{x \\sim p_\\theta}[f(x)]
\\;=\\; \\mathbb{E}_{x \\sim p_\\theta}\\!\\left[ f(x)\\, \\nabla_\\theta \\log p_\\theta(x) \\right].$$

<p>The derivation is one line. Note $\\nabla_\\theta p_\\theta(x) = p_\\theta(x)\\, \\nabla_\\theta \\log p_\\theta(x)$
(the log-derivative identity). Insert under the integral:</p>

$$\\nabla_\\theta \\int f(x)\\, p_\\theta(x)\\, dx
= \\int f(x)\\, \\nabla_\\theta p_\\theta(x)\\, dx
= \\mathbb{E}_{p_\\theta}\\!\\left[ f(x)\\, \\nabla_\\theta \\log p_\\theta(x) \\right].$$

<p>The right side is an expectation we can estimate by sampling: draw $x_1, \\ldots, x_N \\sim p_\\theta$
and average $f(x_i)\\, \\nabla_\\theta \\log p_\\theta(x_i)$. No knowledge of $f$'s gradient is required —
only $f$'s value and the log-gradient of the distribution.</p>

<h3>The quantity $\\nabla_\\theta \\log p_\\theta(x)$ is the score function</h3>

<p>The term $\\nabla_\\theta \\log p_\\theta(x)$ is called the <em>score function</em>.
For a softmax policy over $(s, a)$ pairs:</p>

$$\\frac{\\partial}{\\partial \\theta_{s,a'}} \\log \\pi_\\theta(a \\mid s)
= \\mathbf{1}[a' = a] - \\pi_\\theta(a' \\mid s).$$

<p>This is exactly the gradient of the cross-entropy loss — familiar from supervised learning.
It equals $1 - \\pi_\\theta(a \\mid s)$ for the taken action and $-\\pi_\\theta(a' \\mid s)$
for all other actions. It always sums to zero across all $a'$.</p>

<h3>Gaussian policy score function</h3>

<p>For a Gaussian policy $\\pi_\\theta(a \\mid s) = \\mathcal{N}(a;\\, \\mu_\\theta(s), \\sigma^2)$
the score function with respect to $\\mu$ is:</p>

$$\\nabla_{\\mu} \\log \\pi_\\theta(a \\mid s) = \\frac{a - \\mu_\\theta(s)}{\\sigma^2}.$$

<p>The visualization below lets you see the score function estimator in action on a tractable
one-dimensional Gaussian problem where the true gradient is known analytically.
This gives intuition for bias, variance, and sample efficiency before we apply the estimator
to RL trajectories.</p>

<pg-score-function-gaussian></pg-score-function-gaussian>
`);
  },
};
