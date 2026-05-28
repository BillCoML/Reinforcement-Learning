import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection02: Section = {
  id: "trust-region-kl",
  title: "Trust Regions and the KL Constraint",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>Trust Regions and the KL Constraint</h2>
<p class="tagline"><em>Bound the policy change, not the parameter change.</em></p>

<p>The step-size problem from §1 has a clean diagnosis: we're measuring step size in the wrong space.
The fix is to measure step size <strong>in policy space</strong>, not parameter space.</p>

<h3>Why KL divergence, not L2 distance?</h3>

<p>We want a measure of "how far did the policy move?" Two natural candidates:</p>

<ol>
<li><strong>L2 distance in parameter space:</strong> $\\|\\theta_{\\mathrm{new}} - \\theta_{\\mathrm{old}}\\|_2$.
This is what a fixed learning rate already controls. We just showed it doesn't correspond to policy change.</li>
<li><strong>L2 distance between policy vectors:</strong> $\\|\\pi_{\\theta_{\\mathrm{new}}} - \\pi_{\\theta_{\\mathrm{old}}}\\|_2$.
Better, but a fixed L2 gap means different things at different points on the simplex.
Two actions with probabilities $(0.6, 0.4)$ vs. $(0.001, 0.999)$ represent very different information gains.</li>
</ol>

<p><strong>KL divergence</strong> has the right properties. It measures the information gained by switching
from $\\pi_{\\mathrm{old}}$ to $\\pi_{\\mathrm{new}}$ (in nats). It is parameterization-invariant:
$D_{\\mathrm{KL}}(\\pi_{\\mathrm{old}} \\| \\pi_{\\mathrm{new}})$ doesn't depend on whether the policy
is parameterized by softmax logits or something else. And it has a clean local approximation.</p>

<h3>The trust-region problem</h3>

$$\\theta_{\\mathrm{new}} = \\arg\\max_{\\theta}\\; \\hat{\\mathcal{L}}(\\theta)
\\quad \\text{s.t.}\\quad
D_{\\mathrm{KL}}\\!\\left(\\pi_{\\theta_{\\mathrm{old}}} \\,\\|\\, \\pi_\\theta\\right) \\le \\delta$$

<p>where $\\hat{\\mathcal{L}}(\\theta)$ is the <strong>surrogate objective</strong> —
the importance-sampled advantage estimate:</p>

$$\\hat{\\mathcal{L}}(\\theta) \\;=\\;
\\mathbb{E}_{s, a \\sim \\pi_{\\theta_{\\mathrm{old}}}}
\\!\\left[\\frac{\\pi_\\theta(a|s)}{\\pi_{\\theta_{\\mathrm{old}}}(a|s)} \\,\\hat{A}(s, a)\\right].$$

<p>The probability ratio
$r_t(\\theta) = \\pi_\\theta(a_t|s_t) / \\pi_{\\theta_{\\mathrm{old}}}(a_t|s_t)$
is exactly the importance-sampling weight from <a href="#" data-lesson="importance-sampling"
data-anchor="is-identity">Lesson 6</a>.
The IS variance problem (weights explode for off-distribution actions) is the same problem
trust regions are solving. We are not solving something new; we are solving the L6 problem
in the policy-optimization context.</p>

<h3>The Fisher information and the natural gradient</h3>

<p>For small $\\|\\theta - \\theta_{\\mathrm{old}}\\|$, KL admits the second-order expansion:</p>

$$D_{\\mathrm{KL}}(\\pi_{\\theta_{\\mathrm{old}}} \\| \\pi_\\theta) \\;\\approx\\;
\\tfrac{1}{2}\\,(\\theta - \\theta_{\\mathrm{old}})^\\top
F(\\theta_{\\mathrm{old}})\\,(\\theta - \\theta_{\\mathrm{old}})$$

<p>where $F(\\theta) = \\mathbb{E}\\!\\left[\\nabla_\\theta \\log \\pi_\\theta(a|s)\\,
\\nabla_\\theta \\log \\pi_\\theta(a|s)^\\top\\right]$
is the <strong>Fisher information matrix</strong>.
The Fisher is the expected outer product of score functions —
the same $\\nabla \\log \\pi_\\theta$ that gave us REINFORCE's gradient direction
also defines the curvature that bounds the trust region.</p>

<p>Under the uniform policy on the gridworld, $F$ is a $36 \\times 36$ block-diagonal matrix
(one $4 \\times 4$ block per state). Each block has rank 3 because the softmax has one
redundant degree of freedom per state. The largest eigenvalue per block is $3/4$ —
the constraint ellipse is slightly smaller than the parameter-space ball in those directions.</p>

<p>The <strong>natural gradient</strong> $F^{-1}\\nabla J$ is the direction of steepest ascent
<em>in KL geometry</em> rather than Euclidean geometry.
TRPO is, in essence, "do a line search along the natural gradient direction, subject to a KL constraint."</p>

<ppo-kl-geometry-plot></ppo-kl-geometry-plot>
`);
  },
};
