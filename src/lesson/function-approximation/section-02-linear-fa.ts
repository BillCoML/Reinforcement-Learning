import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection02: Section = {
  id: "linear-fa",
  title: "Linear Function Approximation",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>Linear Function Approximation</h2>
<p class="tagline"><em>The simplest function class. Convergence theory works. Almost.</em></p>

<p>The simplest non-tabular function class is <strong>linear</strong>: represent
$V_\\theta(s) = \\phi(s)^\\top \\theta$ for a feature vector
$\\phi(s) \\in \\mathbb{R}^d$ and parameter vector $\\theta \\in \\mathbb{R}^d$.
The feature vector is a fixed, hand-designed encoding of the state. The parameters
$\\theta$ are learned.</p>

<p>For the curriculum's $3 \\times 3$ gridworld, a natural feature vector is the one-hot
indicator: $\\phi(s) \\in \\mathbb{R}^9$ with a single 1 in position $s$. With this choice,
linear FA is <em>equivalent</em> to tabular representation. To get genuine generalization
we need features that share information across states. A simple choice is
$\\phi(s) = (r, c, 1)$ for state $(r, c)$: three features encoding row, column, and a
bias. This forces $V_\\theta(r, c) = \\theta_1 r + \\theta_2 c + \\theta_3$, a plane in
$(r, c)$ space. The function class is too restrictive — the true $V^\\pi$ is not linear in
$(r, c)$ — but learning is well-defined.</p>

<h3 id="semi-gradient-td">Semi-gradient TD(0)</h3>

<p>Adapting TD(0) to parametric $V_\\theta$: we want to minimize the TD error
$\\delta_t = r + \\gamma V_\\theta(s') - V_\\theta(s)$.
The naive approach is gradient descent on the squared TD error. But the bootstrap target
$r + \\gamma V_\\theta(s')$ depends on $\\theta$ itself. The convention in TD learning is to
treat $r + \\gamma V_\\theta(s')$ as a <em>constant</em> with respect to $\\theta$,
giving the <strong>semi-gradient</strong> update:</p>

$$\\theta \\;\\leftarrow\\; \\theta + \\alpha \\delta_t \\nabla_\\theta V_\\theta(s_t),$$

<p>which for linear FA simplifies to</p>

$$\\theta \\;\\leftarrow\\; \\theta + \\alpha \\delta_t \\phi(s_t).$$

<p>The "semi-gradient" terminology is from Sutton &amp; Barto: it is not a true gradient of
any objective, but a heuristic that converges under the right conditions.</p>

<h3>On-policy linear TD converges</h3>

<p>Tsitsiklis &amp; Van Roy (1997) proved that semi-gradient TD with linear function
approximation converges <em>on-policy</em>. The fixed point is the
<strong>projected Bellman equation</strong>:</p>

$$V_{\\theta^*} \\;=\\; \\Pi T^\\pi V_{\\theta^*},$$

<p>where $\\Pi$ is the orthogonal projection onto the linear function class and $T^\\pi$ is
the Bellman expectation operator. The fixed point $V_{\\theta^*}$ is the best linear
approximation to $V^\\pi$ in the weighted MSE sense, with weights given by the
stationary distribution of $\\pi$. The on-policy weighting makes $\\Pi T^\\pi$ a
contraction in the $\\pi$-weighted norm, guaranteeing a unique fixed point.</p>

<p>When the function class is rich enough to contain $V^\\pi$ (e.g., with one-hot features),
on-policy linear TD recovers it exactly. When not rich enough, the algorithm still
converges, just to the best in-class approximation.
<strong>The convergence guarantee is robust to the function class's expressivity.</strong></p>

<blockquote class="crosslink">
  <strong>Crosslink to Lesson 4.</strong> Tsitsiklis &amp; Van Roy's proof relies on the
  contraction property of $\\Pi T^\\pi$ in the $\\pi$-weighted norm. The Bellman expectation
  operator is a $\\gamma$-contraction (Lesson 4); the projection $\\Pi$ is non-expansive in
  the weighted norm. Their composition is therefore a $\\gamma$-contraction, guaranteeing
  convergence by Banach's fixed-point theorem.
</blockquote>

<dqn-linear-fa-convergence></dqn-linear-fa-convergence>
`);
  },
};
