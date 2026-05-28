import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection05: Section = {
  id: "control-variate-baseline",
  title: "Variance Reduction with Baselines",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>Variance Reduction with Baselines</h2>
<p class="tagline"><em>Subtract a baseline. The gradient stays unbiased. The variance drops.</em></p>

<p>REINFORCE updates the policy in proportion to the return $G_t$. The key insight:
we can subtract any function $b(s_t)$ that depends only on the current state
without changing the expected gradient:</p>

$$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[
  \\sum_t \\gamma^t (G_t - b(s_t)) \\, \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)
\\right].$$

<p>The proof is one line: $\\mathbb{E}[b(s_t) \\cdot \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)] = 0$
because $\\sum_a \\nabla_\\theta \\log \\pi_\\theta(a \\mid s) \\cdot \\pi_\\theta(a \\mid s) = \\nabla_\\theta 1 = 0$.
The score function sums to zero in expectation over actions — a consequence of the score being
a gradient of a normalized distribution.</p>

<h3>What makes a good baseline?</h3>

<p>The optimal baseline (minimizing variance) is the value function $V^\\pi(s)$.
Subtracting $b(s_t) = V^\\pi(s_t)$ gives the <em>advantage</em>:
$A^\\pi(s_t, a_t) = Q^\\pi(s_t, a_t) - V^\\pi(s_t)$.
The advantage isolates whether action $a_t$ is better or worse than average at state $s_t$ —
a tighter signal than the raw return.</p>

<p>In practice we approximate $V^\\pi(s)$ by a learned critic (§6).
Here we compare three baselines on the gridworld:</p>

<ul>
  <li><strong>No baseline (vanilla REINFORCE):</strong> high variance, slow convergence.</li>
  <li><strong>Global mean baseline $\\bar{G}$:</strong> running mean of all returns seen so far. Reduces variance but state-agnostic.</li>
  <li><strong>TD(0) critic baseline:</strong> per-state $V_\\phi(s)$ learned by temporal difference. Largest variance reduction.</li>
</ul>

<pg-variance-reduction-with-baseline></pg-variance-reduction-with-baseline>
`);
  },
};
