import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const contrSection05: Section = {
  id: "bellman-pi-contraction",
  title: "Bellman Operators are Contractions",
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§5</span>Bellman Operators are Contractions</h2>
<p class="tagline">Two theorems, two proofs. Now value iteration converges.</p>

<p>We now apply the Banach theorem to RL. Two operators need contracting:
$T^\\pi$ (Bellman expectation, for policy evaluation) and $T^*$ (Bellman
optimality, for value iteration).</p>

<p>The state space $\\mathcal{S}$ has $K$ states. Value functions live in
$\\mathbb{R}^K$, equipped with the <strong>sup-norm</strong>
$\\|V\\|_\\infty := \\max_s |V(s)|$. The metric is $d(V, V') := \\|V - V'\\|_\\infty$.
This space is complete.</p>

<hr>

<h3>Theorem 1 — Bellman Expectation Contraction</h3>

<p>For any policy $\\pi$ and discount $\\gamma \\in [0, 1)$, the operator</p>

$$(T^\\pi V)(s) := \\sum_a \\pi(a|s) \\left[ r(s, a) + \\gamma \\sum_{s'} P(s'|s, a)\\, V(s') \\right]$$

<p>is a $\\gamma$-contraction on $(\\mathbb{R}^K, \\|\\cdot\\|_\\infty)$.</p>

<p><em>Proof.</em> In matrix form, $T^\\pi V = R^\\pi + \\gamma P^\\pi V$. So</p>

$$T^\\pi V - T^\\pi V' = \\gamma P^\\pi (V - V').$$

<p>Taking sup-norms,</p>

$$\\|T^\\pi V - T^\\pi V'\\|_\\infty = \\gamma \\|P^\\pi(V - V')\\|_\\infty \\leq \\gamma \\|P^\\pi\\|_\\infty \\cdot \\|V - V'\\|_\\infty.$$

<p>The operator norm $\\|P^\\pi\\|_\\infty = 1$ because $P^\\pi$ is row-stochastic.
Hence</p>

$$\\|T^\\pi V - T^\\pi V'\\|_\\infty \\leq \\gamma \\|V - V'\\|_\\infty. \\qquad \\blacksquare$$

<p>The contraction constant is exactly $\\gamma$. Apply Banach: $T^\\pi$ has a unique
fixed point $V^\\pi$, and $V_{k+1} := T^\\pi V_k$ converges to it at rate $\\gamma$.
This is <em>iterative policy evaluation</em>.</p>

<hr>

<h3>Theorem 2 — Bellman Optimality Contraction</h3>

<p>The Bellman optimality operator</p>

$$(T^* V)(s) := \\max_a \\left[ r(s, a) + \\gamma \\sum_{s'} P(s'|s, a)\\, V(s') \\right]$$

<p>is also a $\\gamma$-contraction on $(\\mathbb{R}^K, \\|\\cdot\\|_\\infty)$.</p>

<p><em>Proof.</em> The new ingredient is the <strong>max-Lipschitz lemma</strong>: for any two
functions $f, g : \\mathcal{A} \\to \\mathbb{R}$,</p>

$$\\left| \\max_a f(a) - \\max_a g(a) \\right| \\leq \\max_a |f(a) - g(a)|.$$

<p>Apply this state-by-state. For each $s$, let
$f_a := r(s,a) + \\gamma \\sum_{s'} P(s'|s,a)\\,V(s')$ and
$g_a := r(s,a) + \\gamma \\sum_{s'} P(s'|s,a)\\,V'(s')$. Then</p>

$$\\begin{aligned}
|(T^* V)(s) - (T^* V')(s)| &= \\left|\\max_a f_a - \\max_a g_a\\right| \\\\
&\\leq \\max_a |f_a - g_a| \\\\
&= \\max_a \\left|\\gamma \\sum_{s'} P(s'|s,a)\\,[V(s') - V'(s')]\\right| \\\\
&\\leq \\gamma \\max_a \\sum_{s'} P(s'|s,a) \\|V - V'\\|_\\infty \\\\
&= \\gamma \\|V - V'\\|_\\infty.
\\end{aligned}$$

<p>Taking the max over $s$,</p>

$$\\|T^* V - T^* V'\\|_\\infty \\leq \\gamma \\|V - V'\\|_\\infty. \\qquad \\blacksquare$$

<p>Apply Banach: $T^*$ has a unique fixed point $V^*$, and iterating $V_{k+1} := T^* V_k$
converges at rate $\\gamma$. This is <em>value iteration</em>.</p>

<hr>

<h3>Numerical Verification on the 3×3 Gridworld</h3>

<p>Two arbitrary starting value functions: $V_1 = \\mathbf{0}$ and $V_2 = 2 \\cdot \\mathbf{1}$.
Initial sup-distance: $\\|V_1 - V_2\\|_\\infty = 2$. Apply $T^\\pi$ for uniform policy
($\\gamma = 0.9$):</p>

<table>
  <thead>
    <tr><th style="text-align:right">$k$</th><th style="text-align:right">$\\|V_1^{(k)} - V_2^{(k)}\\|_\\infty$</th><th style="text-align:right">empirical ratio</th></tr>
  </thead>
  <tbody>
    <tr><td style="text-align:right">0</td><td style="text-align:right">2.0000</td><td style="text-align:right">—</td></tr>
    <tr><td style="text-align:right">1</td><td style="text-align:right">1.8000</td><td style="text-align:right">0.9000</td></tr>
    <tr><td style="text-align:right">2</td><td style="text-align:right">1.6200</td><td style="text-align:right">0.9000</td></tr>
    <tr><td style="text-align:right">3</td><td style="text-align:right">1.2758</td><td style="text-align:right">0.7875</td></tr>
    <tr><td style="text-align:right">4</td><td style="text-align:right">1.0252</td><td style="text-align:right">0.8036</td></tr>
    <tr><td style="text-align:right">5</td><td style="text-align:right">0.8119</td><td style="text-align:right">0.7920</td></tr>
    <tr><td style="text-align:right">10</td><td style="text-align:right">0.2305</td><td style="text-align:right">0.7722</td></tr>
  </tbody>
</table>

<p>The first two ratios are exactly $\\gamma = 0.9$. After that they drift slightly
<em>below</em> $\\gamma$ — the theorem only guarantees an upper bound; actual convergence
can be faster once the difference vector has been smoothed by repeated averaging.</p>

<h3>Why This Matters for the Rest of the Curriculum</h3>

<ul>
  <li><strong>Value iteration converges.</strong> Lesson 3 iterates $V_{k+1} = T^* V_k$ until
  $\\|V_{k+1} - V_k\\|_\\infty &lt; \\epsilon(1-\\gamma)/\\gamma$, guaranteeing
  $\\|V_k - V^*\\|_\\infty &lt; \\epsilon$.</li>
  <li><strong>Policy iteration converges in finitely many steps.</strong> Each policy
  evaluation is a Banach iteration at rate $\\gamma$.</li>
  <li><strong>TD(0) and Q-learning converge.</strong> The stochastic Bellman backup
  in TD methods is a stochastic approximation; the Robbins-Monro theorem covers
  this setting. The contraction is a necessary ingredient.</li>
  <li><strong>The deadly triad of function approximation.</strong> When $V$ is replaced
  by $V_\\theta$, the projection step can be non-expansive but not a contraction.
  This is exactly why bootstrapping + off-policy + function approximation can
  diverge (Lesson 6).</li>
</ul>

<bellman-contraction-explorer></bellman-contraction-explorer>`);
  },
};
