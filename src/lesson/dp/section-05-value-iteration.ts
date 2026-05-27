import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const dpSection05: Section = {
  id: 'value-iteration',
  title: 'Value Iteration',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§5</span>Value Iteration</h2>
<p class="tagline">Skip the inner loop. One backup at a time. Same answer.</p>

<p>Modified policy iteration with $m = 1$: do one PE iteration, then immediately
improve. The combined update is</p>

$$\\begin{aligned}
V_{k+1}(s) &= \\max_a \\left[ r(s, a) + \\gamma \\sum_{s'} P(s'|s, a)\\, V_k(s') \\right] \\\\
&= (T^* V_k)(s).
\\end{aligned}$$

<p>PE-then-improve with $m=1$ is the <em>same</em> operation as applying the
<strong>Bellman optimality operator</strong> $T^*$. Hence:</p>

<pre class="dp-pseudocode">Value Iteration

  Initialize: V₀ ≡ 0
  for k = 0, 1, 2, ...
    V_{k+1}(s) ← max_a [r(s,a) + γ Σ P(s'|s,a) V_k(s')]   <span class="dp-comment">for all s</span>
    if ‖V_{k+1} − V_k‖∞ &lt; ε(1−γ)/γ:
      π_final(s) ← argmax_a [r(s,a) + γ Σ P(s'|s,a) V_{k+1}(s')]
      return (V_{k+1}, π_final)</pre>

<p><strong>Convergence.</strong> From Prereq C, $T^*$ is a $\\gamma$-contraction. By Banach,
$V_k \\to V^*$ geometrically at rate $\\gamma$. The stopping criterion is exactly the
<em>a posteriori</em> bound from that lesson.</p>

<h3>Numerical Trace — 5 Iterations on Our Gridworld</h3>

<p>Value iteration on our gridworld ($\\gamma = 0.9$). Value "ripples outward" from
terminal states:</p>

<div class="dp-table-wrap">
<table class="dp-table dp-vi-trace">
  <thead><tr><th>$k$</th><th>(0,0)</th><th>(0,1)</th><th>(0,2)</th><th>(1,0)</th><th>(2,0)</th><th>(2,1)</th></tr></thead>
  <tbody>
    <tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
    <tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td><strong>1</strong></td></tr>
    <tr><td>2</td><td>0</td><td>0</td><td><strong>0.9</strong></td><td>0</td><td><strong>0.9</strong></td><td>1</td></tr>
    <tr><td>3</td><td>0</td><td><strong>0.81</strong></td><td>0.9</td><td><strong>0.81</strong></td><td>0.9</td><td>1</td></tr>
    <tr><td>4</td><td><strong>0.729</strong></td><td>0.81</td><td>0.9</td><td>0.81</td><td>0.9</td><td>1</td></tr>
    <tr><td>5</td><td>0.729</td><td>0.81</td><td>0.9</td><td>0.81</td><td>0.9</td><td>1</td></tr>
  </tbody>
</table>
</div>

<p>Value iteration runs in <strong>5 iterations</strong>. Policy iteration ran in 3 —
<em>fewer outer iterations but more total work per iteration</em>. In the centerpiece
lab below you can race them side-by-side and watch both converge to the same $V^*$.</p>

<p>The story value iteration tells: information about rewards <em>ripples outward</em> from
terminal states, one cell per backup. This is visible in the visualization below.</p>

<h3>Stopping Criterion in Practice</h3>

<p>For $\\gamma = 0.9, \\epsilon = 0.01$:</p>

<div class="dp-table-wrap">
<table class="dp-table">
  <thead><tr><th>$k$</th><th>$\\|V_k - V_{k-1}\\|_\\infty$</th><th>Bound $\\frac{\\gamma}{1-\\gamma}\\|\\Delta\\|$</th><th>True $\\|V_k - V^*\\|_\\infty$</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>1.000</td><td>9.000</td><td>0.900</td></tr>
    <tr><td>2</td><td>0.900</td><td>8.100</td><td>0.810</td></tr>
    <tr><td>3</td><td>0.810</td><td>7.290</td><td>0.729</td></tr>
    <tr><td>4</td><td>0.729</td><td>6.561</td><td>0.000</td></tr>
  </tbody>
</table>
</div>

<p>The bound is <em>very</em> conservative — it predicts errors of ~9.0 when the true error
is 0.9. That's because the bound assumes worst-case contraction; empirical convergence
is much faster. On the other hand: the bound is <em>correct</em> — the true error is always
below it. Sloppy bounds beat hopeful guesses.</p>

<dp-algorithm-lab></dp-algorithm-lab>`);
  },
};
