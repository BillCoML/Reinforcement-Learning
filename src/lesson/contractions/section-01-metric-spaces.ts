import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const contrSection01: Section = {
  id: "metric-space",
  title: "Metric Spaces, Briefly",
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§1</span>Metric Spaces, Briefly</h2>
<p class="tagline">We need a notion of distance, and we need our space to be "closed under limits."</p>

<p>To talk about "contractions" we need a <em>distance</em>. A <strong>metric space</strong> is a
set $X$ equipped with a function $d : X \\times X \\to \\mathbb{R}_{\\geq 0}$ satisfying</p>

<ol>
  <li>$d(x, y) = 0 \\iff x = y$,</li>
  <li>$d(x, y) = d(y, x)$,</li>
  <li>$d(x, z) \\leq d(x, y) + d(y, z)$ &nbsp;(triangle inequality).</li>
</ol>

<p>We won't dwell on these axioms — they capture the geometric meaning of "distance"
in a way that lets us compute. The three metric spaces we'll use in this lesson are:</p>

<ul>
  <li><strong>$\\mathbb{R}$ with $d(x, y) = |x - y|$</strong> — the real line. Trivial, useful for warm-up.</li>
  <li><strong>$\\mathbb{R}^n$ with $d(x, y) = \\|x - y\\|_2$</strong> — Euclidean space.</li>
  <li><strong>$\\mathbb{R}^K$ with $d(x, y) = \\|x - y\\|_\\infty = \\max_i |x_i - y_i|$</strong>
  — the <strong>sup-norm</strong> (a.k.a. max-norm or $\\ell_\\infty$-norm).
  This is the <em>only</em> metric that will matter for the Bellman analysis in §5.</li>
</ul>

<p>A sequence $\\{x_n\\}$ in a metric space <strong>converges</strong> to $x$ if $d(x_n, x) \\to 0$
as $n \\to \\infty$. A sequence is <strong>Cauchy</strong> if $d(x_m, x_n) \\to 0$ as
$m, n \\to \\infty$ — the elements get arbitrarily close to <em>each other</em>.</p>

<p><strong>Completeness.</strong> A metric space is <strong>complete</strong> if every Cauchy sequence
converges to some limit <em>within the space</em>. Equivalent intuition: the space has no
"missing points" that should be there.</p>

<table>
  <thead><tr><th>Space</th><th style="text-align:center">Complete?</th><th>Why it matters</th></tr></thead>
  <tbody>
    <tr><td>$\\mathbb{R}$ (with $|\\cdot|$)</td><td style="text-align:center">✓</td><td>The classical case.</td></tr>
    <tr><td>$\\mathbb{R}^n$ (any norm)</td><td style="text-align:center">✓</td><td>Our default.</td></tr>
    <tr><td>$\\mathbb{Q}$ (with $|\\cdot|$)</td><td style="text-align:center">✗</td><td>A Cauchy sequence of rationals can converge to an irrational.</td></tr>
    <tr><td>$(0,1)$ open interval</td><td style="text-align:center">✗</td><td>A Cauchy sequence approaching 0 has no limit in the space.</td></tr>
    <tr><td>Continuous functions on $[0,1]$, sup-norm</td><td style="text-align:center">✓</td><td>The function-space analog.</td></tr>
  </tbody>
</table>

<blockquote>
<strong>Why does completeness matter for fixed-point theorems?</strong> Because we want the
<em>limit of our iteration</em> to actually exist in the space we care about. If the
limit lives outside the space, the theorem can't tell us anything useful about it.
The Banach theorem will hand us a Cauchy sequence and a candidate limit —
completeness is what lets us say the limit is in $X$ and is, therefore, a
meaningful object.
</blockquote>

<p>For RL, all our metric spaces will be finite-dimensional vector spaces with
familiar norms — they're complete by inheritance from $\\mathbb{R}$. We won't sweat
this hypothesis again, but it's there in the background of every theorem.</p>

<norm-comparison></norm-comparison>`);
  },
};
