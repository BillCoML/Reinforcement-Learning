import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const contrSection02: Section = {
  id: "contraction-property",
  title: "Contractions",
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§2</span>Contractions</h2>
<p class="tagline">A map that shrinks distances by a constant factor.</p>

<p>Let $(X, d)$ be a metric space and $T : X \\to X$ a map.</p>

<p>$T$ is a <strong>contraction</strong> (or <strong>contractive mapping</strong>) if there exists a
constant $c \\in [0, 1)$ such that</p>

$$\\boxed{d(T(x), T(y)) \\;\\leq\\; c \\cdot d(x, y) \\qquad \\text{for all } x, y \\in X.}$$

<p>The constant $c$ is called the <strong>contraction constant</strong> (or <strong>Lipschitz
constant</strong>). Two things to notice:</p>

<ol>
  <li><strong>The "$&lt;$" matters.</strong> If $c = 1$, $T$ is merely <em>non-expansive</em>, not
  a contraction. The theorem in §3 <em>requires</em> $c &lt; 1$ strictly.</li>
  <li><strong>The constant must be uniform.</strong> The inequality has to hold for <em>every</em>
  pair $(x, y)$ with the <em>same</em> $c$.</li>
</ol>

<table>
  <thead><tr><th>$T$</th><th>Domain</th><th style="text-align:center">Contraction?</th><th>Constant</th></tr></thead>
  <tbody>
    <tr><td>$T(x) = 0.5x + 1$</td><td>$\\mathbb{R}$</td><td style="text-align:center">✓</td><td>$c = 0.5$</td></tr>
    <tr><td>$T(x) = 2x - 1$</td><td>$\\mathbb{R}$</td><td style="text-align:center">✗</td><td>Lipschitz constant 2</td></tr>
    <tr><td>$T(x) = x + 1$ (translation)</td><td>$\\mathbb{R}$</td><td style="text-align:center">✗</td><td>$c = 1$ — only non-expansive</td></tr>
    <tr><td>$T(x) = \\cos(x)$</td><td>$\\mathbb{R}$</td><td style="text-align:center">✓ (near $x^*$)</td><td>$c \\approx 0.84$</td></tr>
    <tr><td>$T(x) = x^2$</td><td>$[0,1]$</td><td style="text-align:center">✗ on full interval</td><td>Local only near 0</td></tr>
    <tr><td>$T(V) = R^\\pi + \\gamma P^\\pi V$</td><td>$\\mathbb{R}^K$, sup-norm</td><td style="text-align:center"><strong>✓ (§5 proves this)</strong></td><td>$c = \\gamma$</td></tr>
  </tbody>
</table>

<p><strong>The connection to Lipschitz continuity.</strong> A map $T$ is <strong>Lipschitz</strong> with
constant $L$ if $d(T(x), T(y)) \\leq L \\cdot d(x, y)$. A contraction is a Lipschitz
map with $L = c &lt; 1$. So <em>all</em> contractions are Lipschitz, hence continuous.</p>

<p><strong>The two diagnostic tools for "is this a contraction?"</strong></p>

<ul>
  <li><strong>In $\\mathbb{R}$:</strong> if $T$ is differentiable, $T$ is a contraction with
  constant $c$ iff $|T'(x)| \\leq c &lt; 1$ everywhere on the domain.</li>

  <li><strong>In $\\mathbb{R}^K$ with sup-norm and a linear map $T(x) = Ax + b$:</strong>
  the relevant quantity is the <strong>operator norm</strong>
  $\\|A\\|_\\infty := \\max_i \\sum_j |A_{ij}|$ (maximum absolute row sum).
  Then $T$ is a contraction iff $\\|A\\|_\\infty &lt; 1$. This is <em>exactly</em> the criterion
  we'll use in §5 with $A = \\gamma P^\\pi$ — and $\\|P^\\pi\\|_\\infty = 1$
  (row-stochastic), so $\\|\\gamma P^\\pi\\|_\\infty = \\gamma &lt; 1$.</li>
</ul>

<contraction-iterator-1d></contraction-iterator-1d>`);
  },
};
