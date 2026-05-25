import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const contrSection03: Section = {
  id: "banach-fixed-point",
  title: "The Banach Fixed-Point Theorem",
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§3</span>The Banach Fixed-Point Theorem</h2>
<p class="tagline">Iterate. Converge. Geometrically.</p>

<p><strong>Theorem (Banach Fixed-Point, 1922).</strong> Let $(X, d)$ be a <em>complete</em>
metric space and $T : X \\to X$ a contraction with constant $c \\in [0, 1)$. Then:</p>

<ol>
  <li>$T$ has a <strong>unique fixed point</strong> $x^* \\in X$, meaning $T(x^*) = x^*$.</li>
  <li>For any starting point $x_0 \\in X$, the iterates $x_{k+1} := T(x_k)$ converge to $x^*$.</li>
  <li>The convergence is <strong>geometric</strong>:
  $$\\boxed{d(x_k, x^*) \\;\\leq\\; \\frac{c^k}{1 - c} \\cdot d(x_1, x_0).}$$
  </li>
</ol>

<hr>

<h3>Proof</h3>

<p><strong>Step 1: The iteration produces a Cauchy sequence.</strong> For any $k \\geq 0$,</p>

$$d(x_{k+1}, x_k) = d(T(x_k), T(x_{k-1})) \\leq c \\cdot d(x_k, x_{k-1}).$$

<p>By induction, $d(x_{k+1}, x_k) \\leq c^k \\cdot d(x_1, x_0)$. Now for $m > n$,
by the triangle inequality and geometric summation,</p>

$$d(x_m, x_n) \\leq \\sum_{k=n}^{m-1} d(x_{k+1}, x_k) \\leq \\sum_{k=n}^{m-1} c^k \\cdot d(x_1, x_0) \\leq \\frac{c^n}{1 - c} \\cdot d(x_1, x_0).$$

<p>The right-hand side $\\to 0$ as $n \\to \\infty$, independently of $m$. So $\\{x_k\\}$ is Cauchy.</p>

<p><strong>Step 2: The limit exists in $X$.</strong> Since $X$ is complete, the Cauchy
sequence converges: $x_k \\to x^*$ for some $x^* \\in X$. (This is the <em>only</em>
place completeness is used — it lets us promote "Cauchy" to "has a limit in the space.")</p>

<p><strong>Step 3: The limit is a fixed point.</strong> Apply $T$ to both sides of
$x_k \\to x^*$. Since $T$ is a contraction it is continuous, so $T(x_k) \\to T(x^*)$.
But $T(x_k) = x_{k+1}$, which also converges to $x^*$. By uniqueness of limits, $T(x^*) = x^*$.</p>

<p><strong>Step 4: The fixed point is unique.</strong> If $T(y^*) = y^*$ too, then</p>

$$d(x^*, y^*) = d(T(x^*), T(y^*)) \\leq c \\cdot d(x^*, y^*).$$

<p>Since $c &lt; 1$, this forces $d(x^*, y^*) = 0$, hence $x^* = y^*$.</p>

<p><strong>Step 5: The error bound.</strong> Take $n = k$, $m \\to \\infty$ in the Step 1 inequality:</p>

$$d(x_k, x^*) \\leq \\lim_{m \\to \\infty} d(x_m, x_n)\\big|_{n=k} \\leq \\frac{c^k}{1 - c} \\cdot d(x_1, x_0). \\qquad \\blacksquare$$

<hr>

<p><strong>The proof in one sentence.</strong> Successive iterates get closer because $T$
shrinks distances; sum the geometric series to bound how far they can drift;
completeness ensures the limit exists; uniqueness falls out of the contraction
inequality applied to two purported fixed points.</p>

<hr>

<h3>The Error Bound in Practice</h3>

<p>The bound</p>

$$d(x_k, x^*) \\leq \\frac{c^k}{1 - c} \\cdot d(x_1, x_0)$$

<p>is a <strong>computable stopping criterion</strong>. After $k$ iterations, we know how close we
are to $x^*$ without ever knowing $x^*$ itself: compute $d(x_1, x_0)$ (the first
step's distance) and multiply by $c^k / (1 - c)$.</p>

<p>There's also an <em>a posteriori</em> bound that's often tighter:</p>

$$d(x_k, x^*) \\leq \\frac{c}{1 - c} \\cdot d(x_k, x_{k-1}).$$

<blockquote>
<strong>Forward link → RL</strong> — When we iterate $V_{k+1} = T^* V_k$ in value iteration
(Lesson 3), the stopping criterion
$\\|V_k - V_{k-1}\\|_\\infty \\leq \\epsilon(1-\\gamma)/\\gamma$ guarantees
$\\|V_k - V^*\\|_\\infty \\leq \\epsilon$. This is literally the <em>a posteriori</em>
bound with $c = \\gamma$.
</blockquote>

<p><strong>Numerical verification.</strong> For $T(x) = 0.5x + 1$ with $x_0 = 10$, fixed point $x^* = 2$:</p>

<table>
  <thead>
    <tr><th style="text-align:right">$k$</th><th style="text-align:right">$x_k$</th><th style="text-align:right">actual $|x_k - x^*|$</th><th style="text-align:right">bound $c^k/(1-c)\\cdot|x_1-x_0|$</th></tr>
  </thead>
  <tbody>
    <tr><td style="text-align:right">0</td><td style="text-align:right">10.000</td><td style="text-align:right">8.0000</td><td style="text-align:right">8.0000</td></tr>
    <tr><td style="text-align:right">1</td><td style="text-align:right">6.000</td><td style="text-align:right">4.0000</td><td style="text-align:right">4.0000</td></tr>
    <tr><td style="text-align:right">2</td><td style="text-align:right">4.000</td><td style="text-align:right">2.0000</td><td style="text-align:right">2.0000</td></tr>
    <tr><td style="text-align:right">3</td><td style="text-align:right">3.000</td><td style="text-align:right">1.0000</td><td style="text-align:right">1.0000</td></tr>
    <tr><td style="text-align:right">4</td><td style="text-align:right">2.500</td><td style="text-align:right">0.5000</td><td style="text-align:right">0.5000</td></tr>
    <tr><td style="text-align:right">5</td><td style="text-align:right">2.250</td><td style="text-align:right">0.2500</td><td style="text-align:right">0.2500</td></tr>
    <tr><td style="text-align:right">6</td><td style="text-align:right">2.125</td><td style="text-align:right">0.1250</td><td style="text-align:right">0.1250</td></tr>
  </tbody>
</table>

<banach-iteration-2d></banach-iteration-2d>`);
  },
};
