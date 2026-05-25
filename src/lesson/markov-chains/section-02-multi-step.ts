import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection02: Section = {
  id: "n-step-transitions",
  title: "Multi-Step Transitions and Long-Run Behaviour",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§2</span>Multi-Step Transitions and Long-Run Behaviour</h2>
<p class="tagline">Powers of a matrix encode the future.</p>

<p>If $P_{ij}$ is the one-step transition probability $\\Pr(X_1 = j \\mid X_0 = i)$,
what about two steps? The chain goes $i \\to k \\to j$ for some intermediate
state $k$. By the law of total probability and the Markov property,</p>

$$\\Pr(X_2 = j \\mid X_0 = i)
\\;=\\; \\sum_k \\Pr(X_1 = k \\mid X_0 = i) \\cdot \\Pr(X_2 = j \\mid X_1 = k)
\\;=\\; \\sum_k P_{ik} P_{kj}
\\;=\\; (P^2)_{ij}.$$

<p>The $n$-step transition probabilities are the entries of $P^n$:</p>

$$\\boxed{\\Pr(X_n = j \\mid X_0 = i) \\;=\\; (P^n)_{ij}}$$

<p>This is the <strong>Chapman-Kolmogorov equation</strong> in matrix form. (The traditional
non-matrix statement is $P^{(m+n)}_{ij} = \\sum_k P^{(m)}_{ik} P^{(n)}_{kj}$,
which is just matrix multiplication associativity in disguise.)</p>

<p>If $\\mu_0$ is the initial distribution (a row vector), then the distribution
at time $t$ is</p>

$$\\boxed{\\mu_t \\;=\\; \\mu_0 \\, P^t}$$

<p>— a row vector times a matrix, repeatedly.</p>

<p><strong>Powers of the weather P (pre-verified).</strong> Let's see what happens.</p>

<p>$P^1$:
$\\begin{pmatrix} 0.70 & 0.20 & 0.10 \\\\ 0.30 & 0.40 & 0.30 \\\\ 0.20 & 0.30 & 0.50 \\end{pmatrix}$</p>

<p>$P^2$:
$\\begin{pmatrix} 0.57 & 0.25 & 0.18 \\\\ 0.39 & 0.31 & 0.30 \\\\ 0.33 & 0.31 & 0.36 \\end{pmatrix}$</p>

<p>$P^5$:
$\\begin{pmatrix} 0.468 & 0.279 & 0.252 \\\\ 0.450 & 0.284 & 0.266 \\\\ 0.443 & 0.286 & 0.271 \\end{pmatrix}$</p>

<p>$P^{20}$:
$\\begin{pmatrix} 0.457 & 0.283 & 0.261 \\\\ 0.457 & 0.283 & 0.261 \\\\ 0.457 & 0.283 & 0.261 \\end{pmatrix}$</p>

<p>Two striking things. <strong>First</strong>, the rows of $P^n$ get <em>identical</em> as $n$
grows — meaning the long-run probability of being in state $j$ no longer
depends on where you started. <strong>Second</strong>, that common row is some fixed
distribution — $(0.457, 0.283, 0.261)$. This distribution is what we'll call
<strong>stationary</strong> in §4. The conditions under which this convergence happens
are exactly the content of §3 and §5.</p>

<p>A different shape of behaviour. Consider the two-state chain with
$P = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}$. Then $P^2 = I$, $P^3 = P$,
$P^4 = I$, and so on. The chain oscillates forever; $P^n$ never converges.
Yet $\\pi = (0.5, 0.5)$ is a stationary distribution in the sense of §4. So
"stationary distribution exists" and "the chain converges to it" are <em>two
different statements</em>. Hold that distinction.</p>

<p>Drag $n$ and watch the heatmap of $P^n$. On the weather tab the rows fuse into
$\\pi$; switch to the periodic tab and $P^n$ flips between $I$ and the swap
matrix forever — the contrast is the whole point.</p>

<power-of-p-animator></power-of-p-animator>`,
    );
  },
};
