import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mcSection04: Section = {
  id: "stationary-distribution",
  title: "Stationary Distributions",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§4</span>Stationary Distributions</h2>
<p class="tagline">π solves π = πP. Existence is easy, uniqueness needs work.</p>

<p>A row vector $\\pi \\in \\mathbb{R}^K$ is a <strong>stationary distribution</strong> of $P$
if it is a probability distribution ($\\pi_i \\geq 0$, $\\sum_i \\pi_i = 1$) and
satisfies the <strong>balance equation</strong></p>

$$\\boxed{\\pi \\;=\\; \\pi P, \\qquad \\text{i.e.}\\quad \\pi_j \\;=\\; \\sum_i \\pi_i P_{ij} \\;\\;\\forall j.}$$

<p>Equivalently, $\\pi$ is a left eigenvector of $P$ with eigenvalue 1, normalized
to sum to 1. The intuition: if the chain is <em>currently</em> distributed as $\\pi$,
then after one step it's <em>still</em> distributed as $\\pi$. The mass flowing into
state $j$ (the right-hand side, $\\sum_i \\pi_i P_{ij}$) exactly equals the
mass currently in state $j$ (the left-hand side).</p>

<p><strong>Existence.</strong> Every finite Markov chain has at least one stationary
distribution. This is a consequence of the Perron-Frobenius theorem applied
to the row-stochastic matrix $P$: the spectral radius of $P$ is exactly 1,
and there exists a non-negative left eigenvector for eigenvalue 1, which can
be normalized to a probability distribution.</p>

<p><strong>Uniqueness.</strong> A finite chain has a <em>unique</em> stationary distribution if and
only if it is irreducible. Proof sketch: irreducibility implies that the
left-eigenspace of $P$ for eigenvalue 1 is one-dimensional; reducibility
allows multiple "absorbing" classes each with its own stationary support,
yielding a family of stationary distributions.</p>

<p><strong>Computing π.</strong> Three equivalent methods.</p>

<ol>
  <li><em>Linear algebra.</em> Solve $(P^\\top - I) \\pi^\\top = 0$ subject to
  $\\mathbf{1}^\\top \\pi^\\top = 1$. Set up the augmented system; standard
  linear solve.</li>
  <li><em>Power iteration.</em> Pick any initial distribution $\\mu_0$, compute
  $\\mu_t = \\mu_0 P^t$. If the chain is ergodic, $\\mu_t \\to \\pi$. Stop when
  $\\|\\mu_{t+1} - \\mu_t\\|$ is small.</li>
  <li><em>Eigendecomposition.</em> Find the left eigenvector of $P$ for eigenvalue 1
  directly.</li>
</ol>

<p><strong>Worked example.</strong> For the weather chain, the balance equations are:</p>

$$\\begin{aligned}
\\pi_S &= 0.7 \\pi_S + 0.3 \\pi_C + 0.2 \\pi_R \\\\
\\pi_C &= 0.2 \\pi_S + 0.4 \\pi_C + 0.3 \\pi_R \\\\
\\pi_R &= 0.1 \\pi_S + 0.3 \\pi_C + 0.5 \\pi_R \\\\
1     &= \\pi_S + \\pi_C + \\pi_R
\\end{aligned}$$

<p>Replace one balance equation (any of them — they're linearly dependent) with
the normalization. Solving yields</p>

$$\\pi \\;=\\; \\left(\\tfrac{21}{46},\\; \\tfrac{13}{46},\\; \\tfrac{12}{46}\\right) \\;\\approx\\; (0.4565,\\, 0.2826,\\, 0.2609).$$

<p>Numerically, this is exactly the row that $P^{20}$ converged to in §2. The
convergence wasn't accidental — it was inevitable.</p>

<p><strong>Periodic chain revisited.</strong> For $P = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}$,
the balance equation $\\pi = \\pi P$ gives $\\pi_0 = \\pi_1$, so $\\pi = (0.5, 0.5)$. This
exists and is unique (chain is irreducible). But $P^n$ doesn't converge to
the rank-one matrix $\\mathbf{1} \\pi^\\top = \\begin{pmatrix} 0.5 & 0.5 \\\\ 0.5 & 0.5 \\end{pmatrix}$ —
it oscillates. <strong>Stationary existence does not imply convergence.</strong> What we
need is the next section's content: aperiodicity.</p>

${forwardLink({
  destination: "Lesson 8 — Policy Gradient",
  html: `<p>We'll write the policy gradient as an expectation over the <em>stationary state
  distribution</em> $d^\\pi$ induced by policy $\\pi$. That $d^\\pi$ is exactly the stationary
  distribution of a Markov chain — the one whose transition matrix is $P^\\pi_{ss'} =
  \\sum_a \\pi(a|s)\\, p(s'|s,a)$. Same equation, same theorem, deeper application.</p>`,
})}

<p>Edit the matrix and compute π three ways. Power iteration animates $\\mu_t$
toward π (or, for a periodic chain, watch it refuse to settle); the verify
panel confirms $\\pi P = \\pi$ to machine precision.</p>

<stationary-distribution-finder></stationary-distribution-finder>`,
    );
  },
};
