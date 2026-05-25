import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const contrSection06: Section = {
  id: "contraction-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§6</span>Where You'll See This Again</h2>
<p class="tagline">The contraction theorem powers the next four lessons, then keeps cashing in.</p>

<p>The Banach fixed-point theorem is <em>immediately</em> useful: Lesson 3 (next)
implements value iteration as $V_{k+1} := T^* V_k$, and the theorem is what makes
that loop terminate.</p>

<p>Five places downstream where contractions and fixed points reappear:</p>

<p><strong>1. Lesson 3 — Dynamic Programming.</strong> Value iteration is Banach iteration
on $T^*$. Policy iteration alternates Banach iteration on $T^\\pi$ (the evaluation
step) with strict policy improvement. The stopping criterion
$\\|V_k - V_{k-1}\\|_\\infty &lt; \\epsilon(1-\\gamma)/\\gamma$ is the <em>a posteriori</em>
bound from §3 with $c = \\gamma$. <em>Everything</em> in Lesson 3 is a direct deployment
of this prereq.</p>

<p><strong>2. Lesson 5 — TD Learning.</strong> Sample-based Bellman backups are <em>stochastic</em>
versions of $T^*$. They lose the deterministic contraction property and gain a
noisy-gradient interpretation. The Robbins-Monro theorem replaces Banach for this
setting. The structural similarity is unmistakable.</p>

<p><strong>3. Lesson 6 — Function Approximation.</strong> When $V$ is replaced by $V_\\theta$,
the projection of $T^* V_\\theta$ back onto the parameterized class can be
non-expansive but <em>not</em> a contraction. Failure of the contraction property is
what makes the "deadly triad" deadly.</p>

<p><strong>4. Lesson 9 — Trust Region Methods.</strong> TRPO's natural-gradient step solves a
constrained optimization where the constraint is
$\\|\\pi_{k+1} - \\pi_k\\|_{KL} \\leq \\delta$. The fixed-point intuition —
iterate until convergence; bound the per-step movement — is the same shape.</p>

<p><strong>5. Lesson 11 — RL as Probabilistic Inference.</strong> The soft Bellman operator
$T^{\\text{soft}} V := \\alpha \\log \\sum_a \\exp([r + \\gamma PV]/\\alpha)$ is also a
$\\gamma$-contraction. The proof is a few lines using LogSumExp's Lipschitz
property. Same theorem, shifted reward.</p>

<roadmap-mini active="contractions"></roadmap-mini>`);
  },
};
