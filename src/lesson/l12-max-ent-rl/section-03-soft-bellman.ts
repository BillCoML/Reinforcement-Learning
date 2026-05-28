import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection03: Section = {
  id: "soft-bellman",
  title: "The Soft Bellman Operator",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>The Soft Bellman Operator</h2>
<p class="tagline"><em>Logsumexp where the max used to be.</em></p>

<p>The Bellman optimality operator from Lesson 3 was:</p>

$$(T V)(s) = \\max_a\\, \\bigl[ r(s,a) + \\gamma\\, \\mathbb{E}_{s'}[V(s')] \\bigr].$$

<p>For the entropy-regularized objective, the analogous operator is:</p>

$$(T_\\alpha V)(s) = \\alpha \\log \\sum_a \\exp\\!\\left(
  \\frac{r(s,a) + \\gamma\\, \\mathbb{E}_{s'}[V(s')]}{\\alpha}
\\right).$$

<p>The $\\max$ has become a <strong>logsumexp at temperature $\\alpha$</strong>. This is the
<strong>soft Bellman operator</strong>. Two limits make the connection transparent:</p>

<ul>
<li>As $\\alpha \\to 0$, the logsumexp concentrates on its largest argument:
  $\\alpha \\log \\sum \\exp(x/\\alpha) \\to \\max x$. So $T_\\alpha \\to T$. Standard Bellman.</li>
<li>As $\\alpha \\to \\infty$, the logsumexp washes out differences and approaches
  $\\alpha \\log |A| + \\bar{Q}$ where $\\bar{Q}$ is the average Q-value. The operator averages.</li>
</ul>

<h3>Derivation of the Boltzmann form</h3>

<p>The entropy-regularized Bellman equation is:</p>

$$V_\\alpha^*(s) = \\max_\\pi\\, \\mathbb{E}_{a \\sim \\pi}\\!\\left[
  Q_\\alpha^*(s,a) + \\alpha\\, \\mathcal{H}(\\pi(\\cdot|s))
\\right],$$

<p>where $Q_\\alpha^*(s,a) = r(s,a) + \\gamma \\mathbb{E}_{s'}[V_\\alpha^*(s')]$.
Maximizing over $\\pi$ subject to the simplex constraint $\\sum_a \\pi(a|s) = 1$
via a Lagrangian gives the <strong>Boltzmann policy</strong>:</p>

$$\\pi_\\alpha^*(a|s) = \\frac{\\exp(Q_\\alpha^*(s,a) / \\alpha)}{\\sum_{a'} \\exp(Q_\\alpha^*(s,a') / \\alpha)},$$

<p>and substituting back yields exactly $V_\\alpha^*(s) = \\alpha \\log \\sum_a \\exp(Q_\\alpha^*(s,a)/\\alpha)$
— the logsumexp form above.</p>

<h3>Contraction and convergence</h3>

<p>Is $T_\\alpha$ still a contraction? Yes. The soft Bellman operator is a
$\\gamma$-contraction in sup norm:</p>

$$\\bigl\\| T_\\alpha V - T_\\alpha V' \\bigr\\|_\\infty \\leq \\gamma \\|V - V'\\|_\\infty.$$

<p>The inequality uses that the logsumexp at any temperature is Lipschitz-1 in its
arguments (a standard fact from the analysis of log-sum-exp). The $\\gamma$ factor
comes from the $\\gamma \\mathbb{E}[V(s')]$ term inside the argument.</p>

<p>Lesson 4 established that $\\gamma$-contractions have unique fixed points; that
result transfers directly. <strong>Soft VI converges. The fixed point is unique.
The fixed point is $V_\\alpha^*$.</strong> Nothing structural changes — only what
fixed point you converge to.</p>

<h3>Soft value iteration as an algorithm</h3>

<p>The algorithm is a direct modification of hard value iteration: initialize $V = 0$,
repeatedly apply $T_\\alpha$, stop when $\\|V_{k+1} - V_k\\|_\\infty < \\varepsilon$.
The only change is replacing $\\max_a Q(s,a)$ with $\\alpha \\log \\sum_a \\exp(Q(s,a)/\\alpha)$.
For numerical stability, use the shifted form:</p>

$$\\alpha \\log \\sum_a \\exp\\!\\left(\\frac{Q(s,a) - \\max_{a'} Q(s,a')}{\\alpha}\\right) + \\max_{a'} Q(s,a').$$

<p>This keeps all $\\exp()$ arguments in $(-\\infty, 0]$, preventing overflow at small $\\alpha$
where $Q/\\alpha$ can reach 700+.</p>

<soft-vi-convergence></soft-vi-convergence>
`);
  },
};
