import type { Section } from '../section';
import { sectionFromHTML } from '../section';
import { forwardLink } from '../../components/CrosslinkCallout';

export const dpSection02: Section = {
  id: 'iterative-policy-evaluation',
  title: 'Iterative Policy Evaluation',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§2</span>Iterative Policy Evaluation</h2>
<p class="tagline">Apply T<sup>π</sup> until V<sup>π</sup> emerges.</p>

<p>Given a policy $\\pi$, compute $V^\\pi$. Two approaches.</p>

<p><strong>Approach 1 — Direct solve.</strong> From Lesson 2 §6:</p>

$$V^\\pi = (I - \\gamma P^\\pi)^{-1} R^\\pi.$$

<p>For our 9-state gridworld this is a 9×9 linear solve — instant. For an MDP with
$|\\mathcal{S}| = 10^6$, this is hopeless: forming $(I - \\gamma P^\\pi)$ requires $10^{12}$
entries, and inverting it costs $10^{18}$ operations.</p>

<p><strong>Approach 2 — Iterative.</strong> Apply the Bellman expectation operator $T^\\pi$ repeatedly:</p>

$$\\boxed{V_{k+1}(s) := \\sum_a \\pi(a|s) \\left[ r(s,a) + \\gamma \\sum_{s'} P(s'|s,a)\\, V_k(s') \\right]}$$

<p>Start with $V_0 \\equiv 0$. Prereq C proved $T^\\pi$ is a $\\gamma$-contraction in the
sup-norm. By Banach, $V_k \\to V^\\pi$ geometrically at rate $\\gamma$.</p>

<h3>Stopping Criterion</h3>

<p>From Prereq C's <em>a posteriori</em> bound:</p>

$$\\|V_k - V^\\pi\\|_\\infty \\;\\leq\\; \\frac{\\gamma}{1 - \\gamma} \\|V_k - V_{k-1}\\|_\\infty.$$

<p>So to guarantee $\\|V_k - V^\\pi\\|_\\infty &lt; \\epsilon$, iterate until</p>

$$\\boxed{\\|V_k - V_{k-1}\\|_\\infty \\;&lt;\\; \\frac{(1 - \\gamma)\\,\\epsilon}{\\gamma}.}$$

<p>For $\\gamma = 0.9,\\, \\epsilon = 0.01$, this means stopping when consecutive iterates
differ by less than $0.00111$. The number of iterations needed scales as
$O(\\log(1/\\epsilon) / (1 - \\gamma))$ — linear in $1/(1-\\gamma)$, which is brutal as
$\\gamma \\to 1$.</p>

<h3>Cost Comparison</h3>

<p>Each $T^\\pi$ application is $O(|\\mathcal{S}|^2)$ for dense $P^\\pi$. Total cost for
iterative PE: $O(|\\mathcal{S}|^2 \\cdot \\log(1/\\epsilon) / (1-\\gamma))$.</p>

<p>Compare to direct solve at $O(|\\mathcal{S}|^3)$. <strong>Iterative wins for large
$|\\mathcal{S}|$ and small $1/(1-\\gamma)$.</strong></p>

<h3>Numerical Example</h3>

<p>Iterative PE for the uniform random policy on our gridworld, starting $V_0 \\equiv 0$:</p>

<div class="dp-table-wrap">
<table class="dp-table">
  <thead><tr><th>$k$</th><th>$V_k(0,0)$</th><th>$\\|V_k - V_{k-1}\\|_\\infty$</th></tr></thead>
  <tbody>
    <tr><td>0</td><td>0.0000</td><td>—</td></tr>
    <tr><td>1</td><td>0.0000</td><td>0.2500</td></tr>
    <tr><td>5</td><td>−0.2877</td><td>0.0796</td></tr>
    <tr><td>10</td><td>−0.3859</td><td>0.0252</td></tr>
    <tr><td>20</td><td>−0.4180</td><td>0.0026</td></tr>
    <tr><td>50</td><td>−0.4205</td><td>8.6×10<sup>−6</sup></td></tr>
    <tr><td>∞</td><td>−0.4205</td><td>(limit)</td></tr>
  </tbody>
</table>
</div>

<p>The convergence is geometric at rate $\\gamma = 0.9$: errors shrink by ~10× every ~22
iterations.</p>

${forwardLink({
  destination: 'Lesson 4 — Monte Carlo',
  html: `<p>In Lesson 4 we'll estimate $V^\\pi$ from sampled trajectories, <em>without</em>
    knowing $P$ or $r$. In Lesson 5 (TD) we'll do the same with one-step samples
    instead of full trajectories. Both are model-free versions of <em>this</em> iteration.</p>`,
})}

<iterative-pe-watcher></iterative-pe-watcher>`);
  },
};
