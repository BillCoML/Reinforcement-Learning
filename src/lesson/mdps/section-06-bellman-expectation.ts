import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { sidebar } from "../../components/CrosslinkCallout";

export const mdpSection06: Section = {
  id: "bellman-expectation",
  title: "The Bellman Expectation Equations",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§6</span>The Bellman Expectation Equations</h2>
<p class="tagline">V^π satisfies a recursive identity. Iterate the identity, get the value.</p>

<p>We defined $V^\\pi$ as an expectation over infinite trajectories — fine for theory, useless
for computation. The <strong>Bellman expectation equation</strong> rewrites $V^\\pi$ as a
<em>recursion</em>: “one step + discounted rest.”</p>

<h3>Derivation</h3>
<p>Start with the definition and peel off the first reward:</p>

$$\\begin{aligned}
V^\\pi(s) &= \\mathbb{E}_\\pi\\!\\left[\\sum_{k=0}^\\infty \\gamma^k R_{t+k+1} \\,\\Big|\\, S_t = s\\right] \\\\[2pt]
&= \\mathbb{E}_\\pi\\!\\left[R_{t+1} + \\gamma\\, G_{t+1} \\,\\Big|\\, S_t = s\\right] \\\\[2pt]
&= \\mathbb{E}_\\pi\\!\\left[R_{t+1} \\mid S_t = s\\right] + \\gamma\\, \\mathbb{E}_\\pi\\!\\left[G_{t+1} \\mid S_t = s\\right].
\\end{aligned}$$

<p>The first term is $\\sum_a \\pi(a|s)\\, r(s,a)$. The second uses the <strong>tower property</strong>,
conditioning on $(A_t, S_{t+1})$, and then the Markov property:</p>

$$\\mathbb{E}_\\pi[G_{t+1} \\mid S_t = s] \\;=\\; \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)\\, V^\\pi(s').$$

<p>Combining everything gives the <strong>Bellman expectation equation</strong> for $V^\\pi$:</p>

$$\\boxed{V^\\pi(s) \\;=\\; \\sum_a \\pi(a \\mid s) \\!\\left[\\, r(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a)\\, V^\\pi(s') \\,\\right]}$$

<p>The corresponding equation for $Q^\\pi$ is</p>

$$\\boxed{Q^\\pi(s, a) \\;=\\; r(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a) \\sum_{a'} \\pi(a' \\mid s')\\, Q^\\pi(s', a')}$$

<h3 id="matrix-form-policy-evaluation">Matrix form</h3>
<p>Define the policy-induced transition matrix and reward vector</p>

$$(P^\\pi)_{s, s'} := \\sum_a \\pi(a \\mid s)\\, P(s' \\mid s, a), \\qquad (R^\\pi)_s := \\sum_a \\pi(a \\mid s)\\, r(s, a).$$

<p>Then the Bellman equation is linear:</p>

$$\\boxed{V^\\pi = R^\\pi + \\gamma P^\\pi V^\\pi \\;\\;\\Longleftrightarrow\\;\\; (I - \\gamma P^\\pi)\\, V^\\pi = R^\\pi.}$$

${sidebar(
  "Back-link → Markov Chains",
  `<p>$P^\\pi$ is exactly the policy-induced chain from Prereq A §7. Its stationary
  distribution $d^\\pi$ describes on-policy state visitation. <em>Same matrix, two uses.</em></p>`,
)}

<p>$(I - \\gamma P^\\pi)$ is invertible for any $\\gamma \\in [0,1)$ — the spectral radius of
$\\gamma P^\\pi$ is $\\gamma < 1$, so the Neumann series converges. Hence</p>

$$V^\\pi \\;=\\; (I - \\gamma P^\\pi)^{-1} R^\\pi.$$

<p>This is <strong>policy evaluation by direct solve</strong> — a $9\\times 9$ system for our gridworld,
instant. For $10^6$ states it's too big to invert; we'd use iterative methods (Lesson 3)
or sample-based methods (Lesson 5).</p>

<h3>Iterative interpretation</h3>
<p>Let $T^\\pi V := R^\\pi + \\gamma P^\\pi V$ be the <strong>Bellman operator</strong>. From any $V_0$
(often $0$), iterate $V_{k+1} = T^\\pi V_k$. Then $V_k \\to V^\\pi$ geometrically at rate $\\gamma$:</p>

$$\\|V_{k+1} - V^\\pi\\|_\\infty \\;\\leq\\; \\gamma \\|V_k - V^\\pi\\|_\\infty.$$

<p>$T^\\pi$ is a <strong>γ-contraction in the supremum norm</strong> ($P^\\pi$ is row-stochastic, so
multiplying by $\\gamma$ gives contraction factor $\\gamma$). By the Banach fixed-point theorem
$V^\\pi$ is the unique fixed point, and iteration converges to it from any start. This is the
same machinery that powers value iteration in Lesson 3, with $T^\\pi$ replaced by the
optimality operator $T^*$ (§7).</p>

<h3>Numerical demonstration (pre-verified)</h3>
<p>Iterating $T^\\pi$ for the uniform-random policy from $V_0 = 0$:</p>

<table class="numeric">
  <thead><tr><th>iteration $k$</th><th>$V_k(0,0)$</th></tr></thead>
  <tbody>
    <tr><td>0</td><td class="rl-mono">0.0000</td></tr>
    <tr><td>5</td><td class="rl-mono">−0.288</td></tr>
    <tr><td>10</td><td class="rl-mono">−0.386</td></tr>
    <tr><td>20</td><td class="rl-mono">−0.418</td></tr>
    <tr><td>∞</td><td class="rl-mono">−0.4205</td></tr>
  </tbody>
</table>

<p>It takes about 100 iterations to reach machine precision; the geometric rate $\\gamma = 0.9$
is <em>visible</em>: errors shrink by ~10× every ~22 iterations.</p>

<h3>The Bellman Backup Lab</h3>
<p>This is the heart of the lesson. <strong>Step</strong> the backup and watch value ripple outward
from the terminal states; the convergence trace in Panel C is visibly geometric on the log
scale. Then flip the operator from <strong>expectation</strong> ($T^\\pi$) to <strong>optimality</strong> ($T^*$):
$V_k$ now converges to a <em>different</em> fixed point, $V^*$ instead of $V^\\pi$, and Panel D shows
the greedy policy emerging. That switch is exactly the setup for Lesson 3.</p>

<bellman-backup-lab></bellman-backup-lab>`,
    );
  },
};
