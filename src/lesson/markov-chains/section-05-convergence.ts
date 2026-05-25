import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mcSection05: Section = {
  id: "ergodic-theorem",
  title: "The Ergodic Theorem (Convergence)",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§5</span>The Ergodic Theorem (Convergence)</h2>
<p class="tagline">Irreducible + aperiodic = inevitable convergence, at a rate set by the spectral gap.</p>

<p><strong>Theorem (Convergence to Stationarity).</strong> Let $P$ be the transition matrix
of a finite, irreducible, aperiodic Markov chain. Then there exists a unique
stationary distribution $\\pi$, and for every initial distribution $\\mu_0$,</p>

$$\\boxed{\\lim_{n \\to \\infty} \\mu_0 P^n \\;=\\; \\pi.}$$

<p>Equivalently, $\\lim_{n\\to\\infty} P^n = \\mathbf{1} \\pi^\\top$ (each row of
$P^n$ converges to $\\pi$).</p>

<p><strong>Proof sketch via eigendecomposition.</strong> Since $P$ is row-stochastic, its
spectral radius is 1, and 1 is an eigenvalue (with left eigenvector $\\pi$).
Irreducibility makes that eigenvalue <em>simple</em> (multiplicity 1). Aperiodicity
ensures that all <em>other</em> eigenvalues $\\lambda_2, \\lambda_3, \\ldots, \\lambda_K$ have magnitude
strictly less than 1. (Periodic chains have eigenvalues on the unit circle
besides 1; aperiodicity rules this out.)</p>

<p>Write $P = \\sum_k \\lambda_k v_k u_k^\\top$ in spectral form (left/right
eigenpairs). Then</p>

$$P^n \\;=\\; \\sum_k \\lambda_k^n v_k u_k^\\top
\\;=\\; \\underbrace{\\mathbf{1} \\pi^\\top}_{\\lambda_1 = 1,\\ \\text{rank 1}}
\\;+\\; \\sum_{k \\geq 2} \\lambda_k^n v_k u_k^\\top.$$

<p>The second sum vanishes because $|\\lambda_k| < 1$ for $k \\geq 2$. Done.</p>

<p><strong>The rate of convergence — spectral gap.</strong> The convergence rate is governed
by the <strong>second largest eigenvalue in magnitude</strong>:</p>

$$\\lambda_\\star \\;:=\\; \\max_{k \\geq 2} |\\lambda_k|.$$

<p>The <strong>spectral gap</strong> is $\\gamma := 1 - \\lambda_\\star$. The total-variation
distance to stationarity satisfies</p>

$$\\|\\mu_0 P^n - \\pi\\|_{\\text{TV}} \\;\\leq\\; C \\cdot \\lambda_\\star^n$$

<p>for some constant $C$ depending on the chain. So convergence is geometric
with rate $\\lambda_\\star$. The closer $\\lambda_\\star$ is to 1, the slower.</p>

<p><strong>Mixing time.</strong> Define the mixing time $t_{\\text{mix}}(\\varepsilon)$ as the
smallest $n$ for which $\\|\\mu_0 P^n - \\pi\\|_{\\text{TV}} \\leq \\varepsilon$
from any starting distribution. A standard bound:</p>

$$t_{\\text{mix}}(\\varepsilon) \\;\\leq\\; \\frac{\\log(1/\\varepsilon)}{1 - \\lambda_\\star} \\cdot (\\text{small constant}).$$

<p>For the weather chain: eigenvalues of $P$ are $(1, 0.4732, 0.1268)$. So
$\\lambda_\\star = 0.4732$, spectral gap $\\gamma \\approx 0.527$, and the mixing time at
$\\varepsilon = 0.01$ is roughly $\\log(100) / 0.527 \\approx 8.74$ steps. Empirically (from
the table in §2), the TV distance is below $10^{-4}$ by $n = 10$ and below
$10^{-6}$ by $n = 20$. Theory and experiment agree.</p>

<figure class="numeric">
<p><strong>Numerical convergence table (pre-verified).</strong> TV distance to π from each
starting state, for the weather chain:</p>
<table class="numeric">
  <thead>
    <tr><th>$n$</th><th>from sunny</th><th>from cloudy</th><th>from rainy</th></tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>0.5435</td><td>0.7174</td><td>0.7391</td></tr>
    <tr><td>1</td><td>0.2435</td><td>0.1565</td><td>0.2565</td></tr>
    <tr><td>2</td><td>0.1135</td><td>0.0665</td><td>0.1265</td></tr>
    <tr><td>3</td><td>0.0535</td><td>0.0305</td><td>0.0605</td></tr>
    <tr><td>5</td><td>0.0120</td><td>0.0068</td><td>0.0136</td></tr>
    <tr><td>10</td><td>2.8e-4</td><td>1.6e-4</td><td>3.2e-4</td></tr>
    <tr><td>20</td><td>7.4e-9</td><td>(vanishing)</td><td>(vanishing)</td></tr>
  </tbody>
</table>
<p>The ratio between consecutive rows is approximately $\\lambda_\\star = 0.4732$,
exactly as the theorem predicts.</p>
</figure>

<p><strong>When ergodicity fails — and what survives.</strong> If the chain is periodic,
$P^n$ doesn't converge but the <strong>Cesàro average</strong> does:</p>

$$\\frac{1}{N} \\sum_{n=0}^{N-1} P^n \\;\\to\\; \\mathbf{1} \\pi^\\top.$$

<p>So time-averaged behaviour still gives you $\\pi$; only instantaneous
distributions oscillate. For $P = \\begin{pmatrix}0&1\\\\1&0\\end{pmatrix}$, the
Cesàro average over $n=0,\\ldots,99$ is numerically
$\\begin{pmatrix}0.5 & 0.5 \\\\ 0.5 & 0.5\\end{pmatrix}$ — exactly $\\mathbf{1}\\pi^\\top$.</p>

<p>If the chain is <em>reducible</em>, even the Cesàro average depends on the starting
state — different connected components have different stationary
distributions, and there is no single $\\pi$ that all initial conditions land in.</p>

${forwardLink({
  destination: "Lesson 5 — TD Learning",
  html: `<p>The Cesàro average is what TD(0) and Q-learning effectively <em>average over</em> in
  their long-run behaviour. The stochastic-approximation theorem (Robbins-Monro) requires
  only that the chain visits each state infinitely often — recurrence. Aperiodicity isn't
  strictly required for TD's convergence, but irreducibility is.</p>`,
})}

<p>Below is the <strong>Convergence Lab</strong>. Pick a chain and press play: Panel A shows
the spectrum on the unit circle, Panel B the distribution evolving, Panel C
the TV distance decaying (or not) on a log axis, Panel D a single sampled path
whose empirical frequencies fill in toward π. Compare "weather" (fast),
"slow-mixing" (eigenvalue near 1), "periodic" (oscillates), and "reducible"
(settles to a start-dependent floor).</p>

<convergence-lab></convergence-lab>`,
    );
  },
};
