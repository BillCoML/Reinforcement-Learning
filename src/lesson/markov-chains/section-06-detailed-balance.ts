import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mcSection06: Section = {
  id: "detailed-balance",
  title: "Detailed Balance (Reversibility)",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§6</span>Detailed Balance (Reversibility)</h2>
<p class="tagline">A sufficient condition for stationarity, and a glimpse of MCMC.</p>

<p>Solving $\\pi = \\pi P$ for the stationary distribution is a system of $K$ linear
equations. Often we can shortcut this with a stronger structural condition.</p>

<p><strong>Detailed balance.</strong> A distribution $\\pi$ and a transition matrix $P$ are
said to satisfy <strong>detailed balance</strong> if</p>

$$\\boxed{\\pi_i P_{ij} \\;=\\; \\pi_j P_{ji} \\quad \\forall i, j.}$$

<p>This says the probability flow from $i$ to $j$ exactly equals the flow from
$j$ to $i$, <strong>pairwise</strong>. Detailed balance is <em>strictly stronger</em> than the
balance equation $\\pi = \\pi P$: if detailed balance holds for $\\pi$, then summing
both sides over $i$ gives $\\sum_i \\pi_i P_{ij} = \\pi_j \\sum_i P_{ji} = \\pi_j$,
so $\\pi = \\pi P$. The converse is false — many chains have stationary
distributions without detailed balance.</p>

<p>A chain admitting a detailed-balance distribution is called <strong>reversible</strong>.
Run such a chain in reverse time, and the statistics are identical.</p>

<p><strong>Why it's useful.</strong> Two reasons.</p>

<p><em>Reason 1: Verification is local.</em> Checking $\\pi = \\pi P$ requires verifying $K$
equations involving the whole distribution. Detailed balance is $K(K-1)/2$
pairwise checks — but each one involves only two entries of $\\pi$. If you can
<em>guess</em> $\\pi$ and verify detailed balance pairwise, you've found the stationary
distribution.</p>

<p><em>Reason 2: MCMC.</em> The Markov chain Monte Carlo (MCMC) methodology <em>engineers</em>
chains that have a target distribution $\\pi$ as their detailed-balance
distribution, then runs them to draw approximate samples from $\\pi$. The
Metropolis-Hastings acceptance rule is <em>defined</em> by enforcing detailed balance.
We won't dwell on MCMC here, but it's the most important application of
detailed balance outside of physics.</p>

<p><strong>Worked example.</strong> A 3-state birth-death chain:</p>

$$P = \\begin{pmatrix} 0.5 & 0.5 & 0 \\\\ 0.3 & 0.4 & 0.3 \\\\ 0 & 0.6 & 0.4 \\end{pmatrix}.$$

<p>Solving $\\pi = \\pi P$ (or guessing-and-verifying) gives
$\\pi = \\left(\\tfrac{6}{21},\\, \\tfrac{10}{21},\\, \\tfrac{5}{21}\\right) \\approx (0.2857, 0.4762, 0.2381)$.</p>

<p>Detailed balance check:</p>
<ul>
  <li>$\\pi_0 P_{01} = \\tfrac{6}{21} \\cdot 0.5 = \\tfrac{3}{21}$ vs
  $\\pi_1 P_{10} = \\tfrac{10}{21} \\cdot 0.3 = \\tfrac{3}{21}$ ✓</li>
  <li>$\\pi_1 P_{12} = \\tfrac{10}{21} \\cdot 0.3 = \\tfrac{3}{21}$ vs
  $\\pi_2 P_{21} = \\tfrac{5}{21} \\cdot 0.6 = \\tfrac{3}{21}$ ✓</li>
  <li>$\\pi_0 P_{02} = 0$ vs $\\pi_2 P_{20} = 0$ ✓ (trivially)</li>
</ul>

<p>All pairs satisfy detailed balance — the chain is reversible.</p>

${forwardLink({
  destination: "Lesson 17 — RLHF & Preference Models",
  html: `<p>Detailed balance shows up again in a subtle way: the Bradley-Terry preference model
  gives a <em>symmetric</em> structure to pairwise comparison "transitions" that mirrors the
  reversibility condition. We'll point back here when we get there.</p>`,
})}

<p>Switch between the reversible birth-death chain and an asymmetric one.
"Compute π" sizes each arrow by its flow $\\pi_i P_{ij}$; "Check detailed
balance" marks every pair ✓ or ⚠.</p>

<detailed-balance-flow></detailed-balance-flow>`,
    );
  },
};
