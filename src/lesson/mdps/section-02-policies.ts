import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { sidebar } from "../../components/CrosslinkCallout";

export const mdpSection02: Section = {
  id: "policy",
  title: "Policies",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§2</span>Policies</h2>
<p class="tagline">A policy is a rule for living.</p>

<p>The agent doesn't get to control transitions directly; they're set by $P$. What the
agent <em>does</em> control is how it chooses actions. A <strong>policy</strong> specifies this choice.</p>

<p><strong>Deterministic policy.</strong> A map $\\pi: \\mathcal{S} \\to \\mathcal{A}$. From state $s$,
the agent takes action $\\pi(s)$, always.</p>

<p><strong>Stochastic policy.</strong> A conditional distribution $\\pi(\\cdot \\mid s)$ over actions,
given the state. From state $s$, the agent samples $a \\sim \\pi(\\cdot \\mid s)$. Deterministic
policies are the special case where the distribution is a delta.</p>

<p>Two policies on our gridworld worth keeping in mind:</p>
<ul>
  <li><strong>Uniform random:</strong> $\\pi(a \\mid s) = 1/4$ for all $(s, a)$. The agent flails.
  Our worst-realistic baseline.</li>
  <li><strong>An optimal policy:</strong> at $(0,0)$ go right; at $(0,1)$ go right; at $(0,2)$ go
  down; and so on. There are actually <em>multiple</em> optimal policies here (two symmetric
  shortest paths around the pit). We'll compute one in §7.</li>
</ul>

<p>Why ever consider <em>stochastic</em> policies if a deterministic one suffices? Three reasons
that will matter downstream.</p>
<ol>
  <li><strong>Exploration.</strong> A deterministic policy in an unknown MDP will never discover
  state-actions it has never tried. Adding randomness — ε-greedy (from Bandits), Boltzmann,
  Gaussian noise — is the simplest way to keep exploring. Lessons 5 and 7 lean on this.</li>
  <li><strong>Differentiability.</strong> When we parameterize policies as neural networks
  $\\pi_\\theta(a \\mid s)$ and want $\\nabla_\\theta J(\\theta)$, the gradient must flow through the
  action distribution. A deterministic argmax has zero gradient almost everywhere; a
  stochastic policy does not. Lesson 8 (policy gradient) cannot exist without stochastic
  policies.</li>
  <li><strong>Optimality under partial observability and adversarial settings.</strong> Against an
  adversary or under partial observation, the <em>optimal</em> policy can be genuinely stochastic
  (think rock-paper-scissors). Even in fully-observable MDPs, optimal <em>exploration</em>
  policies are stochastic.</li>
</ol>

<p>For finite MDPs with full observability — the setting of this entire lesson — there
<em>always exists</em> a deterministic optimal policy. We'll prove this in §7. So if you only
care about exploitation, deterministic is enough. The three reasons above explain why every
lesson from Lesson 7 onward will use stochastic policies anyway.</p>

${sidebar(
  "Back-link → Bandits",
  `<p>Bandits used stochastic policies for exploration reason 1 (ε-greedy, Thompson). The
  other two reasons didn't apply because there was no gradient to compute and no adversary.</p>`,
)}

<p>In the explorer below, arrow opacity is proportional to $\\pi(a \\mid s)$: uniform random
shows four equally-faded arrows; a deterministic policy shows one solid arrow. Build a
<em>custom</em> policy by bumping action weights on a selected cell, then “Save” it — V4 and V6
will evaluate the very policy you saved.</p>

<policy-explorer></policy-explorer>`,
    );
  },
};
