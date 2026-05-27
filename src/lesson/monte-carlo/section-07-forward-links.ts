import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection07: Section = {
  id: "mc-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§7</span>Where You'll See This Again</h2>
<p class="tagline"><em>Three downstream lessons will build directly on MC.</em></p>

<p>Three subsequent lessons depend directly on this one.</p>

<p><strong>Lesson 8 (Temporal-Difference Learning)</strong> is the immediate next step. TD
methods are presented as a "what if we bootstrap?" variant of MC. The opening of
Lesson 8 explicitly compares the MC update ($G_t$ as the target) with the TD(0)
update ($r + \\gamma \\hat V^\\pi(s')$ as the target), and develops $n$-step TD as the
family of methods between them. The first half of Lesson 8 is essentially "MC, but
with bootstrapping," and the conventions of MC policy evaluation and control
transfer with minor modifications. The forward-link is the strongest in the
curriculum.</p>

<p><strong>Lesson 9 (Function Approximation and Deep Q-Networks)</strong> uses MC returns as
targets in some of its training regimes. The "deadly triad" discussion explicitly
contrasts MC (which sidesteps the triad because it does not bootstrap) with TD
(which falls into it). The MC return appears as a baseline target whose stability is
the standard against which the TD methods' instability is judged.</p>

<p><strong>Lesson 10 (Policy Gradient Methods)</strong> uses the MC return as the unbiased
estimate of $Q^\\pi(s, a)$ in the simplest REINFORCE gradient. The policy gradient
theorem requires <em>some</em> estimate of the return; the MC return is the simplest
unbiased one. Variance reduction techniques in Lesson 10 (baselines, advantage
functions, GAE) are all aimed at reducing the variance of this MC estimate while
keeping it unbiased or low-bias.</p>

<p><strong>Lesson 15 (Offline RL)</strong> uses off-policy MC as one baseline method. The
variance problem we documented in Section 5 is the central obstacle of offline RL,
and many of the algorithms in Lesson 15 (BCQ, CQL, behavior cloning, density-ratio
estimation) are working around exactly the issue we surfaced here. The "39 effective
samples out of 10,000" gridworld diagnostic from Section 5 is the canonical
illustration of why pure off-policy MC fails in the offline setting.</p>

<div class="component-host">
  <roadmap-mini active="monte-carlo"></roadmap-mini>
</div>
`);
  },
};
