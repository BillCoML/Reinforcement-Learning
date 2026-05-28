import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection06: Section = {
  id: "entropy-slider-lab",
  title: "Entropy Slider Lab",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§6</span>Entropy Slider Lab</h2>
<p class="tagline"><em>The full picture in one slider.</em></p>

<p>The centerpiece of this lesson is an interactive explorer driven by a single global
temperature slider $\\alpha$. Pre-computed soft VI is loaded from approximately 61
logarithmically-spaced $\\alpha$ values; the slider snaps to the nearest pre-computed
point and updates all six panels synchronously.</p>

<p>Six panels together show what a single number — $\\alpha$ — does to the policy,
the values, the entropy, the goal-reaching probability, the trajectory lengths, and the
qualitative character of the agent's behavior. Move the slider slowly from left to right.
Watch Panel B: $V^\\pi(\\text{start})$ (green) and $V_\\text{soft}(\\text{start})$ (violet)
start together at $V^* = 0.729$ and then diverge as $\\alpha$ grows — the soft value
climbing (entropy bonus inflates it) while the true return collapses. Watch Panel D:
the goal-reach bar shrinks rapidly past $\\alpha \\approx 0.1$. Watch Panel F: the
caption changes character, naming the regime you're in.</p>

<entropy-slider-lab></entropy-slider-lab>

<h3>The L10 softmax cap</h3>

<p>The pink horizontal line in Panel B marks $V^\\pi = 0.7217$ — the L10 REINFORCE
softmax cap. It lies at $\\alpha \\approx 0.02$. The REINFORCE algorithm that "fell short"
of $V^*$ was actually converging to the correct fixed point of a slightly different
objective. The cap is not a failure of representation; it is the right answer to the
right question.</p>

<h3>The four regimes</h3>

<ul>
<li><strong>Tie-breaking ($\\alpha \\leq 0.01$)</strong>: essentially greedy, slight
  stochasticity where actions are nearly tied.</li>
<li><strong>Useful ($0.01 < \\alpha \\leq 0.05$)</strong>: goal-directed but explores.
  This is SAC territory.</li>
<li><strong>Trade-off ($0.05 < \\alpha \\leq 0.1$)</strong>: takes longer paths;
  entropy has a real efficiency cost.</li>
<li><strong>Failure ($\\alpha > 0.1$)</strong>: agent learns to avoid the goal.</li>
</ul>

<p>The key insight: these are not arbitrary human labels. They emerge from the objective.
At exactly $\\alpha \\approx 0.1$, the entropy bonus per step ($0.1 \\times \\log 4 \\approx 0.139$)
starts to become comparable to the per-step discounted value of reaching the goal.
The transition from "useful" to "failure" is a phase transition in the agent's behavior,
visible clearly in Panel D.</p>
`);
  },
};
