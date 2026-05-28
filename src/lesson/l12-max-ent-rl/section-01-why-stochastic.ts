import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection01: Section = {
  id: "why-stochastic",
  title: "Why Stochasticity Might Be the Objective",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>Why Stochasticity Might Be the Objective</h2>
<p class="tagline"><em>The hard optimum isn't always the right target.</em></p>

<p>The standard RL objective is to find a policy that maximizes expected return:
$\\pi^* = \\arg\\max_\\pi \\mathbb{E}^\\pi[\\sum_t \\gamma^t r_t]$.
The optimal policy under this objective is almost always deterministic — for a finite
MDP, the optimal policy can always be chosen to be deterministic without loss.
On the $3\\times 3$ gridworld, $\\pi^*$ picks "right" with probability 1 in state $(0,0)$.
This is the answer Lessons 3, 5, and 8 all produced. And it achieves $V^*(0,0) = 0.7290$.</p>

<p>But: <em>Is determinism actually what we want?</em></p>

<p>We make three cases that it might not be.</p>

<p><strong>Robustness.</strong> A deterministic policy is brittle. If any transition probability
differs slightly from the agent's model, or if the reward has measurement noise, the
deterministic policy may be far from optimal under the true dynamics. A stochastic policy
has built-in robustness — it hedges by sampling neighboring actions.</p>

<p><strong>Exploration.</strong> Lessons 1–10 treated exploration as a separate concern
($\\varepsilon$-greedy, decaying schedules, GLIE conditions). A stochastic optimal
policy folds exploration into the objective itself: the policy never collapses to a single
action, so it <em>inherently</em> explores.</p>

<p><strong>Multimodality.</strong> If two actions are nearly equally good, a deterministic
policy must pick one — the choice is arbitrary and brittle. A stochastic policy
acknowledges near-ties and hedges.</p>

<h3>The Lesson 10 softmax cap, reframed</h3>

<p>In Lesson 10, REINFORCE on the gridworld converged to $V^\\pi(0,0) \\approx 0.722$
rather than $V^* = 0.729$. In Lesson 11 we called this the "softmax cap" and attributed
it to the finite-temperature softmax representation. Now we give the correct explanation:</p>

<p class="crosslink-callout">
<strong>The softmax cap is not an artifact.</strong> The converged REINFORCE policy is the
<em>exactly correct</em> soft-optimal policy at a small temperature. Specifically, at
$\\alpha = 0.02$, the entropy-regularized optimal policy gives $V^\\pi(0,0) = 0.7217$.
The cap is the right answer to a slightly different question.
</p>

<p>The visualization below plots $V^\\pi(0,0)$ — the true expected discounted return
under the soft-optimal policy — across the full range of temperatures. At $\\alpha = 0$
it equals $V^* = 0.729$. At $\\alpha = 0.02$ it passes through the L10 cap (0.722).
As $\\alpha$ grows, the policy becomes increasingly diffuse and $V^\\pi$ falls sharply.</p>

<hard-soft-spectrum></hard-soft-spectrum>

<p>The shaded "Useful" band is where stochasticity helps without destroying task performance.
The shaded "Failure" band (red, $\\alpha \\gtrsim 0.15$) is where the agent starts
avoiding the goal — a phenomenon we will explore in §5.</p>
`);
  },
};
