import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection02: Section = {
  id: "entropy-regularized-objective",
  title: "The Entropy-Regularized Objective",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§2</span>The Entropy-Regularized Objective</h2>
<p class="tagline"><em>Reward plus entropy, weighted by $\\alpha$.</em></p>

<p>We define the objective formally:</p>

$$J_\\alpha(\\pi) = \\mathbb{E}^\\pi\\!\\left[
  \\sum_{t=0}^{\\infty} \\gamma^t \\bigl( r(s_t, a_t) + \\alpha\\, \\mathcal{H}(\\pi(\\cdot | s_t)) \\bigr)
\\right]$$

<p>where $\\mathcal{H}(p) = -\\sum_a p(a) \\log p(a)$ is the Shannon entropy in nats,
and $\\alpha \\geq 0$ is the <strong>temperature</strong>. The objective is the expected
discounted sum of two things: reward (the familiar RL objective) and entropy of the
policy at each visited state.</p>

<p>The parameter $\\alpha$ converts nats into reward-units — a larger $\\alpha$ means
a single nat of entropy is "worth" more reward. Two limits clarify:</p>

<ul>
<li><strong>$\\alpha = 0$</strong>: only reward matters. The objective is standard RL.
  The optimal policy is deterministic.</li>
<li><strong>$\\alpha \\to \\infty$</strong>: only entropy matters. The optimal policy
  maximizes per-state entropy subject to the dynamics — for a finite action space this
  is the uniform policy. (At intermediate $\\alpha$ on episodic tasks, we'll see in §5
  that the agent prefers to <em>not</em> terminate, because terminating forfeits future
  entropy.)</li>
</ul>

<p>At intermediate $\\alpha$, the policy is stochastic — sharper than uniform in states
where one action is much better, closer to uniform in states where actions are nearly tied.
The exact shape we derive in §3 and §4.</p>

<h3>Two subtleties</h3>

<p><strong>State-conditional entropy, not trajectory entropy.</strong> The entropy term
$\\mathcal{H}(\\pi(\\cdot|s))$ is the entropy of the agent's action distribution given
state $s$ — it is what the agent <em>controls</em>. Trajectory entropy also includes
contributions from environment stochasticity, which the agent cannot control. Lesson 11's
PPO entropy bonus was the same thing; SAC (Lesson 13) uses it too. Lesson 17's RLHF
swaps this term for a KL penalty to a reference policy.</p>

<p><strong>Discount $\\gamma$ applies to both terms.</strong> The standard formulation
discounts the entropy bonus by $\\gamma^t$ alongside the reward. This is what makes
the objective tractable (it reduces to a Bellman equation). An undiscounted entropy term
is a different (less standard) formulation.</p>

<p>The visualization below shows the single-state case — a 1-step bandit with two actions
at reward difference $\\Delta r = r_1 - r_2$. As $\\alpha$ increases, the optimal mixing
probability migrates from "all-in on the better action" toward 50/50.</p>

<objective-surface></objective-surface>
`);
  },
};
