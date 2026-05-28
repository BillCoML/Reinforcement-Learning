import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection01: Section = {
  id: "policy-gradient-motivation",
  title: "From Values to Policies",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>From Values to Policies</h2>
<p class="tagline"><em>Why parameterize the policy directly? Three reasons that motivate the field.</em></p>

<p>Lessons 8 and 9 built value-based RL: learn $Q(s, a)$, act greedily.
The recipe works beautifully on discrete-action problems with sufficient data.
But it has three structural limitations that motivate a different approach.</p>

<p><strong>Limitation 1: Continuous action spaces.</strong>
Q-learning's policy is $\\pi(s) = \\arg\\max_{a} Q(s, a)$.
For continuous $\\mathcal{A}$, the $\\arg\\max$ is an optimization problem at every step.
The direct approach: parameterize the policy itself, never take an $\\arg\\max$.</p>

<p><strong>Limitation 2: Stochastic policies.</strong>
Q-learning's induced policy is deterministic (greedy).
In partial observability, multi-agent settings, and adversarial environments,
the optimal policy is genuinely stochastic.
Policy gradient methods natively parameterize stochastic policies.</p>

<p><strong>Limitation 3: Direct optimization.</strong>
Value-based RL is indirect — we optimize a $Q$-function and hope the greedy policy is good.
The mapping from $Q$-error to policy error can be discontinuous.
Policy gradient is direct: we optimize the policy against a smooth performance metric.</p>

<p>From this lesson forward, the central object is
$\\pi_\\theta(a \\mid s)$ — a parameterized distribution over actions — with parameters $\\theta$ updated by gradient ascent on the expected return:</p>

$$J(\\theta) \\;:=\\; \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[ \\sum_{t=0}^{T-1} \\gamma^t r_{t+1} \\right].$$

<p>A common misconception is that policy gradient methods abandon value functions.
They don't. Most PG methods use a value function as a baseline (§5),
as a critic in actor-critic (§6), or as the source of an advantage signal (§7).
The TD machinery from Lesson 8 is fully reused — the critic in actor-critic is exactly TD(0).</p>

<pg-from-q-values-to-policies></pg-from-q-values-to-policies>
`);
  },
};
