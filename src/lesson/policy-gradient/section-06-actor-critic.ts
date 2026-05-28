import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection06: Section = {
  id: "actor-critic",
  title: "Actor-Critic Methods",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§6</span>Actor-Critic Methods</h2>
<p class="tagline"><em>Update both actor and critic per step. Trade some bias for much lower variance.</em></p>

<p>REINFORCE with baseline still uses full Monte Carlo returns — it waits until episode end
to update. One-step actor-critic makes a different trade: update both the policy (actor)
and the value estimate (critic) at <em>every step</em> using a TD target.</p>

<p>Per step, given transition $(s_t, a_t, r_{t+1}, s_{t+1})$:</p>

$$\\delta_t = r_{t+1} + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t) \\quad \\text{(TD error)}$$
$$V_\\phi(s_t) \\mathrel{+}= \\alpha_c \\cdot \\delta_t \\quad \\text{(critic update)}$$
$$\\theta \\mathrel{+}= \\alpha_a \\cdot \\delta_t \\cdot \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t) \\quad \\text{(actor update)}$$

<p>The TD error $\\delta_t$ serves simultaneously as the critic's learning target and
the actor's advantage estimate. This is the minimal actor-critic: one network for the policy,
one lookup table for the value function, both updated online.</p>

<h3>Bias-variance compared to REINFORCE</h3>

<p>REINFORCE with TD baseline uses the unbiased MC return $G_t$ but still updates only at
episode end. One-step actor-critic uses the biased TD target $r + \\gamma V(s')$ and updates
every step. The trade:</p>

<ul>
  <li>Faster learning per unit of experience (per-step updates, not per-episode).</li>
  <li>Bootstrapping introduces bias when $V_\\phi \\neq V^\\pi$ (early in training).</li>
  <li>In practice, the lower variance dominates — actor-critic typically converges faster
      than vanilla REINFORCE on the gridworld.</li>
</ul>

<p>This parallels the MC vs TD comparison from Lesson 7-8 exactly.
The narrative is the same: more bootstrapping = lower variance, more bias;
more MC = lower bias, more variance.</p>

<pg-policy-gradient-lab></pg-policy-gradient-lab>
`);
  },
};
