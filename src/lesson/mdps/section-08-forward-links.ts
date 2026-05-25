import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection08: Section = {
  id: "mdps-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§8</span>Where You'll See This Again</h2>
<p class="tagline">Every algorithm in the RL canon solves, generalizes, or learns one of these equations.</p>

<p>The MDP and its Bellman equations sit at the center of everything that follows. A field
map:</p>

<ul>
  <li><strong>Lesson 3 (Dynamic Programming):</strong> iterate the operators when the model is known.
  Policy iteration alternates evaluation $V^\\pi = (I-\\gamma P^\\pi)^{-1}R^\\pi$ and improvement
  $\\pi'(s) = \\arg\\max_a Q^\\pi(s,a)$; value iteration applies $T^*$ to convergence. Both are
  <em>exact</em>, requiring full $(P, r)$.</li>
  <li><strong>Lesson 4 (Monte Carlo):</strong> estimate $V^\\pi$ by averaging observed returns $G_t$
  across trajectories — no Bellman equation at all. Robust, model-free, slow.</li>
  <li><strong>Lesson 5 (TD Learning):</strong> sampled Bellman backups — replace
  $\\sum_{s'} P(s'|s,a)V(s')$ with the single sample $r + \\gamma V(s')$. The single most
  important idea in classical RL.</li>
  <li><strong>Lesson 6 (Function Approximation):</strong> replace tabular $V$ with $V_\\theta$. The
  Bellman equations now hold only approximately; the “deadly triad” can diverge.</li>
  <li><strong>Lesson 7 (DQN):</strong> learn $Q_\\theta \\approx Q^*$ from samples, minimizing the squared
  Bellman-optimality residual $(r + \\gamma \\max_{a'} Q_{\\bar\\theta}(s',a') - Q_\\theta(s,a))^2$.</li>
  <li><strong>Lesson 8 (Policy Gradient):</strong> learn $\\pi_\\theta$ directly;
  $\\nabla_\\theta J = \\mathbb{E}_{s\\sim d^\\pi, a\\sim\\pi}[\\nabla_\\theta \\log \\pi_\\theta(a|s)\\, A^\\pi(s,a)]$ —
  and $A^\\pi = Q^\\pi - V^\\pi$ is from this lesson.</li>
  <li><strong>Lesson 10+ (Max-Ent RL):</strong> add an entropy bonus; the Bellman equation acquires a
  $\\log Z$ term and the optimal policy becomes a softmax over Q-values.</li>
  <li><strong>Lesson 16 (Diffusion in RL):</strong> even diffusion policies are training a sampler that
  maximizes $\\mathbb{E}_\\pi[Q^\\pi(s,a)]$ — the same value function we just defined.</li>
</ul>

<p>The MDP isn't a starting point you forget. It's the <strong>center of gravity</strong> of the
curriculum: every lesson going forward is either a way to solve the Bellman equation, a way
to <em>approximately</em> solve it, or a way to relax the assumptions and re-derive a generalized
form.</p>

<roadmap-mini active="mdps"></roadmap-mini>`,
    );
  },
};
