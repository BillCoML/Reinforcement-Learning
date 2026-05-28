import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection08: Section = {
  id: "l11-forward-links",
  title: "Forward Links",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§8</span>Forward Links</h2>
<p class="tagline"><em>Toward maximum entropy, continuous actions, and LLMs.</em></p>

<div class="crosslink-callout">
<strong>→ Lesson 12 — Maximum-Entropy RL</strong><br>
The entropy bonus $c_2 \\cdot \\mathcal{H}[\\pi_\\theta(\\cdot|s)]$ in the PPO objective was treated
here as a regularizer that prevents premature policy collapse.
In Lesson 12 it becomes the <em>primary</em> objective: $J(\\theta) = \\mathbb{E}[\\sum_t (r_t + \\alpha\\, \\mathcal{H}(\\pi(\\cdot|s_t)))]$.
The "softmax cap" from Lesson 10 — the gap between a converged softmax policy's $V(s_0) \\approx 0.722$
and $V^* = 0.729$ — gets reframed as a <em>feature</em>: a max-ent policy <em>should</em> be
stochastic at convergence; that stochasticity is the point.
The bridge from L11 to L12 is the entropy bonus coefficient $c_2$, which becomes
the temperature $\\alpha$ in the max-ent objective.
</div>

<div class="crosslink-callout">
<strong>→ Lesson 13 — Soft Actor-Critic</strong><br>
SAC is the continuous-action realization of max-ent actor-critic.
It inherits two pieces directly from this lesson: the clipped ratio's spirit
(SAC uses a similar bound on policy change, via the reparameterization trick rather than
an explicit clip) and the twin-critic architecture (a direct descendant of Lesson 9's Double DQN).
Lesson 13 is also where continuous actions become serious — the gridworld with discrete actions
is left behind, and a continuous-action environment takes over.
The 1D Gaussian sidebar in the centerpiece (V4) is the visual bridge.
</div>

<div class="crosslink-callout">
<strong>→ Lesson 17 — RLHF and DPO (the capstone)</strong><br>
PPO is the workhorse algorithm for LLM alignment via RLHF.
The pipeline: a preference dataset trains a reward model; the reward model produces scalar rewards
on language-model rollouts; PPO uses those rewards to fine-tune the language model under a KL
constraint to a reference (typically the SFT model).
Three pieces of Lesson 11 come together there: the IS ratio from §2 is the probability ratio
between the fine-tuned model and the SFT reference; the clipped surrogate from §4 is the per-token
PPO objective; the KL constraint from §2 (now an explicit penalty rather than a clipping bound)
keeps the policy near the SFT manifold.
DPO (Rafailov et al. 2023) is a closed-form alternative that bypasses the reward model entirely;
we treat it as PPO's sibling in Lesson 17.
</div>

<h3>On the trimmed branches</h3>

<p>The original curriculum plan included Lesson 14 (Model-Based RL), Lesson 15 (Offline RL),
and Lesson 16 (Diffusion in RL) before the capstone.
Those lessons are deferred from the production schedule — not out of intellectual importance,
but out of scope constraints.
Model-based RL with PPO as the planner shows up in MuZero;
offline RL with PPO's conservatism is the foundation of CQL and IQL;
diffusion policies trained with PPO are an active research area.
Canonical references appear in the out-of-scope appendix.</p>

<p>The spine is now Lesson 10 → <strong>Lesson 11</strong> → Lesson 12 → Lesson 13 → Lesson 17.
You are at the hinge between foundations and the capstone.</p>

<ppo-roadmap-mini></ppo-roadmap-mini>
`);
  },
};
