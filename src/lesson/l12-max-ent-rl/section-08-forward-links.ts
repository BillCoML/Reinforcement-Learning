import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection08: Section = {
  id: "l12-forward-links",
  title: "Forward Links",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§8</span>Forward Links</h2>
<p class="tagline"><em>Toward SAC, toward RLHF.</em></p>

<div class="crosslink-callout">
<strong>→ Lesson 13 — Soft Actor-Critic</strong><br>
SAC is the continuous-action realization of soft policy iteration. It parameterizes
$Q_\\text{soft}$ with a neural network, derives the policy via Boltzmann projection
(using a reparameterized Gaussian to make the softmax differentiable over continuous
actions), and uses the auto-tuned $\\alpha$ trick to learn the temperature from a
target-entropy constraint. Three pieces of this lesson reappear in L13: the soft Bellman
operator from §3 (now applied to a neural Q-function), the Boltzmann policy from §4
(now a reparameterized Gaussian), and the failure mode from §5 (avoided by operating
in infinite-horizon continuous control). The auto-tuning of $\\alpha$ is SAC's most
pragmatic answer to the temperature-selection problem raised in §6.
</div>

<div class="crosslink-callout">
<strong>→ Lesson 17 — RLHF and DPO (the capstone)</strong><br>
In RLHF, the entropy term is replaced by a KL penalty to a reference policy:
$$J_\\text{RLHF}(\\pi) = \\mathbb{E}^\\pi\\!\\left[\\sum_t r(s_t, a_t)\\right]
- \\beta\\, D_\\mathrm{KL}\\!\\left(\\pi(\\cdot|s_t) \\| \\pi_\\mathrm{ref}(\\cdot|s_t)\\right).$$
The structure is exactly parallel to §2 with $\\alpha \\mathcal{H}(\\pi) \\to
-\\beta D_\\mathrm{KL}(\\pi \\| \\pi_\\mathrm{ref})$.
The two are related: KL to uniform is entropy plus a constant; KL to a non-uniform
reference is what distinguishes RLHF.
The failure mode of §5 disappears: the reference $\\pi_\\mathrm{ref}$ is goal-directed
(a supervised fine-tuned LLM), so deviating to dawdle costs KL, not gains entropy.
The optimal policy under the RLHF objective has the same Boltzmann form, but with
$\\pi_\\mathrm{ref}$ multiplying the exponential:
$$\\pi^*(a|s) \\propto \\pi_\\mathrm{ref}(a|s)\\, \\exp(Q(s,a)/\\beta).$$
This is the closed-form DPO update target. Lesson 17 will derive it.
</div>

<h3>On the trimmed branches</h3>

<p>Model-based RL (deferred L14) has a clean max-ent variant: planning under an
entropy-regularized objective is itself an inference problem — a corollary of §7.
Offline RL (deferred L15) uses max-ent regularization heavily; CQL's conservative
Q-learning can be viewed as a KL-constrained Boltzmann policy, exactly the form above.
Diffusion in RL (deferred L16) takes the RL-as-inference view to its limit, treating
action generation as sampling from a learned posterior. The pointers above give
the conceptual bridges.</p>

<maxent-roadmap-mini></maxent-roadmap-mini>
`);
  },
};
