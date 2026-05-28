import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection03: Section = {
  id: "policy-gradient-theorem",
  title: "The Policy Gradient Theorem",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>The Policy Gradient Theorem</h2>
<p class="tagline"><em>Apply the score function trick to trajectory likelihood. The transition dynamics cancel.</em></p>

<p>A trajectory $\\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \\ldots)$ has likelihood under $\\pi_\\theta$:</p>

$$p_\\theta(\\tau) = \\mu(s_0) \\prod_{t \\geq 0} \\pi_\\theta(a_t \\mid s_t)\\, P(s_{t+1} \\mid s_t, a_t).$$

<p>The performance objective is $J(\\theta) = \\mathbb{E}_{\\tau \\sim p_\\theta}[G(\\tau)]$
where $G(\\tau) = \\sum_t \\gamma^t r_{t+1}$. Taking the gradient:</p>

$$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[
  G(\\tau) \\, \\nabla_\\theta \\log p_\\theta(\\tau)
\\right].$$

<p>Now expand the log-likelihood. The initial state distribution $\\log \\mu(s_0)$
and the transition log-probabilities $\\log P(s_{t+1} \\mid s_t, a_t)$ do not depend on $\\theta$.
They vanish when differentiated:</p>

$$\\nabla_\\theta \\log p_\\theta(\\tau) = \\sum_{t \\geq 0} \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t).$$

<p>Substituting and distributing the return:</p>

$$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[
  \\sum_{t=0}^{T-1} G_t \\cdot \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)
\\right],$$

<p>where $G_t = \\sum_{k=t}^{T-1} \\gamma^{k-t} r_{k+1}$ is the reward-to-go from step $t$.
(Past rewards before step $t$ have zero expected correlation with $\\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)$
— they are independent given the trajectory up to $t$ — so only future rewards contribute.)</p>

<p>This is the <strong>policy gradient theorem</strong>. The key insight: the gradient of the
expected return is an expectation that we can estimate from sampled trajectories,
without knowing the environment's dynamics $P(s' \\mid s, a)$.
Policy gradient is model-free by construction.</p>

<h3>The score function for the softmax policy</h3>

<p>For the tabular softmax policy, $\\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)$ is the
score function vector derived in §2. Only the parameters $\\theta_{s_t, \\cdot}$ for the
current state are touched — the update is local to the visited state.</p>

<pg-policy-gradient-theorem></pg-policy-gradient-theorem>
`);
  },
};
