import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection04: Section = {
  id: "ppo-clipped",
  title: "PPO and the Clipped Ratio",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>PPO and the Clipped Ratio</h2>
<p class="tagline"><em>A surrogate with a built-in trust region.</em></p>

<p>PPO (Proximal Policy Optimization, Schulman et al. 2017) drops TRPO's second-order machinery
and replaces it with a first-order trick: modify the surrogate objective so that the gradient
<em>naturally</em> vanishes when the policy has moved too far.</p>

<h3>From IS surrogate to clipped surrogate</h3>

<p>Recall the IS surrogate from §2:</p>

$$\\hat{\\mathcal{L}}^{\\mathrm{IS}}(\\theta) \\;=\\;
\\hat{\\mathbb{E}}\\!\\left[ r_t(\\theta)\\, \\hat{A}_t \\right],
\\qquad
r_t(\\theta) \\;=\\;
\\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\mathrm{old}}}(a_t|s_t)}.$$

<p>PPO replaces it with the <strong>clipped surrogate</strong>:</p>

$$\\hat{\\mathcal{L}}^{\\mathrm{CLIP}}(\\theta) \\;=\\;
\\hat{\\mathbb{E}}\\!\\left[
  \\min\\!\\bigl(
    r_t(\\theta)\\, \\hat{A}_t,\\;
    \\mathrm{clip}\\!\\bigl(r_t(\\theta),\\, 1-\\varepsilon,\\, 1+\\varepsilon\\bigr)\\,
    \\hat{A}_t
  \\bigr)
\\right].$$

<h3>Shape of the objective</h3>

<p>The $\\min$ creates four cases depending on the sign of $\\hat{A}_t$:</p>

<table class="algo-table">
<thead><tr><th>$\\hat{A}$ sign</th><th>$r$ vs. $[1-\\varepsilon, 1+\\varepsilon]$</th><th>$\\min$ picks</th><th>Gradient</th></tr></thead>
<tbody>
<tr><td>$\\hat{A} > 0$</td><td>$r < 1-\\varepsilon$</td><td>$r \\cdot \\hat{A}$ (smaller)</td><td>Alive — pushes $r$ up toward 1</td></tr>
<tr><td>$\\hat{A} > 0$</td><td>$r > 1+\\varepsilon$</td><td>$(1+\\varepsilon)\\hat{A}$ (smaller)</td><td><strong>Zero</strong> — already pushed too far</td></tr>
<tr><td>$\\hat{A} < 0$</td><td>$r < 1-\\varepsilon$</td><td>$(1-\\varepsilon)\\hat{A}$ (smaller)</td><td><strong>Zero</strong> — already decreased too far</td></tr>
<tr><td>$\\hat{A} < 0$</td><td>$r > 1+\\varepsilon$</td><td>$r \\cdot \\hat{A}$ (smaller)</td><td>Alive — pushes $r$ down toward 1</td></tr>
</tbody>
</table>

<h3>Why the min, not just the clip?</h3>

<p>Without the $\\min$, the clipped objective $\\mathrm{clip}(r) \\cdot \\hat{A}$ would have
zero gradient <em>everywhere outside</em> $[1-\\varepsilon, 1+\\varepsilon]$ — even when the policy
is moving in the wrong direction. The $\\min$ ensures the gradient is alive when the policy
is moving <em>back into</em> the trust region.</p>

<p>Specifically: if $\\hat{A} > 0$ and $r < 1-\\varepsilon$ (we've mistakenly decreased the
probability of a good action), the unclipped $r \\cdot \\hat{A}$ is the smaller term,
and the $\\min$ picks it — so the gradient pushes $r$ back up. The clip only "bites" when
the policy is moving <em>away</em> from the old policy in a way the objective wants to keep going.</p>

<h3>Multi-epoch updates and the ratio's drift</h3>

<p>PPO does <em>multiple</em> gradient steps on the same batch. At the first step, $\\theta = \\theta_{\\mathrm{old}}$,
so every ratio $r_t = 1$ exactly. As $\\theta$ moves, the ratios drift away from 1.
The clipping only starts to bind after the policy has moved sufficiently. In practice with
$\\varepsilon = 0.2$ and moderate settings, the clip fraction is under 5% throughout training.
At aggressive settings (lr=2.0, batch=5, epochs=10), the clip fraction climbs to 30–45% early
and then falls as the policy converges — the centerpiece (§6) shows this directly.</p>

<h3>The full PPO objective</h3>

$$\\hat{\\mathcal{L}}^{\\mathrm{PPO}} \\;=\\;
\\hat{\\mathbb{E}}\\!\\left[
  \\hat{\\mathcal{L}}^{\\mathrm{CLIP}} \\;-\\;
  c_1\\,(V_\\phi(s_t) - \\hat{R}_t)^2 \\;+\\;
  c_2\\,\\mathcal{H}\\!\\left[\\pi_\\theta(\\cdot | s_t)\\right]
\\right].$$

<p>The entropy bonus $c_2 \\mathcal{H}[\\pi_\\theta]$ is a preview of Lesson 12 —
it is exactly the maximum-entropy objective, weighted by $c_2$,
and it prevents the policy from collapsing onto a single action prematurely.
Lesson 12 will make this term the <em>primary</em> objective of an entire algorithm family.</p>

<ppo-clipped-surrogate-curve></ppo-clipped-surrogate-curve>
`);
  },
};
