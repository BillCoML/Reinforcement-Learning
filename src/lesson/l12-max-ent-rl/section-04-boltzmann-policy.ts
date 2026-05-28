import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection04: Section = {
  id: "boltzmann-policy",
  title: "The Boltzmann Policy",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>The Boltzmann Policy</h2>
<p class="tagline"><em>Soft-optimal policies are Boltzmann distributions.</em></p>

<p>We pause to look closely at the soft-optimal policy derived in §3:</p>

$$\\pi_\\alpha^*(a|s) = \\frac{\\exp(Q_\\alpha^*(s,a) / \\alpha)}{Z_\\alpha(s)},
\\qquad
Z_\\alpha(s) = \\sum_{a'} \\exp(Q_\\alpha^*(s,a') / \\alpha).$$

<p>Three observations:</p>

<p><strong>The policy is determined by Q.</strong> Once you have $Q_\\alpha^*$, the policy
is a softmax. This is the cleanest possible connection between value-based and policy-based
methods: in max-ent RL, value learning and policy learning are <em>the same problem</em>.
SAC (Lesson 13) leverages this: it learns $Q_\\text{soft}$ and derives the policy via
softmax-projection. There is no separate "policy network" in the sense of Lesson 10 —
the policy is a function of Q.</p>

<p><strong>Temperature sets sharpness.</strong> Small $\\alpha$ concentrates on the argmax
(recovering greedy as $\\alpha \\to 0$); large $\\alpha$ flattens toward uniform. The
"right" $\\alpha$ is task-dependent and is itself an object of optimization in SAC v2
(Haarnoja et al. 2018b), which learns $\\alpha$ automatically by constraining the policy's
average entropy to a target.</p>

<p><strong>$Z_\\alpha(s)$ is a partition function.</strong> This is the same partition
function that appears in the RL-as-inference view (§7): the normalizer of the posterior
over actions given that the trajectory is "optimal." The names in this lesson — partition
function, Boltzmann distribution, temperature — come from statistical mechanics.
Many of the mathematical tools are shared.</p>

<h3>Policy entropy from Q</h3>

<p>The entropy of the Boltzmann policy has a useful closed form:</p>

$$\\mathcal{H}(\\pi_\\alpha^*(\\cdot|s)) = \\log Z_\\alpha(s) - \\frac{1}{\\alpha} \\mathbb{E}_{a \\sim \\pi_\\alpha^*}[Q_\\alpha^*(s,a)].$$

<p>This "log-Z minus expected energy" identity lets us compute entropy without summing
$-p \\log p$ directly, which is numerically tricky for sharp policies. It also shows
that entropy decreases as $\\alpha$ decreases (sharper policy → smaller log Z relative
to $\\mathbb{E}[Q]$).</p>

<h3>Numerical stability</h3>

<p>The partition function $Z_\\alpha(s)$ overflows when $Q/\\alpha$ is large.
The same max-shift trick used for the soft Bellman update applies here:
subtract $\\max_a Q(s,a)$ before exponentiating. This is the same numerical trick
used in neural network softmax over logits — the mathematical identity is:</p>

$$Z_\\alpha(s) = \\exp(\\max_a Q/\\alpha) \\cdot \\sum_a \\exp\\!\\left(\\frac{Q(s,a) - \\max_{a'} Q(s,a')}{\\alpha}\\right).$$

<p>The visualization shows the Boltzmann policy across all nine gridworld states at the
selected temperature. At $\\alpha \\to 0$, bars become delta spikes; at $\\alpha \\gtrsim 0.1$,
bars flatten toward uniform; at $\\alpha \\approx 0.05$, the interesting intermediate regime
is visible — states near the goal have sharp policies while far states are nearly uniform.</p>

<boltzmann-policy-grid></boltzmann-policy-grid>
`);
  },
};
