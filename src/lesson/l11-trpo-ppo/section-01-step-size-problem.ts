import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection01: Section = {
  id: "step-size-problem",
  title: "The Step-Size Problem",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>The Step-Size Problem</h2>
<p class="tagline"><em>The policy gradient is a direction, not a step size.</em></p>

<p>Lesson 10 gave us the policy gradient theorem and three increasingly capable gradient estimators:
REINFORCE, REINFORCE+baseline, and actor-critic with GAE. With a learning rate of around 0.5
on the 3×3 gridworld, actor-critic converges to $V(s_0) \\approx 0.725$ — close to the
theoretical maximum of $V^* = 0.7290$. We declared success and moved on.</p>

<p>But we never asked: <em>what is a learning rate, really?</em></p>

<p>A learning rate $\\alpha$ controls how large a step we take in <strong>parameter space</strong> —
in $\\theta$. The update rule $\\theta \\leftarrow \\theta + \\alpha \\hat{\\nabla}J(\\theta)$ moves
$\\theta$ by $\\alpha$ times the estimated gradient direction. If $\\alpha$ is small, the step is small.
If $\\alpha$ is large, the step is large.</p>

<p>What we care about is steps in <strong>policy space</strong> — in $\\pi_\\theta(a \\mid s)$.
The map from $\\theta$ to $\\pi$ is the softmax, and it is nonlinear: the same step in $\\theta$
can move $\\pi$ by a very small amount or by a very large amount, depending on where you started.</p>

<ul>
<li><strong>Near a uniform policy ($\\theta \\approx \\mathbf{0}$):</strong> a step of $\\Delta\\theta = 0.1$
barely moves the action probabilities.</li>
<li><strong>Near a sharp policy ($\\theta \\approx (10, -10, -10, -10)$):</strong> a step of $\\Delta\\theta = 0.1$
in any direction also barely moves anything — the policy is already locked.</li>
<li><strong>At a moderately confident policy ($\\theta \\approx (2, 0, 0, 0)$):</strong> a step of
$\\Delta\\theta = 1.0$ can change which action is most likely.</li>
</ul>

<p>This is the <strong>step-size problem</strong>: there is no fixed $\\alpha$ that controls the size of the
policy change uniformly across training. The same learning rate can be too small at the start and
too large near convergence, or the reverse — and there is no warning when you've chosen wrong.</p>

<h3>A second framing: stale advantage estimates</h3>

<p>Policy gradient updates are valid only for a small neighborhood of $\\theta_{\\mathrm{old}}$.
The advantage estimate $\\hat{A}(s, a)$ was collected under $\\pi_{\\theta_{\\mathrm{old}}}$.
If $\\theta_{\\mathrm{new}}$ moves the policy far enough that $\\pi_{\\theta_{\\mathrm{new}}}$
produces a very different state-action distribution, the advantage estimate becomes irrelevant:
you are taking a gradient step using stale data.
The standard term is <em>off-policyness</em> — a single big update makes your old batch
off-policy with respect to your new policy.</p>

<h3>Why the gridworld doesn't show the full problem</h3>

<p>On the 3×3 softmax gridworld, the step-size problem is real but mild.
The softmax keeps policy probabilities in the open simplex $(0, 1)^4$,
so even large $\\theta$ updates can't produce probability 0 or 1.
Recovery from a bad update is cheap because the action space is small and the MDP is deterministic.</p>

<p>The empirical evidence:</p>
<ul>
<li>At $\\alpha = 0.1$, vanilla PG converges to $V(s_0) \\approx -0.11$ — essentially stuck at the
noise floor, because each update is too small.</li>
<li>At $\\alpha = 5.0$, vanilla PG converges to $V(s_0) \\approx 0.728$ — fine, because
softmax is forgiving on this small problem.</li>
</ul>

<p>The lesson from the gridworld is that the "sweet spot" for learning rate is fairly wide on a tabular softmax.
The lesson from <em>practice</em> is that the sweet spot narrows dramatically as you move to deep
continuous-action policies: a Gaussian policy's variance can collapse in a single update;
an MLP policy's features can be scrambled by a batch with a reward outlier;
an LLM fine-tuned with PPO can drift off the SFT manifold and never recover.
The collapse story is real — it just isn't the gridworld's story.</p>

<ppo-param-vs-policy-space></ppo-param-vs-policy-space>
`);
  },
};
