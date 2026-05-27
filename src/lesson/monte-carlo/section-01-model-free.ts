import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection01: Section = {
  id: "model-free-setting",
  title: "The Model-Free Setting",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>The Model-Free Setting</h2>
<p class="tagline"><em>No transition probabilities. Just trajectories and returns.</em></p>

<p>The dynamic programming methods of Lesson 5 all share an assumption: the agent has
full access to the MDP's transition kernel $P(s' \\mid s, a)$ and reward function
$R(s, a)$. Policy evaluation computes $V^\\pi$ by solving a linear system involving
$P_\\pi$ and $R_\\pi$. Value iteration backs up
$\\max_a \\sum_{s'} P(s' \\mid s, a) [R(s, a, s') + \\gamma V(s')]$ at every step.
Even the asynchronous and Gauss-Seidel variants assume that for any $(s, a)$
we can enumerate the next-state distribution.</p>

<p>In most real settings this assumption is not satisfied. We do not have a closed-form
transition kernel for a robot, for a video game, for a language model. What we have is
the ability to <strong>interact</strong> with the environment: pick an action, observe a reward,
observe a next state. The agent collects sequences of the form</p>

$$\\tau \\;=\\; (s_0, a_0, r_1, s_1, a_1, r_2, s_2, \\ldots, s_T)$$

<p>where the rewards and next states are drawn from the environment's underlying (but
inaccessible) dynamics. The challenge is to extract from these sequences enough
information to do the work that DP previously did with $P$ and $R$ directly.</p>

<p>This is the <strong>model-free</strong> setting. The agent does not know the model, does not
learn the model, and uses only sampled experience. The flavor of methods that work
here is fundamentally statistical: estimate expectations by averages, accept the
variance that comes with finite samples, and design algorithms whose convergence
relies on the law of large numbers rather than on a contraction in a known operator.</p>

<p>Two large families of model-free methods will occupy the curriculum from this lesson
forward. <strong>Monte Carlo methods</strong> wait until the end of an episode and use the
empirical return as a sample of $V^\\pi(s)$. <strong>Temporal-Difference methods</strong>
(Lesson 8) use a bootstrap: estimate $V^\\pi(s)$ from $r + \\gamma \\hat V^\\pi(s')$,
plugging the agent's current estimate into the right-hand side. MC has zero bias and
high variance. TD has bias from bootstrapping and substantially lower variance.
This lesson is about the MC family. Lesson 8 is about TD.</p>

<h3>The return as a noisy observation of the value</h3>

<p>The state value function $V^\\pi(s)$ is defined as the expected discounted return
when starting in state $s$ and following policy $\\pi$:</p>

$$V^\\pi(s) \\;:=\\; \\mathbb{E}_\\pi\\!\\left[ \\sum_{t=0}^{\\infty} \\gamma^t R_{t+1} \\,\\bigg|\\, S_0 = s \\right].$$

<p>For an episodic task, the sum terminates at the end of the episode. Crucially, if we
<em>sample</em> a trajectory $\\tau$ starting from $s$ under policy $\\pi$, the random variable</p>

$$G_0(\\tau) \\;:=\\; \\sum_{t=0}^{T-1} \\gamma^t R_{t+1}$$

<p>is an <strong>unbiased estimate of $V^\\pi(s)$</strong>. Take many independent trajectories,
average their returns, and by the law of large numbers the average converges to
$V^\\pi(s)$. That single observation is the entire idea of Monte Carlo policy evaluation.</p>

<p>The complication is that we usually want $V^\\pi(s)$ for <em>every</em> state $s$, not
just one. Section 2 takes up the question of how to use a single trajectory to update
value estimates for all states it visits.</p>

<div class="crosslink-callout">
  <strong>Forward link · Function approximation and the deadly triad.</strong>
  In Lesson 9 we will face a setting where we cannot maintain a separate $V(s)$ for
  every state — there are too many states — and must instead parameterize $V$ with a
  neural network. The Monte Carlo estimator generalizes cleanly to that setting: the
  return is still a noisy observation of $V^\\pi(s)$ and can be used as a regression
  target. The TD estimator, in contrast, runs into the deadly triad when combined with
  bootstrapping and off-policy learning. One reason MC remains useful despite its
  variance is that it sidesteps the deadly triad entirely.
</div>

<div class="component-host">
  <model-free-vs-model-based></model-free-vs-model-based>
</div>
`);
  },
};
