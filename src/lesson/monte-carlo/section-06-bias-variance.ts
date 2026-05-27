import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection06: Section = {
  id: "mc-vs-td",
  title: "Variance, Bias, and the TD Comparison",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§6</span>Variance, Bias, and the TD Comparison</h2>
<p class="tagline"><em>MC has zero bias and huge variance. TD trades some bias for less variance. Lesson 8 picks up here.</em></p>

<p>MC methods have a clean theoretical profile. The MC estimator is <strong>unbiased</strong>:
each return $G_t$ is exactly a sample of $V^\\pi(s_t)$, and averaging unbiased
samples gives an unbiased estimator. The MC estimator's <strong>variance</strong>, however,
is the entire return's variance. For a horizon-$T$ episode, this is the variance of a
sum of $T$ discounted rewards, which generally grows with $T$ (more rewards
contribute, each with its own randomness). For long-horizon problems, MC variance is
substantial and growing.</p>

<p>There is a structurally different approach that trades some of MC's zero bias for
substantially lower variance. It is <strong>bootstrapping</strong>: estimate $V^\\pi(s)$ not
by waiting for the full return, but by plugging in your current estimate
$\\hat V^\\pi(s')$ for the next state and using $r + \\gamma \\hat V^\\pi(s')$ as the
target. The result is a biased estimator — the current $\\hat V$ is not yet
correct — but the variance is much lower because only one reward and one bootstrap
contribute to the target rather than a full return.</p>

<p>This is the <strong>TD(0)</strong> update, and it is the central content of Lesson 8. The
bias from bootstrapping vanishes as $\\hat V \\to V^\\pi$, and under suitable step-size
schedules (the Robbins-Monro conditions) TD(0) converges. The variance reduction is
often dramatic: ten to a hundred times smaller than MC on long-horizon problems.</p>

<h3>A picture of the trade-off</h3>

<p>A common visualization is the <strong>bias-variance plane</strong>, with bias on one axis
and variance on the other:</p>

<ul>
  <li>MC sits at zero bias, high variance.</li>
  <li>TD(0) sits at moderate bias, low variance.</li>
  <li>$n$-step TD interpolates between them; as $n \\to T$, $n$-step TD becomes MC;
      as $n \\to 1$, it becomes TD(0).</li>
</ul>

<p>The $n$-step TD method (Lesson 8) is one of the most elegant parameterized families
in classical RL: a single dial that smoothly moves between the two extremes, and the
optimal setting depends on problem-specific properties.</p>

<h3>What MC keeps doing well</h3>

<p>Despite TD's variance advantage, MC remains useful for several reasons. MC is
unbiased, which matters when we need theoretical guarantees. MC sidesteps
bootstrapping, which means it avoids the deadly triad (Lesson 9) when combined with
function approximation. MC is simple to implement: no learning rates, no target
networks, no TD error backups, just trajectory averages. And MC is the natural fit
for the offline-RL setting (Lesson 15) where we have a fixed dataset and want to
estimate the value of a policy without the distributional issues that bootstrapping
introduces.</p>

<p>In modern deep RL, pure MC is rare but the MC return is often used as a high-$n$
end of $n$-step TD, and as a target for the value function in policy-gradient methods
(Lesson 10 uses it directly as the unbiased gradient target).</p>

<div class="component-host">
  <bias-variance-plane-preview></bias-variance-plane-preview>
</div>
`);
  },
};
