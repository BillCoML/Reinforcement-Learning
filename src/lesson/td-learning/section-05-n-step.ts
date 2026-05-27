import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection05: Section = {
  id: "n-step-td",
  title: "n-step TD: Interpolating Between TD(0) and MC",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>n-step TD: Interpolating Between TD(0) and MC</h2>
<p class="tagline"><em>Look ahead n steps before bootstrapping. A continuous family between TD(0) and MC.</em></p>

<p>TD(0) bootstraps after one step. MC waits for the entire episode. Between these
extremes is a continuous family of algorithms that look ahead $n$ steps before
bootstrapping. The <strong>$n$-step return</strong> is</p>

$$G_t^{(n)} \\;:=\\; r_{t+1} + \\gamma r_{t+2} + \\cdots + \\gamma^{n-1} r_{t+n} + \\gamma^n V(s_{t+n}).$$

<p>For $n = 1$ this is the TD(0) target $r_{t+1} + \\gamma V(s_{t+1})$. As $n \\to T - t$
(length of remaining episode), the bootstrap term vanishes and $G_t^{(n)}$ becomes the
MC return $G_t$. The $n$-step TD update:</p>

$$V(s_t) \\;\\leftarrow\\; V(s_t) + \\alpha [G_t^{(n)} - V(s_t)].$$

<p>Implementation requires a buffer: the algorithm must wait $n$ steps after a state is
visited before computing the update for that state. Once the episode ends, all remaining
updates use MC returns (no bootstrap, since the future is fully observed).</p>

<h3>The bias-variance trade-off across $n$</h3>

<p>Small $n$ has high bias (strongly relies on the bootstrap, which is wrong) and low
variance (few rewards in the sum). Large $n$ has low bias (uses many observed rewards)
and high variance. The <strong>U-shape intuition</strong> — there exists an interior optimum where
bias and variance balance — is true on long-horizon, noisy-reward problems. It is
<em>not</em> universally true.</p>

<p>Empirical on the running gridworld ($N = 2000$, $\\alpha = 0.1$, 20 trials):</p>

<table class="rl-table">
<thead><tr><th>$n$</th><th>mean</th><th>std</th><th>RMSE</th></tr></thead>
<tbody>
<tr><td>1</td><td>−0.4450</td><td>0.0296</td><td>0.0385</td></tr>
<tr><td>2</td><td>−0.4634</td><td>0.0715</td><td>0.0833</td></tr>
<tr><td>4</td><td>−0.4664</td><td>0.1235</td><td>0.1317</td></tr>
<tr><td>8</td><td>−0.5003</td><td>0.1561</td><td>0.1753</td></tr>
<tr><td>16</td><td>−0.4933</td><td>0.2145</td><td>0.2265</td></tr>
<tr><td>100</td><td>−0.4918</td><td>0.2232</td><td>0.2343</td></tr>
<tr><td>(MC)</td><td>−0.4216</td><td>0.0107</td><td>0.0107</td></tr>
</tbody>
</table>

<p>RMSE grows monotonically with $n$ on this gridworld. The horizon is too short for the
textbook U-shape to emerge. MC has the lowest RMSE of all — for short episodes, MC's
per-trajectory variance is small enough that constant-$\\alpha$ TD methods cannot match it.</p>

<p>On long-horizon problems (typically $T > 50$ with noisy rewards), n-step TD's U-shape
becomes visible, with optimal $n$ typically between 4 and 16. This is the regime modern
deep RL operates in, which is why n-step variants are widely deployed.</p>

<div class="crosslink-callout">
  <strong>Forward link · GAE (Lesson 10).</strong>
  The Generalized Advantage Estimator (GAE) is a weighted exponential average of
  $n$-step advantages, controlled by a parameter $\\lambda \\in [0,1]$. With $\\lambda = 0$
  it reduces to a one-step advantage (TD(0)-style); with $\\lambda = 1$ it reduces to the
  MC advantage. The exponential weighting is the $n$-step generalization that policy
  gradient methods actually use, descending directly from the TD($\\lambda$) apparatus
  we develop in Section 6.
</div>

<div class="component-host">
  <n-step-backup-diagrams></n-step-backup-diagrams>
</div>
`);
  },
};
