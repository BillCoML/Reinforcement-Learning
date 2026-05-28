import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection04: Section = {
  id: "reinforce-algorithm",
  title: "REINFORCE",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§4</span>REINFORCE</h2>
<p class="tagline"><em>Monte Carlo policy gradient (Williams, 1992): roll out, compute returns, update.</em></p>

<p>REINFORCE (Williams, 1992) is the direct implementation of the policy gradient theorem
using Monte Carlo estimates of the return. The algorithm is simple:</p>

<ol>
  <li>Roll out one episode: $(s_0, a_0, r_1, \\ldots, s_{T-1}, a_{T-1}, r_T)$.</li>
  <li>Compute reward-to-go $G_t = \\sum_{k=t}^{T-1} \\gamma^{k-t} r_{k+1}$ for each step $t$.</li>
  <li>Update parameters: $\\theta \\mathrel{+}= \\alpha \\sum_{t} \\gamma^t G_t \\, \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)$.</li>
</ol>

<p>For the tabular softmax policy on the $3 \\times 3$ gridworld, REINFORCE converges to
$V(0,0) \\approx 0.722$ — slightly below the tabular optimum $V^*(0,0) = 0.729$.
The gap is the <em>cost of representational stochasticity</em>: the softmax policy
can assign probability $\\epsilon > 0$ to non-optimal actions, and these residual
probabilities reduce the achievable return.</p>

<h3>Unbiasedness and high variance</h3>

<p>REINFORCE is an unbiased estimator of the policy gradient — the expected update
is exactly $\\nabla_\\theta J(\\theta)$. But the variance can be enormous.
The return $G_t$ fluctuates wildly across trajectories, especially early in training
when the policy is near-uniform. High-variance gradients mean many steps point in
wrong directions, and learning is slow and unstable.</p>

<p>The visualization below shows a training trace on the gridworld: V(s₀=0) estimated
after each episode, the per-episode gradient norm, and the policy's action probabilities
at the start state. Watch how the gradient norm decays as the policy sharpens.</p>

<pg-reinforce-training-trace></pg-reinforce-training-trace>
`);
  },
};
