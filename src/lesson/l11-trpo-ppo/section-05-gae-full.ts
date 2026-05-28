import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection05: Section = {
  id: "gae-full",
  title: "GAE: Generalized Advantage Estimation",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§5</span>GAE: Generalized Advantage Estimation</h2>
<p class="tagline"><em>The λ-interpolator, in full.</em></p>

<p>Lesson 10 §7 previewed GAE as a λ-interpolation between the TD(0) advantage
and the Monte Carlo advantage. Here we derive it properly and study its
bias-variance trade on the gridworld.</p>

<h3>Derivation</h3>

<p>The starting point is the TD residual:</p>

$$\\delta_t \\;=\\; r_t + \\gamma V(s_{t+1}) - V(s_t).$$

<p>The GAE advantage with parameter $\\lambda \\in [0, 1]$ is:</p>

$$\\hat{A}_t^{\\mathrm{GAE}(\\lambda)} \\;=\\;
\\sum_{k=0}^{\\infty} (\\gamma \\lambda)^k\\, \\delta_{t+k}.$$

<p>At $\\lambda = 0$, $\\hat{A}_t^{\\mathrm{GAE}(0)} = \\delta_t$ — the TD residual, the
one-step bootstrap advantage from Lesson 8.</p>

<p>At $\\lambda = 1$, GAE collapses to the MC advantage (after telescoping):</p>

$$\\hat{A}_t^{\\mathrm{GAE}(1)} \\;=\\;
\\sum_{k=0}^{\\infty} \\gamma^k\\, \\delta_{t+k}
\\;=\\; G_t - V(s_t).$$

<p>GAE interpolates between these two extremes the same way $n$-step TD (Lesson 8 §5)
interpolated between TD(0) and MC.
The connection is direct: GAE($\\lambda$) is to advantage estimation what TD($\\lambda$)
is to value estimation. The pink color reserved in §2 echoes the pink used for
$n$-step TD in Lesson 8.</p>

<h3>The bias-variance trade</h3>

<p>At low $\\lambda$, the advantage estimate has low variance (depends on few rewards)
but high bias (depends on the critic's accuracy at $V(s_{t+1})$).
At high $\\lambda$, the estimate has high variance but low bias.</p>

<p>Empirical results on the 3×3 gridworld with batch=5 (a noisy critic), 20 seeds:</p>

<table class="algo-table">
<thead><tr><th>$\\lambda$</th><th>Final $V(s_0)$</th><th>Std (20 seeds)</th></tr></thead>
<tbody>
<tr><td>0.00</td><td>0.724</td><td>0.001</td></tr>
<tr><td>0.50</td><td>0.724</td><td>0.001</td></tr>
<tr><td>0.90</td><td>0.722</td><td>0.002</td></tr>
<tr><td>0.95</td><td>0.722</td><td>0.002</td></tr>
<tr><td>1.00</td><td>0.721</td><td>0.002</td></tr>
</tbody>
</table>

<p>Both the mean and the std worsen monotonically as $\\lambda \\to 1$.
On this small problem with deterministic dynamics, the critic is easy to learn
and the bootstrap bias is negligible — so the variance reduction from $\\lambda = 0$ wins.</p>

<p>The canonical $\\lambda = 0.95$ default is a deep-RL convention that originates from much
larger problems (continuous control, neural critics) where the critic is inaccurate
and the bootstrap bias is substantial. The lesson reports both numbers and explains the gap,
rather than pretending the heuristic holds at every scale.</p>

<ppo-gae-lambda-sweep></ppo-gae-lambda-sweep>
`);
  },
};
