import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection07: Section = {
  id: "ppo-empirics",
  title: "Empirical Comparisons and Hyperparameter Sensitivity",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§7</span>Empirical Comparisons and Hyperparameter Sensitivity</h2>
<p class="tagline"><em>Where the win is real, and where it isn't.</em></p>

<p>A direct empirical comparison of vanilla PG and PPO across learning rates (20 seeds, batch=10,
200 iterations, $\\lambda = 0.95$, $\\varepsilon = 0.2$):</p>

<ppo-lr-sensitivity-table></ppo-lr-sensitivity-table>

<h3>Two observations</h3>

<p><strong>First: PPO works across a wide range of learning rates.</strong>
Its final $V(s_0)$ stays above 0.72 from lr=0.3 through lr=5.0.
Vanilla PG's sweet spot is much narrower: at lr=0.1, it is stuck near $-0.11$;
at lr=0.5, it is still about 0.05 below PPO; only above lr=1.0 does it catch up.
PPO's advantage is <em>robustness</em>: less sensitive to the choice of learning rate,
which translates directly into "less hyperparameter tuning required."</p>

<p><strong>Second: at the high end of the sweep (lr=2.0, lr=5.0), vanilla PG and PPO
are statistically indistinguishable.</strong>
On this MDP, with this policy class, with enough learning rate, vanilla PG works.
The "vanilla PG catastrophically fails" narrative does not hold here.
We report this directly.</p>

<p>The headline run for the lesson — PPO at lr=0.5, batch=20, 200 iters, 10 seeds —
gives a final $V(s_0) = 0.7270 \\pm 0.0003$. Compared to Lesson 10's REINFORCE+baseline
at the same scale ($0.7250 \\pm 0.0011$), PPO is slightly higher mean and about
$4\\times$ lower standard deviation. The improvement is incremental at this scale, not revolutionary.
PPO's honest advantages on the gridworld are: (a) sample efficiency
(multiple epochs per batch), and (b) hyperparameter robustness. The dramatic collapse
story that motivates trust regions lives in continuous-action deep RL, not here.</p>

<h3>The clip ε sweep</h3>

<p>A second experiment at lr=1.5, batch=10, 5 seeds, varying $\\varepsilon$:</p>

<table class="algo-table">
<thead><tr><th>$\\varepsilon$</th><th>Final $V(s_0)$</th><th>Std</th><th>Late clip frac.</th></tr></thead>
<tbody>
<tr><td>0.05</td><td>0.7254</td><td>0.0008</td><td>0.003</td></tr>
<tr><td>0.10</td><td>0.7260</td><td>0.0003</td><td>0.002</td></tr>
<tr><td>0.20</td><td>0.7264</td><td>0.0002</td><td>0.000</td></tr>
<tr><td>0.40</td><td>0.7278</td><td>0.0002</td><td>0.005</td></tr>
</tbody>
</table>

<p>At this learning rate, $\\varepsilon$ barely matters above 0.1 — the ratios stay close to 1 anyway.
The standard $\\varepsilon = 0.2$ is a reasonable default. On larger problems (continuous control,
deep critics), $\\varepsilon$ matters more because the ratios can drift further per batch.</p>
`);
  },
};
