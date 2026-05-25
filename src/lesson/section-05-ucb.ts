import type { Section } from "./section";
import { sectionFromHTML } from "./section";
import { forwardLink } from "../components/CrosslinkCallout";

export const section05: Section = {
  id: "ucb1",
  title: "UCB: Optimism Under Uncertainty",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§5</span>UCB: Optimism Under Uncertainty</h2>
<p class="tagline">When in doubt, act as if your best guess is the upper bound.</p>

<p>ε-greedy's failure is structural: it explores <em>uniformly across arms</em> even when one
arm is obviously fine and the others are obviously bad. A better algorithm would
target its exploration to arms where the agent is <em>uncertain</em>. UCB does exactly that.</p>

<p>The recipe is "<strong>optimism in the face of uncertainty</strong>." For each arm, maintain an
<em>upper confidence bound</em> $\\mathrm{UCB}_i(t)$ such that, with high probability, $\\mu_i \\leq \\mathrm{UCB}_i(t)$.
Then pull the arm with the highest UCB. If an arm's UCB is high, either (a) its
mean really is high — great, pull it — or (b) it's high because we're uncertain about
it — also fine, pulling it gives us information.</p>

<p>To make this concrete we need a confidence interval for the empirical mean. The
right tool is <strong>Hoeffding's inequality</strong> (Hoeffding, 1963): for $R_1, \\dots, R_n$ i.i.d.
in $[0, 1]$ with mean $\\mu$, and $\\hat{\\mu}_n = (1/n) \\sum R_s$,</p>

$$\\boxed{\\Pr\\!\\left(|\\hat{\\mu}_n - \\mu| > \\varepsilon\\right) \\;\\leq\\; 2 \\exp(-2 n \\varepsilon^2)}$$

<p>Set the right-hand side to $2 t^{-\\alpha}$ for some $\\alpha > 0$. Solving,
$\\varepsilon = \\sqrt{\\alpha \\log t / (2n)}$. Substitute $\\alpha = 4$ and you get the canonical UCB1 bonus
$\\sqrt{2 \\log t / N_i}$. With $\\alpha = 4$, the failure probability at time $t$ for a single
arm is $2 t^{-4}$, and union-bounding over all $t \\cdot K$ (arm, time) pairs still
gives a vanishing failure probability.</p>

<h3>Algorithm (UCB1)</h3>
<pre><code>for t = 1, 2, …, T:
    if t ≤ K:                                    # initialize
        a_t = t
    else:
        for each arm i:
            UCB_i = μ̂_i + sqrt(2 * log(t) / N_i)
        a_t = argmax_i UCB_i
    pull a_t, observe R_t, update N_{a_t} and μ̂_{a_t}</code></pre>

<p>That's it. One line of bonus, one argmax.</p>

<figure class="numeric">
  <figcaption>Numerical trace (pre-verified)</figcaption>
  <p>Initialize by pulling arms 1, 2, 3 once with rewards $(0, 1, 0)$. At $t = 4$
  (about to make the 4th pull):</p>
  <table>
    <thead><tr><th>Arm</th><th>$\\hat{\\mu}_i$</th><th>$N_i$</th><th>bonus $\\sqrt{2\\ln 4 / N_i}$</th><th>UCB</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>0.000</td><td>1</td><td>1.6651</td><td>1.6651</td></tr>
      <tr><td>2</td><td>1.000</td><td>1</td><td>1.6651</td><td>2.6651</td></tr>
      <tr><td>3</td><td>0.000</td><td>1</td><td>1.6651</td><td>1.6651</td></tr>
    </tbody>
  </table>
  <p>→ pull arm 2. Observe reward $R = 1$. Now $N_2 = 2,\\ \\hat{\\mu}_2 = 1.0$. At $t = 5$:</p>
  <table>
    <thead><tr><th>Arm</th><th>$\\hat{\\mu}_i$</th><th>$N_i$</th><th>bonus $\\sqrt{2\\ln 5 / N_i}$</th><th>UCB</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>0.000</td><td>1</td><td>1.7941</td><td>1.7941</td></tr>
      <tr><td>2</td><td>1.000</td><td>2</td><td>1.2686</td><td>2.2686</td></tr>
      <tr><td>3</td><td>0.000</td><td>1</td><td>1.7941</td><td>1.7941</td></tr>
    </tbody>
  </table>
  <p style="margin-bottom:0">→ pull arm 2 again. Notice that the bonus on arm 1 and arm 3 <em>grew</em> slightly even
  though they weren't pulled, because $\\log t$ increased while their $N_i$ didn't.
  That's the engine: under-pulled arms slowly become more attractive.</p>
</figure>

<h3>Regret bound (Auer et al. 2002, Theorem 1)</h3>
<p>UCB1 satisfies</p>

$$\\boxed{R_T \\;\\leq\\; \\sum_{i: \\Delta_i > 0} \\frac{8 \\log T}{\\Delta_i} \\;+\\; \\left(1 + \\frac{\\pi^2}{3}\\right) \\sum_{i=1}^K \\Delta_i}$$

<p>The first term is the <strong>instance-dependent regret</strong>, the second is a finite
"settling" cost. For our running example:</p>

<table>
  <thead><tr><th>$T$</th><th>UCB1 upper bound $\\leq$</th></tr></thead>
  <tbody>
    <tr><td>100</td><td>278.88</td></tr>
    <tr><td>1000</td><td>417.04</td></tr>
    <tr><td>10000</td><td>555.19</td></tr>
    <tr><td>100000</td><td>693.35</td></tr>
  </tbody>
</table>

<p>These are <em>upper</em> bounds — actual UCB1 regret is much smaller; the bound is loose
by typical factors of 3–8×. From our simulation, UCB1's empirical regret
at $T = 5000$ is $\\approx 82$. The upper bound at $T = 5000$ is $\\approx 486$. The bound's
<em>shape</em> (log-$T$) is correct, even though its constant is conservative.</p>

<h3>The slick proof idea</h3>
<p>Suppose UCB1 pulls a suboptimal arm $i$ at time $t$. That means $\\mathrm{UCB}_i(t) \\geq \\mathrm{UCB}_*(t)$.
One of three things must be true:</p>
<ol>
  <li>$\\hat{\\mu}_*$ underestimates $\\mu^*$ by more than the bonus — this is rare (Hoeffding).</li>
  <li>$\\hat{\\mu}_i$ overestimates $\\mu_i$ by more than the bonus — also rare.</li>
  <li>The bonus $\\sqrt{2 \\log t / N_i}$ is so large that even truthful estimates would
  confuse it with arm $*$. This requires $N_i < 8 \\log t / \\Delta_i^2$.</li>
</ol>
<p>Cases 1 and 2 happen with probability $\\leq 2t^{-4}$, summable. Case 3 caps
$N_i(T)$ at roughly $8 \\log T / \\Delta_i^2$. Multiply by $\\Delta_i$ and sum: regret bound.</p>

${forwardLink({
  destination: "Lesson 13 — Model-Based RL",
  html: `<p>The "optimism" principle is not just a bandit trick. <strong>R-MAX</strong> (Brafman &amp;
  Tennenholtz 2002) is optimism-under-uncertainty for full MDPs: assume unknown
  transitions yield maximum reward, then plan. Modern derivatives include <strong>UCB-VI</strong>
  (Azar et al. 2017) for tabular MDPs and OFU algorithms across deep RL. The recipe scales.</p>`,
})}

<p>The panel below is the clearest window into UCB. Each arm's bar is its confidence
interval; the dot is $\\hat{\\mu}_i$. The arm whose <em>upper</em> edge is highest gets pulled and
its bar snaps tighter, while the others widen a touch as $\\log t$ grows. Step through it
and read off the pick reason — it's exactly the argmax over upper edges.</p>

<ucb-confidence-bounds></ucb-confidence-bounds>`,
    );
  },
};
