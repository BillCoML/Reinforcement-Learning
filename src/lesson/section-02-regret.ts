import type { Section } from "./section";
import { sectionFromHTML } from "./section";
import { forwardLink } from "../components/CrosslinkCallout";

export const section02: Section = {
  id: "regret-definition",
  title: "Regret: The Right Yardstick",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§2</span>Regret: The Right Yardstick</h2>
<p class="tagline">We measure how much we lost by not knowing the truth.</p>

<p>Suppose at time $t$ the learner pulls arm $a_t$ and receives reward $R_t \\sim \\nu_{a_t}$.
The expected reward of the algorithm over $T$ rounds is
$\\mathbb{E}\\!\\left[\\sum_{t=1}^{T} R_t\\right] = \\sum_{t=1}^{T} \\mathbb{E}[\\mu_{a_t}]$,
where the outer expectation is over both the algorithm's randomness and the reward noise.</p>

<p>The natural-feeling objective is "maximize total reward." It's the wrong objective.
Maximum total reward is $T \\cdot \\mu^*$ — you can't beat always-pulling-the-best-arm. But
$T \\cdot \\mu^*$ depends on $\\mu^*$, which is a property of the problem, not the algorithm.
What we actually want to compare across algorithms (and across problem instances)
is <em>how close the algorithm gets to the oracle</em>.</p>

<p>Define the <strong>pseudo-regret</strong>:</p>

$$\\boxed{R_T \\;=\\; T \\cdot \\mu^* \\;-\\; \\mathbb{E}\\!\\left[\\sum_{t=1}^{T} \\mu_{a_t}\\right]}$$

<p>— the gap between the oracle's expected total reward and the algorithm's. (Some
authors define regret with the realized rewards $R_t$ instead of $\\mu_{a_t}$; the
difference is noise and washes out in expectation, but $\\mu_{a_t}$ gives cleaner
analysis. We'll use pseudo-regret throughout and just call it "regret" when context
is clear.)</p>

<p>A useful rewrite. Let $N_i(T)$ be the random number of times arm $i$ is pulled in
the first $T$ rounds, so $\\sum_i N_i(T) = T$. Then</p>

$$R_T \\;=\\; \\mathbb{E}\\!\\left[\\sum_{t=1}^{T} (\\mu^* - \\mu_{a_t})\\right]
       \\;=\\; \\sum_{i=1}^{K} \\Delta_i \\, \\mathbb{E}[N_i(T)],$$

<p>where $\\Delta_i = \\mu^* - \\mu_i \\geq 0$. This is the <strong>gap decomposition</strong>. It says: regret
equals the sum, over suboptimal arms, of the gap to optimal times the expected
number of times you pulled that suboptimal arm. The optimal arm contributes zero
regardless of how many times you pull it.</p>

<p>This decomposition is the whole game. Two algorithms can have wildly different
<em>behaviours</em> yet identical regret if they pull each suboptimal arm the same
expected number of times. And the lower bound on $\\mathbb{E}[N_i(T)]$ for any reasonable
algorithm is what gives us the famous logarithmic regret floor.</p>

<figure class="numeric">
  <figcaption>Numerical example (pre-verified)</figcaption>
  <p>Pulled arm 1 fifty times, arm 2 thirty times, arm 3 four hundred and twenty times
  over $T = 500$. With our running example $\\Delta = (0.4, 0.2, 0)$:</p>
  $$R_{500} \\;=\\; 0.4 \\cdot 50 + 0.2 \\cdot 30 + 0 \\cdot 420 \\;=\\; 26.0$$
  <p>If instead the algorithm had played arm 1 only twice and arm 2 only five times,
  spending the rest of its budget on arm 3:</p>
  $$R_{500} \\;=\\; 0.4 \\cdot 2 + 0.2 \\cdot 5 + 0 \\cdot 493 \\;=\\; 1.8$$
  <p style="margin-bottom:0">A 14× improvement in regret from a <em>concentration of pulls on the best arm</em>, without
  ever achieving infinite-precision knowledge of any $\\mu_i$.</p>
</figure>

<h3>The Lai–Robbins lower bound</h3>

<p>(Lai &amp; Robbins, 1985.) For any algorithm with <em>uniformly good</em> performance across
all bandit instances — meaning sub-polynomial regret on every instance — and any
instance with positive gaps,</p>

$$\\liminf_{T \\to \\infty} \\frac{R_T}{\\log T} \\;\\geq\\; \\sum_{i: \\Delta_i > 0} \\frac{\\Delta_i}{\\mathrm{KL}(\\nu_i \\,\\|\\, \\nu^*)}.$$

<p>For our running example (Bernoulli arms, $\\mu = (0.3, 0.5, 0.7)$):</p>

<table>
  <thead><tr><th>Arm</th><th>$\\Delta_i$</th><th>$\\mathrm{KL}(\\nu_i \\,\\|\\, \\nu^*)$</th><th>$\\Delta_i / \\mathrm{KL}_i$</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>0.40</td><td>0.338919</td><td>1.1802</td></tr>
    <tr><td>2</td><td>0.20</td><td>0.087177</td><td>2.2942</td></tr>
    <tr><td>3</td><td>0</td><td>—</td><td>—</td></tr>
    <tr class="sum-row"><td>sum</td><td></td><td></td><td>3.4744</td></tr>
  </tbody>
</table>

<p>So <strong>no algorithm</strong> can do better than $3.4744 \\cdot \\log T$ regret asymptotically on
this instance. At $T = 5000$ that's $\\approx 29.6$. At $T = 100000$ that's $\\approx 40.0$. The
lower bound is brutally tight: we'll see Thompson sampling come close.</p>

${forwardLink({
  destination: "Lesson 15 (offline RL) · Lesson 17 (RLHF)",
  html: `<p>Bernoulli KL appears here for the first time but is exactly the same quantity that
  drives sample-complexity bounds in offline RL and the Bradley–Terry preference model
  in RLHF. The KL divergence is doing structural work, not decoration.</p>`,
})}

<p>The picture below makes the decomposition tangible. Each suboptimal arm contributes a
red band that grows at rate $\\Delta_i$ per pull of that arm; the optimal arm's band is
invisible because $\\Delta_3 = 0$. The dashed line is the Lai–Robbins floor — scrub $t$
and switch traces to see which strategies stay near it and which run away from it.</p>

<regret-decomposition></regret-decomposition>`,
    );
  },
};
