import type { Section } from "./section";
import { sectionFromHTML } from "./section";
import { forwardLink } from "../components/CrosslinkCallout";

export const section04: Section = {
  id: "epsilon-greedy",
  title: "ε-Greedy and Friends",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§4</span>ε-Greedy and Friends</h2>
<p class="tagline">Exploration as a constant tax.</p>

<p>The simplest fix for greedy's failure is to explicitly budget exploration. At each
step, with probability $1 - \\varepsilon$ pull the empirically-best arm; with probability $\\varepsilon$
pull a uniformly random arm.</p>

<h3>Algorithm (ε-greedy)</h3>
<pre><code>for t = 1, 2, …, T:
    if t ≤ K:           # initialization: pull each arm once
        a_t = t
    elif rand() &lt; ε:
        a_t = uniform({1, …, K})
    else:
        a_t = argmax_i μ̂_i,t-1
    pull a_t, observe R_t, update N_{a_t} and μ̂_{a_t}</code></pre>

<p>This works in the sense that it doesn't lock onto the wrong arm. The empirical means
$\\hat{\\mu}_i = (1/N_i) \\sum_{s: a_s = i} R_s$ are unbiased and consistent. As $N_i \\to \\infty$,
$\\hat{\\mu}_i \\to \\mu_i$ a.s., and eventually $\\arg\\max_i \\hat{\\mu}_i = \\arg\\max_i \\mu_i = i^*$.</p>

<p>But the regret is <strong>linear in $T$</strong>. Here's why. The probability of exploring on any
given step is $\\varepsilon$. Conditional on exploring, the agent picks a uniformly random arm,
which is suboptimal with probability $(K-1)/K$. So the expected per-step regret
from exploration alone is at least</p>

$$\\varepsilon \\cdot \\frac{1}{K} \\sum_i \\Delta_i \\;=\\; \\varepsilon \\cdot \\bar{\\Delta}
\\;\\;\\Rightarrow\\;\\; R_T \\gtrsim \\varepsilon \\cdot \\bar{\\Delta} \\cdot T.$$

<p>For our running example with $\\varepsilon = 0.1$, $K = 3$, $\\sum \\Delta_i = 0.6$:</p>

$$\\text{per-step regret rate} \\;\\geq\\; 0.1 \\cdot \\frac{0.6}{3} \\;=\\; 0.02$$

<p>So at $T = 10{,}000$, ε-greedy with constant $\\varepsilon = 0.1$ accumulates at least ~200
regret from exploration alone. Empirically, over many seeds, we measure
$R_{5000} \\approx 111$ — comfortably above the Lai–Robbins floor of $29.6$.</p>

<h3>The fix: decay ε</h3>
<p>If $\\varepsilon_t \\to 0$ slowly enough, the exploration tax vanishes
asymptotically but you've still explored enough to find the best arm. The classic
schedule (Auer, Cesa-Bianchi &amp; Fischer 2002) is $\\varepsilon_t = \\min(1, cK / (d^2 t))$ for
suitable constants. With this schedule ε-greedy achieves $O(\\log T)$ regret — though
the constants are worse than UCB's, and the schedule depends on knowing the gap
$d \\approx \\min \\Delta_i$, which is uncomfortable.</p>

<p>A softer cousin is <strong>Boltzmann exploration</strong> (a.k.a. softmax):</p>

$$\\Pr(a_t = i) \\;=\\; \\frac{\\exp(\\hat{\\mu}_i / \\tau)}{\\sum_j \\exp(\\hat{\\mu}_j / \\tau)}$$

<p>with temperature $\\tau > 0$. At $\\tau \\to 0$ this is pure greedy; at $\\tau \\to \\infty$ it's uniform.
Boltzmann exploration is a useful pedagogical bridge because it foreshadows
<strong>maximum-entropy RL</strong> (Lesson 10), where this same softmax becomes the
<em>optimal</em> policy under entropy-regularized objectives. For Lesson 1, file Boltzmann
under "honourable mention" — it's a heuristic, not a principled algorithm.</p>

${forwardLink({
  destination: "Lesson 10 — Max-Entropy RL",
  html: `<p>Boltzmann exploration is not just a bandit heuristic. The Bellman equation for
  entropy-regularized RL makes the softmax policy <em>exactly optimal</em>, not approximately
  so. The temperature $\\tau$ becomes the entropy coefficient. The same formula, twice —
  once heuristic, once principled.</p>`,
})}

<p><strong>When to use ε-greedy in practice.</strong> Almost never as your final algorithm if you
have any other option. But it is the workhorse of deep RL (DQN uses ε-greedy with
linear decay, Lesson 7) because it generalizes trivially to function approximation:
"argmax over actions, with probability $\\varepsilon$ take a random one." UCB and Thompson are
harder to lift into deep nets. So ε-greedy survives in deep RL not because it's
good but because it's portable.</p>

<p>Run it below. Watch the regret curve climb at a near-constant rate (the exploration
tax), the pull frequencies settle with arm 3 dominating, and the timeline flicker
with the occasional random exploration pull. Try switching to the decay schedule to
see the curve bend toward the floor.</p>

<epsilon-greedy-explorer></epsilon-greedy-explorer>`,
    );
  },
};
