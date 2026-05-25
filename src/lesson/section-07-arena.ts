import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section07: Section = {
  id: "bandits-empirical-comparison",
  title: "Empirical Comparison",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§7</span>Empirical Comparison</h2>
<p class="tagline">Three philosophies, one regret curve at a time.</p>

<p>We have three algorithms and one running example. Let's run them.</p>

<p>The plot below shows pseudo-regret $R_t$ averaged over independent seeds, for
$t = 1\\dots5000$, on the Bernoulli bandit with $\\mu = (0.3, 0.5, 0.7)$. Final-regret
numbers (from our offline simulation, embedded directly):</p>

<table>
  <thead><tr><th>Algorithm</th><th>$R_{5000}$ (avg)</th><th>shape</th><th>comment</th></tr></thead>
  <tbody>
    <tr><td>Random</td><td>~1000</td><td>linear</td><td>exploration tax forever</td></tr>
    <tr><td>ε-greedy (ε=0.10)</td><td>110.7</td><td>linear (slow)</td><td>constant tax ≈ 0.022/step</td></tr>
    <tr><td>ε-greedy (ε=0.01)</td><td>95.3</td><td>mixed</td><td>better tax but fragile lock-in</td></tr>
    <tr><td>UCB1</td><td>82.3</td><td>log-shaped</td><td>matches theory</td></tr>
    <tr><td>Thompson sampling</td><td>16.6</td><td>log-shaped</td><td>near Lai–Robbins floor (29.6)</td></tr>
    <tr class="sum-row"><td>Lai–Robbins LB</td><td>29.6</td><td>log</td><td>the floor</td></tr>
  </tbody>
</table>

<p>Three observations worth highlighting:</p>

<ol>
  <li><strong>UCB1's regret is log-shaped but with a noticeably worse constant than
  Thompson's.</strong> That's a real phenomenon, not noise. UCB1's confidence bonus uses
  Hoeffding, which is loose for Bernoulli (the variance is $\\mu(1-\\mu)$, often well below
  the $1/4$ worst case Hoeffding assumes). <strong>KL-UCB</strong> (Garivier &amp; Cappé, 2011) closes
  the gap by replacing Hoeffding with a Bernoulli-KL confidence interval. For this
  lesson we stick with UCB1; KL-UCB is a footnote.</li>

  <li><strong>Thompson sampling beats the Lai–Robbins floor at finite $T$.</strong> That's not a
  violation — the floor is asymptotic. Thompson is below the asymptote in finite-$T$
  regimes; the relationship is about the <em>slope</em> $R_T / \\log T$ as $T \\to \\infty$.</li>

  <li><strong>ε-greedy with smaller ε wins eventually but loses early.</strong> With ε = 0.01 the
  exploration tax is tiny but the initial commitment is fragile — one unlucky
  early streak can lock the algorithm on a bad arm for a long time (its regret has a
  <em>fat tail</em>: std ≈ 160 at $T=5000$, vs ≈ 25 for ε = 0.10). Picking ε is a hyperparameter
  nightmare. <strong>UCB and Thompson have no equivalent knob</strong>, which is most of why
  they're preferred in practice.</li>
</ol>

<p>When you reach for an algorithm in the wild: <strong>Thompson first</strong> (if you can do
posterior sampling), <strong>UCB1 if you can't and want guarantees</strong>, <strong>ε-greedy if you
need to wire it into deep nets and don't care about a constant factor of 2–5×</strong>.</p>

<p>Now stop reading and play. Change the gaps, add arms, switch reward families,
randomize the problem, crank the horizon. Watch the regret curves separate — but
more importantly, watch the <strong>pull-distribution bars</strong> at the bottom. That's where the
philosophies become visible: Thompson concentrates pulls on the best arm
aggressively, UCB explores more evenly, and ε-greedy leaves a characteristic uniform
exploration smear across every arm, forever.</p>

<algorithm-battle-arena></algorithm-battle-arena>

<p style="font-size:15px;color:var(--rl-ink-muted)">Seeds and the algorithm roster are encoded in the URL
(<code>?seed=42&amp;algos=ucb,ts</code>) — copy the address bar to share an exact configuration.
"Export PNG" saves the current regret plot.</p>`,
    );
  },
};
