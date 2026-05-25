import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section01: Section = {
  id: "the-bandit-problem",
  title: "The Bandit Problem",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§1</span>The Bandit Problem</h2>
<p class="tagline">K levers, hidden distributions, one objective.</p>

<p>You stand in front of $K$ slot machines. Each machine $i \\in \\{1, \\dots, K\\}$ pays out a
random reward $R$ drawn from some fixed distribution $\\nu_i$ with mean $\\mu_i$. You
don't know any of the $\\mu_i$. You have $T$ pulls to spend. After your last pull,
you'll be paid the total reward you accumulated. What do you do?</p>

<p>This is the <strong>multi-armed bandit</strong> problem. It's a caricature, but it is <em>exactly</em>
the right caricature: it isolates the single thing that distinguishes reinforcement
learning from supervised learning, which is that you don't get to see the answer to
a question you didn't ask. If you pull arm 1, you find out something about arm 1.
You learn nothing about arms 2 through $K$. The reward of the action <em>not taken</em> is
gone forever — not in the universe, just not in your data.</p>

<p>Two extreme strategies fail immediately. <strong>Pure exploitation:</strong> pull each arm once,
then pull whichever produced the highest reward for the remaining $T-K$ rounds.
With Bernoulli rewards and small $K$, this strategy will <em>lock onto the wrong arm
with constant probability</em> — its loss grows linearly in $T$. <strong>Pure exploration:</strong>
pull arms uniformly at random forever. You learn the $\\mu_i$ precisely in the limit,
but you only earn the <em>average</em> arm's reward along the way. Loss again linear in
$T$. The fact that both extremes fail with the same scaling tells us something
non-trivial: the right strategy must <em>interleave</em> exploration and exploitation,
and the schedule of that interleaving is the algorithm.</p>

<p>We'll fix one running example throughout this lesson:</p>

<blockquote>
  <span class="label">Running example</span>
  <p>Three Bernoulli arms with means $\\mu = (0.3,\\, 0.5,\\, 0.7)$. The optimal arm is
  arm 3 with $\\mu^* = 0.7$. The <strong>suboptimality gaps</strong> are
  $\\Delta_1 = 0.4$, $\\Delta_2 = 0.2$, $\\Delta_3 = 0$.</p>
</blockquote>

<p>This is small enough to reason about by hand and rich enough to display the full
behavioural repertoire of every algorithm we'll meet.</p>

<p>A few non-obvious things to notice. First, the problem is <strong>stationary</strong> — $\\nu_i$
doesn't change over time. Bandits in the wild are often non-stationary; that's a
real complication and we'll mention it in §8, but our analysis assumes stationarity.
Second, the <strong>horizon $T$ is known</strong>. This isn't always true either. UCB will turn
out to be horizon-free, which is a nice property. Third, <strong>rewards are observed
immediately</strong> after the action. In full RL (Lesson 2 onward) rewards can be delayed
arbitrarily, and that delay is most of the difficulty.</p>

<p>For now, ignore those generalizations. One state, $K$ actions, immediate stochastic
reward. The thing we want to understand is: <em>given that you must explore to learn
and you must exploit to earn, what's the right schedule, and how do we measure
whether a schedule is any good?</em></p>

<bandit-machine means="0.3,0.5,0.7"></bandit-machine>
<p style="font-size:15px;color:var(--rl-ink-muted)">A warm-up. Pull the levers a few times and feel the
variance — then reveal the hidden means and see how far your empirical estimates
have drifted from the truth.</p>`,
    );
  },
};
