import type { Section } from "./section";
import { sectionFromHTML } from "./section";
import { sidebar } from "../components/CrosslinkCallout";

export const section03: Section = {
  id: "exploration-exploitation",
  title: "The Exploration–Exploitation Dilemma",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§3</span>The Exploration–Exploitation Dilemma</h2>
<p class="tagline">Why this isn't a normal optimization problem.</p>

<p>Bandits are deceptively close to a problem you've seen: estimate $K$ means from $K$
streams of i.i.d. samples. If you were given the budget $T$ and told "estimate the
$\\mu_i$ as accurately as possible," the answer is uninteresting — pull each arm $T/K$
times. The estimator with minimum variance is uniform allocation. But you weren't
given that objective. You were told to <em>maximize cumulative reward</em>. And the moment
you pull arm 1, you've spent budget that you can never spend on arm 3 — even though
arm 3 is, unbeknownst to you, the better arm.</p>

<p>This is the <strong>exploration–exploitation dilemma</strong>. Information has a price, and the
currency is reward. Every "wasted" pull on a known-bad arm is regret. Every pull on
the apparent-best arm forgoes information about the others.</p>

<p>The dilemma has a precise structural form: <strong>at each step, your decision depends on
the entire history</strong>. There is no Markov property here. (Wait — what? Doesn't RL
<em>love</em> the Markov property? It does, and we'll restore it in Lesson 2 by making
the <em>posterior</em> over $\\mu_i$ part of the state. For now, accept that bandit history
genuinely matters.)</p>

<p>Three coordinates locate any bandit algorithm:</p>

<ul>
  <li><strong>What estimate of $\\mu_i$ does it maintain?</strong> (Sample mean? Posterior mean? Median?)</li>
  <li><strong>How does it incorporate uncertainty?</strong> (Ignore it? Add a confidence bonus? Sample from the posterior?)</li>
  <li><strong>How does it convert estimates + uncertainty into an action choice?</strong></li>
</ul>

<p>ε-greedy, UCB, and Thompson sampling differ along all three axes, and yet — this is
the lesson's secret — they all achieve sub-linear regret. The space of "good"
algorithms is wide. The space of <em>near-optimal</em> algorithms is much narrower, and
that narrowing is what the next three sections are about.</p>

${sidebar(
  "Sidebar — Knowing what you don't know",
  `<p>UCB's confidence bonus and Thompson's posterior are two languages for the same
  idea: <em>uncertainty itself drives action</em>. This recurs in Bayesian deep RL
  (variational dropout in DQN), in safe RL (CVaR constraints), and most clearly in
  RL-as-inference (Lesson 11), where the entire optimal policy is derived as a
  posterior. Hold this thought.</p>`,
)}

<p>Before meeting the algorithms that get this right, look at what the two extremes
actually do. Both curves below are linear in $t$ — pure greedy locks onto a wrong arm
with constant probability, pure random pays the exploration tax forever. The
log-shaped Lai–Robbins floor sits far beneath both. The entire job of §4–§6 is to
get down to that floor.</p>

<two-extreme-failure></two-extreme-failure>`,
    );
  },
};
