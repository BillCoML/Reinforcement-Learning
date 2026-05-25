import type { Section } from "./section";
import { sectionFromHTML } from "./section";
import { sidebar } from "../components/CrosslinkCallout";

export const section08: Section = {
  id: "bandits-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§8</span>Where You'll See This Again</h2>
<p class="tagline">Forward links — what bandits unlocks downstream.</p>

<p>Four threads from this lesson run through the rest of the curriculum.</p>

<p><strong>Thread 1: Bandits as a degenerate MDP.</strong> A bandit is a one-state MDP. The "policy"
is just a distribution over actions, since there's only one state to condition on.
This perspective unlocks Lesson 2 (MDPs) — you'll see Bellman equations collapse to
trivial identities in the bandit case, which is the gentlest possible introduction.</p>

<p><strong>Thread 2: Exploration in deep RL.</strong> ε-greedy is the workhorse exploration
strategy in deep RL precisely because of its portability (Lesson 7, DQN). UCB and
Thompson are harder to extend (Q-network confidence intervals are non-trivial; we'll
discuss bootstrapped DQN and Bayesian variants). The exploration–exploitation
question never goes away — it just gets harder.</p>

<p><strong>Thread 3: Posterior sampling and inference-RL.</strong> Thompson sampling's "act as if
your beliefs were true" is the seed of an enormous research thread. In Lesson 11
(RL as probabilistic inference) you'll see the entire optimal-policy derivation as
posterior sampling. SAC (Lesson 12) inherits this. PSRL (posterior sampling for RL)
generalizes Thompson to MDPs.</p>

<p><strong>Thread 4: Bandit-style preference learning in RLHF.</strong> In Lesson 17, we'll meet
the Bradley–Terry model — a <em>preference bandit</em> where each "pull" is a pairwise
comparison $(a, b)$, and the goal is to learn a reward model from preferences.
Many of the bandit primitives (confidence intervals, posterior sampling) reappear
with subtle modifications.</p>

${sidebar(
  "Sidebar — Contextual bandits",
  `<p>The natural generalization of "K arms" is "K arms, but the rewards depend on a
  feature vector $x_t$ observed each round." This is <strong>contextual bandit</strong> territory
  (LinUCB, Thompson sampling for linear/logistic models). Contextual bandits are
  extensively used in practice (news recommendation, ad placement, A/B testing).
  We'll mention them in passing in Lesson 2 but won't dedicate a lesson — they're a
  worthy side trip the learner can take once they have linear function approximation
  from Lesson 6.</p>`,
)}

<p>Here's the map. This lesson sits at the root; the four arrows are the threads above.
Markov Chains (Prereq A) and Markov Decision Processes (Lesson 2) are built — click them to
jump across; the rest aren't written yet, so hover a node to see the connection.</p>

<roadmap-mini active="bandits"></roadmap-mini>`,

    );
  },
};
