import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const pgSection08: Section = {
  id: "pg-forward-links",
  title: "What Comes Next",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§8</span>What Comes Next</h2>
<p class="tagline"><em>Policy gradient opens the second tier: trust regions, entropy regularization, soft actor-critic, RLHF.</em></p>

<p>REINFORCE and one-step actor-critic are the foundation. Every modern deep RL algorithm
is a policy gradient variant. The open problems they leave motivate the next lessons:</p>

<p><strong>Problem 1: Step size sensitivity.</strong>
Gradient ascent without constraints can take catastrophically large steps —
collapsing the policy to a bad local optimum in one update.
Lesson 11 (TRPO/PPO) introduces trust regions and clipped objectives that
prevent destructive updates while preserving the sample efficiency of large batches.</p>

<p><strong>Problem 2: Entropy collapse.</strong>
The policy naturally sharpens over training, reducing exploration.
Lesson 12 (Max-Entropy RL / SAC) adds an entropy bonus $\\mathcal{H}[\\pi(\\cdot \\mid s)]$
to the objective, preventing premature collapse and enabling better exploration.</p>

<p><strong>Problem 3: Off-policy data.</strong>
On-policy policy gradient discards data after each update — sample-inefficient.
Lesson 13 (SAC) combines actor-critic with a replay buffer and twin Q-networks,
converting to off-policy updates with importance weights.</p>

<p><strong>Connection to RLHF.</strong>
The policy gradient methods developed here are the algorithmic core of RLHF.
In PPO-based alignment: the "policy" is the language model, the "critic" is a
separate value head, and the "reward" comes from a human preference model.
The same actor-critic update from §6 — with clipping — is applied to token distributions
at each step of the language model's forward pass.</p>

<roadmap-mini active="policy-gradient"></roadmap-mini>
`);
  },
};
