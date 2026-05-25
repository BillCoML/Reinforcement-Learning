import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection01: Section = {
  id: "mdp-tuple",
  title: "From Bandits + Chains to MDPs",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§1</span>From Bandits + Chains to MDPs</h2>
<p class="tagline">Add a control knob to a Markov chain. Watch it become decision-making.</p>
<p class="scaffold-note">Prose and V1 (MDP Anatomy Explorer) land in Step 5.</p>`,
    );
  },
};
