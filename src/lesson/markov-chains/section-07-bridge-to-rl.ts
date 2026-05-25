import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection07: Section = {
  id: "policy-induced-chain",
  title: "The Bridge to RL",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§7</span>The Bridge to RL</h2>
<p class="tagline">A policy turns an MDP into a Markov chain, and everything we just learned applies.</p>
<p class="rl-scaffold-note">Section content arrives in build step 7.</p>`,
    );
  },
};
