import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection01: Section = {
  id: "markov-property",
  title: "The Markov Property",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§1</span>The Markov Property</h2>
<p class="tagline">Memorylessness, with one caveat.</p>
<p class="rl-scaffold-note">Section content arrives in build step 5.</p>`,
    );
  },
};
