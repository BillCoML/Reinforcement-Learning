import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection04: Section = {
  id: "stationary-distribution",
  title: "Stationary Distributions",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§4</span>Stationary Distributions</h2>
<p class="tagline">π solves π = πP. Existence is easy, uniqueness needs work.</p>
<p class="rl-scaffold-note">Section content arrives in build step 6.</p>`,
    );
  },
};
