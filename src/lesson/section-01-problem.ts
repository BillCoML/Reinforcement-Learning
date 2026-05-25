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
       <p class="placeholder">[section 1 — to be filled in Step 4]</p>`,
    );
  },
};
