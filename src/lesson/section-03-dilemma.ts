import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section03: Section = {
  id: "exploration-exploitation",
  title: "The Exploration–Exploitation Dilemma",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§3</span>The Exploration–Exploitation Dilemma</h2>
       <p class="tagline">Why this isn't a normal optimization problem.</p>
       <p class="placeholder">[section 3 — to be filled in Step 4]</p>`,
    );
  },
};
