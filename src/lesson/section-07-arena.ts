import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section07: Section = {
  id: "bandits-empirical-comparison",
  title: "Empirical Comparison",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§7</span>Empirical Comparison</h2>
       <p class="tagline">Three philosophies, one regret curve at a time.</p>
       <p class="placeholder">[section 7 — centerpiece, to be filled in Step 6]</p>`,
    );
  },
};
