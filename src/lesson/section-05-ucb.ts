import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section05: Section = {
  id: "ucb1",
  title: "UCB: Optimism Under Uncertainty",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§5</span>UCB: Optimism Under Uncertainty</h2>
       <p class="tagline">When in doubt, act as if your best guess is the upper bound.</p>
       <p class="placeholder">[section 5 — to be filled in Step 5]</p>`,
    );
  },
};
