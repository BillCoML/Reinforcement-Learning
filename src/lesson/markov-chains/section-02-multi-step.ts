import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection02: Section = {
  id: "n-step-transitions",
  title: "Multi-Step Transitions and Long-Run Behaviour",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§2</span>Multi-Step Transitions and Long-Run Behaviour</h2>
<p class="tagline">Powers of a matrix encode the future.</p>
<p class="rl-scaffold-note">Section content arrives in build step 5.</p>`,
    );
  },
};
