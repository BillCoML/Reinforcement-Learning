import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection04: Section = {
  id: "state-value-function",
  title: "State Value Function Vπ",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§4</span>State Value Function <span class="rl-mono">V<sup>π</sup></span></h2>
<p class="tagline">Expected return, given that you start here and follow π.</p>
<p class="scaffold-note">Prose and V4 (Value Heatmap) land in Step 6.</p>`,
    );
  },
};
