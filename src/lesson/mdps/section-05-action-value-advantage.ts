import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection05: Section = {
  id: "action-value-function",
  title: "Action Value Qπ and Advantage Aπ",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§5</span>Action Value <span class="rl-mono">Q<sup>π</sup></span> and Advantage <span class="rl-mono">A<sup>π</sup></span></h2>
<p class="tagline">V, conditioned on the first move. And how much better that move is than average.</p>
<p class="scaffold-note">Prose and V5 (Q-Quadrants and Advantage) land in Step 6.</p>`,
    );
  },
};
