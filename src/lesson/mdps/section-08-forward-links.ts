import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection08: Section = {
  id: "mdps-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§8</span>Where You'll See This Again</h2>
<p class="tagline">Every algorithm in the RL canon solves, generalizes, or learns one of these equations.</p>
<p class="scaffold-note">Prose and V8 (Roadmap Mini update) land in Step 9.</p>`,
    );
  },
};
