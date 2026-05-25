import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section08: Section = {
  id: "bandits-forward-links",
  title: "Where You'll See This Again",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§8</span>Where You'll See This Again</h2>
       <p class="tagline">Forward links — what bandits unlocks downstream.</p>
       <p class="placeholder">[section 8 — to be filled in Step 7]</p>`,
    );
  },
};
