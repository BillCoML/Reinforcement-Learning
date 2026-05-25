import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection06: Section = {
  id: "detailed-balance",
  title: "Detailed Balance (Reversibility)",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§6</span>Detailed Balance (Reversibility)</h2>
<p class="tagline">A sufficient condition for stationarity, and a glimpse of MCMC.</p>
<p class="rl-scaffold-note">Section content arrives in build step 6.</p>`,
    );
  },
};
