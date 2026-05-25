import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection03: Section = {
  id: "communicating-classes",
  title: "Classifying States",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§3</span>Classifying States</h2>
<p class="tagline">Reachability, recurrence, periodicity — three independent axes.</p>
<p class="rl-scaffold-note">Section content arrives in build step 5.</p>`,
    );
  },
};
