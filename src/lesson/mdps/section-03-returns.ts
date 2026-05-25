import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection03: Section = {
  id: "return-discount",
  title: "Returns and Discounting",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§3</span>Returns and Discounting</h2>
<p class="tagline">Adding up rewards over time, with a thumb on the scale.</p>
<p class="scaffold-note">Prose and V3 (Return Composer) land in Step 5.</p>`,
    );
  },
};
