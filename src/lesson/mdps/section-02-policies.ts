import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection02: Section = {
  id: "policy",
  title: "Policies",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§2</span>Policies</h2>
<p class="tagline">A policy is a rule for living.</p>
<p class="scaffold-note">Prose and V2 (Policy Explorer) land in Step 5.</p>`,
    );
  },
};
