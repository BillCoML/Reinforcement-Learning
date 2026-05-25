import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection06: Section = {
  id: "bellman-expectation",
  title: "The Bellman Expectation Equations",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§6</span>The Bellman Expectation Equations</h2>
<p class="tagline">Vπ satisfies a recursive identity. Iterate the identity, get the value.</p>
<p class="scaffold-note">Prose and V6 (Bellman Backup Lab — centerpiece) land in Step 7.</p>`,
    );
  },
};
