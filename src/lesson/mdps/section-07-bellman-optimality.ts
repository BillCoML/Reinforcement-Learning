import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mdpSection07: Section = {
  id: "bellman-optimality",
  title: "Bellman Optimality",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§7</span>Bellman Optimality</h2>
<p class="tagline">Replace the sum-over-π with a max. Get V*. Greedy w.r.t. V* is optimal.</p>
<p class="scaffold-note">Prose and V7 (Optimality Explorer) land in Step 8.</p>`,
    );
  },
};
