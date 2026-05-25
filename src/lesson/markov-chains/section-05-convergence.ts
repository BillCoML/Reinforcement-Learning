import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection05: Section = {
  id: "ergodic-theorem",
  title: "The Ergodic Theorem (Convergence)",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§5</span>The Ergodic Theorem (Convergence)</h2>
<p class="tagline">Irreducible + aperiodic = inevitable convergence, at a rate set by the spectral gap.</p>
<p class="rl-scaffold-note">Section content (the centerpiece) arrives in build step 6.</p>`,
    );
  },
};
