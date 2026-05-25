import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section04: Section = {
  id: "epsilon-greedy",
  title: "ε-Greedy and Friends",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§4</span>ε-Greedy and Friends</h2>
       <p class="tagline">Exploration as a constant tax.</p>
       <p class="placeholder">[section 4 — to be filled in Step 5]</p>`,
    );
  },
};
