import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section02: Section = {
  id: "regret-definition",
  title: "Regret: The Right Yardstick",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§2</span>Regret: The Right Yardstick</h2>
       <p class="tagline">We measure how much we lost by not knowing the truth.</p>
       <p class="placeholder">[section 2 — to be filled in Step 4]</p>`,
    );
  },
};
