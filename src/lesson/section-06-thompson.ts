import type { Section } from "./section";
import { sectionFromHTML } from "./section";

export const section06: Section = {
  id: "thompson-sampling",
  title: "Thompson Sampling: Posterior in Action",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§6</span>Thompson Sampling: Posterior in Action</h2>
       <p class="tagline">Pull as if your beliefs were true.</p>
       <p class="placeholder">[section 6 — to be filled in Step 5]</p>`,
    );
  },
};
