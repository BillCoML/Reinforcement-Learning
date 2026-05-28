import type { Section } from "../section";
import { ppoMeta } from "./meta";
import { ppoSection01 } from "./section-01-step-size-problem";
import { ppoSection02 } from "./section-02-trust-region-kl";
import { ppoSection03 } from "./section-03-trpo";
import { ppoSection04 } from "./section-04-ppo-clipped";
import { ppoSection05 } from "./section-05-gae-full";
import { ppoSection06 } from "./section-06-ppo-lab";
import { ppoSection07 } from "./section-07-empirics";
import { ppoSection08 } from "./section-08-forward-links";

export { ppoMeta };

export const ppoSections: Section[] = [
  ppoSection01,
  ppoSection02,
  ppoSection03,
  ppoSection04,
  ppoSection05,
  ppoSection06,
  ppoSection07,
  ppoSection08,
];
