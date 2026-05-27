import type { Section } from "../section";
import { mcMeta } from "./meta";
import { mcSection01 } from "./section-01-model-free";
import { mcSection02 } from "./section-02-fv-vs-ev";
import { mcSection03 } from "./section-03-mc-policy-eval";
import { mcSection04 } from "./section-04-mc-control";
import { mcSection05 } from "./section-05-off-policy";
import { mcSection06 } from "./section-06-bias-variance";
import { mcSection07 } from "./section-07-forward-links";

export const mcSections: Section[] = [
  mcSection01,
  mcSection02,
  mcSection03,
  mcSection04,
  mcSection05,
  mcSection06,
  mcSection07,
];

export { mcMeta };
