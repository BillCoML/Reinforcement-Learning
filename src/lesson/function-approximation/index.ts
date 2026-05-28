import type { Section } from "../section";
import { dqnMeta } from "./meta";
import { dqnSection01 } from "./section-01-tabular-limits";
import { dqnSection02 } from "./section-02-linear-fa";
import { dqnSection03 } from "./section-03-deadly-triad";
import { dqnSection04 } from "./section-04-neural-networks";
import { dqnSection05 } from "./section-05-dqn-recipe";
import { dqnSection06 } from "./section-06-stability-lab";
import { dqnSection07 } from "./section-07-dqn-family";
import { dqnSection08 } from "./section-08-forward-links";

export { dqnMeta };

export const dqnSections: Section[] = [
  dqnSection01,
  dqnSection02,
  dqnSection03,
  dqnSection04,
  dqnSection05,
  dqnSection06,
  dqnSection07,
  dqnSection08,
];
