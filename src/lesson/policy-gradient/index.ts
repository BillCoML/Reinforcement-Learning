import type { Section } from "../section";
import { pgMeta } from "./meta";
import { pgSection01 } from "./section-01-policy-space";
import { pgSection02 } from "./section-02-score-function";
import { pgSection03 } from "./section-03-pg-theorem";
import { pgSection04 } from "./section-04-reinforce";
import { pgSection05 } from "./section-05-variance-reduction";
import { pgSection06 } from "./section-06-actor-critic";
import { pgSection07 } from "./section-07-advantage";
import { pgSection08 } from "./section-08-forward-links";

export { pgMeta };

export const pgSections: Section[] = [
  pgSection01,
  pgSection02,
  pgSection03,
  pgSection04,
  pgSection05,
  pgSection06,
  pgSection07,
  pgSection08,
];
