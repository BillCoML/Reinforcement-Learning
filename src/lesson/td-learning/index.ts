import type { Section } from "../section";
import { tdMeta } from "./meta";
import { tdSection01 } from "./section-01-bootstrap";
import { tdSection02 } from "./section-02-td-zero";
import { tdSection03 } from "./section-03-sarsa";
import { tdSection04 } from "./section-04-q-learning";
import { tdSection05 } from "./section-05-n-step";
import { tdSection06 } from "./section-06-td-lambda";
import { tdSection07 } from "./section-07-convergence";
import { tdSection08 } from "./section-08-forward-links";

export const tdSections: Section[] = [
  tdSection01,
  tdSection02,
  tdSection03,
  tdSection04,
  tdSection05,
  tdSection06,
  tdSection07,
  tdSection08,
];

export { tdMeta };
