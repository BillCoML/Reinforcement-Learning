/** MDPs lesson: ordered sections + metadata, consumed by the router. */
import type { Section } from "../section";
import { mdpMeta } from "./meta";
import { mdpSection01 } from "./section-01-mdp-tuple";
import { mdpSection02 } from "./section-02-policies";
import { mdpSection03 } from "./section-03-returns";
import { mdpSection04 } from "./section-04-state-value";
import { mdpSection05 } from "./section-05-action-value-advantage";
import { mdpSection06 } from "./section-06-bellman-expectation";
import { mdpSection07 } from "./section-07-bellman-optimality";
import { mdpSection08 } from "./section-08-forward-links";

export const mdpSections: Section[] = [
  mdpSection01,
  mdpSection02,
  mdpSection03,
  mdpSection04,
  mdpSection05,
  mdpSection06,
  mdpSection07,
  mdpSection08,
];

export { mdpMeta };
