import type { Section } from "../section";
import { contractionsMeta } from "./meta";
import { contrSection01 } from "./section-01-metric-spaces";
import { contrSection02 } from "./section-02-contractions";
import { contrSection03 } from "./section-03-banach-theorem";
import { contrSection04 } from "./section-04-counterexamples";
import { contrSection05 } from "./section-05-bellman-contractions";
import { contrSection06 } from "./section-06-forward-links";

export const contractionsSections: Section[] = [
  contrSection01,
  contrSection02,
  contrSection03,
  contrSection04,
  contrSection05,
  contrSection06,
];

export { contractionsMeta };
