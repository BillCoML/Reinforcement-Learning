/** Markov Chains lesson: ordered sections + metadata, consumed by the router. */
import type { Section } from "../section";
import { markovMeta } from "./meta";
import { mcSection01 } from "./section-01-markov-property";
import { mcSection02 } from "./section-02-multi-step";
import { mcSection03 } from "./section-03-classification";
import { mcSection04 } from "./section-04-stationary";
import { mcSection05 } from "./section-05-convergence";
import { mcSection06 } from "./section-06-detailed-balance";
import { mcSection07 } from "./section-07-bridge-to-rl";

export const markovSections: Section[] = [
  mcSection01,
  mcSection02,
  mcSection03,
  mcSection04,
  mcSection05,
  mcSection06,
  mcSection07,
];

export { markovMeta };
