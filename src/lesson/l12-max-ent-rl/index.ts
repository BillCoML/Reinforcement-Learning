import type { Section } from "../section";
import { maxentMeta } from "./meta";
import { maxentSection01 } from "./section-01-why-stochastic";
import { maxentSection02 } from "./section-02-objective";
import { maxentSection03 } from "./section-03-soft-bellman";
import { maxentSection04 } from "./section-04-boltzmann-policy";
import { maxentSection05 } from "./section-05-failure-mode";
import { maxentSection06 } from "./section-06-entropy-slider-lab";
import { maxentSection07 } from "./section-07-rl-as-inference";
import { maxentSection08 } from "./section-08-forward-links";

export { maxentMeta };

export const maxentSections: Section[] = [
  maxentSection01,
  maxentSection02,
  maxentSection03,
  maxentSection04,
  maxentSection05,
  maxentSection06,
  maxentSection07,
  maxentSection08,
];
